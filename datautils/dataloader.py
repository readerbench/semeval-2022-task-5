import torch
import torchvision
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from matplotlib import pyplot as plt, patches
from transformers import BertTokenizer
from functools import partial
from sutime import SUTime
import re

OBJECT_VOCAB_SIZE = 1600
gcn_conf = None

def prepare_loaders(config):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    dloader = partial(DataLoader,  batch_size=config['batch_size'], collate_fn=semeval_collate_fn)

    config['tokenizer'] = tokenizer
    config['train_loader'] = dloader(semeval_data_loader('train', config, tokenizer))
    config['validate_loader'] = dloader(semeval_data_loader('validate', config, tokenizer))
    config['test_loader'] = dloader(semeval_data_loader('test', config, tokenizer))

    return config, tokenizer

def semeval_data_loader(dataset:str='train', config=None, tokenizer=None):
    assert dataset in ['train', 'validate', 'test'], 'Allowed values for dataset: train, validate, test'
    # to device!
    if (Path(config['data_path']) / (dataset + ".csv")).exists():
        return SemevalDataset(
            Path(config['data_path']) / (dataset + ".csv"),
            Path(config['data_path']) / "img",
            config['feature_path'], 
            tokenizer, 
            config['max_txt_len'],
            shuffle=(dataset=='train'),
            min_obj_confidence=config['min_object_conf'],
            use_nms=config['use_nms'],
            nms_threshold=config['nms_threshold'],
            nr_classes=config['nr_classes'],
            text_cleanup=config['with_cleanup'],
            sentiment_path=config['sentiment_feature_path'],
            use_gcn=config['with_gcn'],
            gcn_embedding_dim=config['gcn_embedding_dim'],
        )
    else:
        return None
    

class SemevalDataset(Dataset):
    def __init__(self, csv_file, image_folder, features_folder, tokenizer, max_length, *, shuffle=True, nr_classes=2,
             obj_vocab=None, use_nms=False, nms_threshold=0.7, min_obj_confidence=0.0, ignore_objects=None, text_cleanup=False,
             sentiment_path=None, use_gcn=False, gcn_embedding_dim=0 ):
        self.csv_file = csv_file
        self.image_folder = Path(image_folder)
        self.features_folder = Path(features_folder)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.obj_vocab = obj_vocab
        self.min_obj_confidence = min_obj_confidence
        self.ignore_objects = ignore_objects
        self.use_nms = use_nms
        self.use_gcn = use_gcn
        self.nms_threshold = nms_threshold
        self.nr_classes = nr_classes

        self.df = pd.read_csv(self.csv_file, sep="\t")

        if self.nr_classes==4:
            # Classifying only positive records
            self.df = self.df[self.df['misogynous']==1]

        if shuffle:
            self.df = self.df.sample(frac=1.0)
        
        self.text_cleanup = text_cleanup
        if self.text_cleanup:
            self.sutime = SUTime(mark_time_ranges=True, include_range=True)
        self.sentiment_path = sentiment_path
        self.gcn_embedding_dim = gcn_embedding_dim


    def __len__(self):
        return len(self.df)

    def cleanup_text(self, text):
        def remove_strings(text):
            garbage_strings = ["Twitter for iPhone", "VERY DEMOTIVATIONAL.com", "made with mematic","Meme Center","MemeCenter"]
            
            text = re.sub("@(\w){1,15}", "", text) # Twitter Usernames

            rep = [re.escape(x) for x in garbage_strings]
            
            text = re.sub("([A-Za-z0-9]\.|[A-Za-z0-9][A-Za-z0-9-]{0,61}[A-Za-z0-9]\.){1,3}(com|net|org|co\.uk|co|ru|eu)", "", text, flags=re.IGNORECASE)    

            pattern = re.compile("|".join(rep), flags=re.IGNORECASE)
            text = pattern.sub("", text)
            text = re.sub("\s\s+"," ", text) # Multiple spaces
            return text


        def remove_timestamps(sutime, text):
            date_seq = sutime.parse(text)
            res = text
            for seq in date_seq:
                if seq['type'] in ['DATE', 'TIME', 'DURATION']:
                    res = text[:seq['start']] + " "*(seq['end'] - seq['start']) + text[seq['end']:]
            if len(date_seq):
                res = " ".join(res.split())
            return res

        text = remove_timestamps(self.sutime, text)
        # text = remove_strings(text)

        return text

    def __getitem__(self, item):
        record = self.df.iloc[item]
        data_id = record.file_name.lower().replace('.jpg','')

        text = record['Text Transcription']
        if self.text_cleanup:
            text = self.cleanup_text(text)

        tokens = self.tokenizer.encode(text,  max_length=self.max_length, truncation=True, padding='do_not_pad',return_tensors='pt').squeeze()

        features, pos_features, objects, _ = self._load_img_feature(data_id)
        nr_bboxes = features.shape[0]
        if 'misogynous' in record:
            # Training or Validation
            if self.nr_classes==2:
                label = torch.tensor(record.misogynous)
                label = torch.nn.functional.one_hot(label, num_classes=2)
            elif self.nr_classes==4:
                label = torch.tensor([
                        record.shaming,
                        record.stereotype,
                        record.objectification,
                        record.violence])
            else:
                label = torch.tensor([
                        1 - record.misogynous,
                        record.shaming,
                        record.stereotype,
                        record.objectification,
                        record.violence])
        else:
            label = None
        
        if self.use_gcn:
            gcn_ids = [-1]+tokens.tolist()[1:-1]+[-1]
            gcn_ids.extend([int(x)+self.tokenizer.vocab_size for x in objects])
            # gcn_ids = gcn_ids[:self.gcn_embedding_dim]

        attn_mask = torch.ones(len(tokens) + nr_bboxes, dtype=torch.long)


        res = dict(            
            id=data_id,
            tokens=tokens,
            image_features=features,
            image_pos_features=pos_features,
            attn_mask=attn_mask,
            text=text,
            label=label,
        )
        if self.sentiment_path:
            feature_file_name = Path(self.sentiment_path) / f"{data_id}.npy"
            sent_feature = np.load(str(feature_file_name))
            sent_feature = torch.Tensor(sent_feature)

            res.update({'sent_features':sent_feature})
        if self.use_gcn:
            res.update({'gcn_tokens':gcn_ids})

        return res

    def _load_img_feature(self, image_id, normalize=False):
        feature_file_name = self.features_folder / f"{image_id}.npz"
        data = np.load(str(feature_file_name), allow_pickle=True)
        data_info = data['info'].item()

        img_width = data_info['image_w']
        img_height = data_info['image_h']

        w, h = data['bbox'][:,2]-data['bbox'][:,0], data['bbox'][:,3]-data['bbox'][:,1]

        objects = data_info['objects_id']
        objects_conf = data_info['objects_conf']

        if normalize:
            data['bbox'] = data['bbox'] / np.array([img_width, img_height, img_width, img_height])
        img_pos_feat = np.column_stack((data['bbox'], w, h, w * h))
        img_pos_feat = torch.tensor(img_pos_feat)
        img_feat = torch.tensor(data['x'])

        valid_boxes = (objects_conf > self.min_obj_confidence)

        img_feat = img_feat[valid_boxes]
        img_pos_feat = img_pos_feat[valid_boxes]
        objects = objects[valid_boxes]
        objects_conf = objects_conf[valid_boxes] 

        if self.use_nms and len(objects):
            obj_map = {}

            for i, obj_id in enumerate(objects):
                if obj_id in obj_map:
                    obj_map[obj_id].append(i)
                else:
                    obj_map[obj_id] = [i]

            for obj_id in obj_map:
                boxes = img_pos_feat[obj_map[obj_id]]
                boxes = torch.column_stack(
                    [
                    boxes[:, 0],
                    boxes[:, 1],
                    boxes[:, 0]+boxes[:, 2],
                    boxes[:, 1]+boxes[:, 3]
                    ])
                
                confidences = torch.tensor(objects_conf[obj_map[obj_id]])

                keep = torchvision.ops.nms(boxes, confidences, self.nms_threshold)
                obj_map[obj_id] = np.take(obj_map[obj_id], keep)
            
            indices = np.concatenate(list(obj_map.values()))            
            img_feat = img_feat[indices]
            img_pos_feat = img_pos_feat[indices]
            objects = objects[indices]
            objects_conf = objects_conf[indices] 



        return img_feat, img_pos_feat, objects, objects_conf

    def show_image(self, idx=None, data_id=None, img=None, with_features=False, min_confidence=0.0):
        """
        Displays an image of the dataset. At least one of the following input arguments needs to be given:
            idx - Index of the image in the dataset
            data_id - ID of the data point with the image (stated in the original .jsonl file)
            img - PIL image to display
        """
        assert  idx is not None or data_id or img, "Must specify at least one image identifier"

        label = None
        rects = []

        if idx is not None: # idx==0 is legit
            record = self.df.iloc[idx]
            data_id = record.file_name.lower().replace('.jpg','')
            label = record.misogynous

        if data_id:
            with open(self.image_folder / f"{data_id}.jpg","rb") as f:
                img = Image.open(f).convert("RGB")
            label = self.df[self.df['file_name']==f"{data_id}.jpg"].misogynous if label is None else label

                    

        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(img)

        if with_features:
            obj_names = {}
            if self.obj_vocab:
                with open(self.obj_vocab, "r") as f:
                        obj_names = {i:name.split(',')[0].strip() for i,name in enumerate(f.readlines())}
            feat, feat_pos, obj, obj_c = self._load_img_feature(image_id=data_id) 
            for i, rect in enumerate(feat_pos):
                if obj_c[i]>=min_confidence:
                    obj_id = obj[i]
                    x, y, _, _, w, h, _ = rect
                    ax.add_patch(patches.Rectangle((x,y), w, h, fill=False, edgecolor='red', lw=2))
                    ax.text(x,(y-5),f"{obj_c[i]:.2f}",verticalalignment='top', color='white', fontsize=15, weight='bold')
                    if obj_id in obj_names:
                        ax.text(x+w-40,(y-5),obj_names[obj_id],verticalalignment='top', color='magenta', fontsize=15, weight='bold')


        if label is not None:
            plt.title(
                "Label: %s (%i)" % ("Positive" if label == 1 else "Negative", label))

def semeval_collate_fn(inputs):
    
    gcn_tokens = None
    sent_features = None

    if 'gcn_tokens' in inputs[0]:
        (file_ids, token_ids, img_feats, img_pos_feats, 
            attn_masks, texts, labels, gcn_tokens) = list(zip(*[d.values() for d in inputs]))
    elif 'sent_features' in inputs[0]:
        (file_ids, token_ids, img_feats, img_pos_feats, 
            attn_masks, texts, labels, sent_features) = list(zip(*[d.values() for d in inputs]))
    else:
        (file_ids, token_ids, img_feats, img_pos_feats, 
            attn_masks, texts, labels) = list(zip(*[d.values() for d in inputs]))

    txt_lens, num_bbs = list(map(len,token_ids)), list(map(len,img_feats))

    input_ids = pad_sequence(token_ids, batch_first=True)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    img_feats = pad_sequence(img_feats, batch_first=True)
    img_pos_feats = pad_sequence(img_pos_feats, batch_first=True)
    attn_masks  = pad_sequence(attn_masks, batch_first=True)

    out_size = attn_masks.size(1)       # Max joint input size (features+texttokens)
    batch_size = attn_masks.size(0)     # Batch size
    max_len = input_ids.size(1)         # Max Text Tokens length

    # Gather Index - where to get the corresponding embeddings for each Text token/Image feature 
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1) # First Tokens are Text tokens

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data    
    if labels[0] is not None:
        labels = torch.stack(labels, dim=0)
        
    batch = {
            'file_ids': file_ids, 
            'input_ids': input_ids,
            'image_features': img_feats,
            'image_pos_features': img_pos_feats,
            'position_ids':position_ids,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'labels': labels,
            }    
    
    if sent_features is not None:
        sent_features = torch.stack(sent_features, dim=0)
        batch.update({'sent_features':sent_features})

    if gcn_tokens is not None:
        max_len = gather_index.size(1)
        gcn_pad_func = lambda arr: [x  + [-1] * (max_len - len(x)) for x in arr]

        batch_gcn_vocab_ids_paded = np.array(gcn_pad_func(gcn_tokens)).reshape(-1)
        # generate eye matrix according to gcn_vocab_size+1, the 1 is for gcn_pad_func prefix -1, then change to the row with all 0 value.
        # batch_gcn_swop_eye=torch.eye(gcn_conf.vocab_size+1)[batch_gcn_vocab_ids_paded][:,:-1]
        batch_gcn_swop_eye=torch.eye(gcn_conf.vocab_size)[batch_gcn_vocab_ids_paded]
        # This tensor is for transform batch_embedding_tensor to gcn_vocab order
        # -1 is seq_len. usage: batch_gcn_swop_eye.matmul(batch_seq_embedding)
        batch_gcn_swop_eye=batch_gcn_swop_eye.view(batch_size,-1, gcn_conf.vocab_size).transpose(1,2)
        batch.update({'gcn':batch_gcn_swop_eye})


    return batch        