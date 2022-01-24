import time
import random
import pandas as pd
import pickle as pkl
import torch
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import dropout
from torch.optim import SGD, Adam, Adamax, AdamW
from datautils.dataloader import OBJECT_VOCAB_SIZE
import datautils.dataloader
from utils.gcn import GCNConfig, get_torch_gcn
from utils.logger import LOGGER, add_log_to_file, log_tensorboard
from pathlib import Path
from model.uniter import UniterForPretraining, UniterModel, UniterConfig
from model.semeval import SemevalUniter, SemevalUniterVGG19Sentiment, SemevalUniterVGG19Sentiment2, VGCN_Bert
from utils.uniter import IMG_DIM, IMG_LABEL_DIM
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.tensorboard import SummaryWriter

@dataclass(repr=True)
class Metrics:
    train_loss: float
    train_acc: float
    train_prec: float
    train_recall: float
    train_f1: float
    valid_loss: float
    valid_acc: float
    valid_prec: float
    valid_recall: float
    valid_f1: float
    def __getitem__(self, key):
        return super().__getattribute__(key)

    def __repr__(self):
        res = ""
        if self.train_loss is not None or self.train_f1 is not None:
            loss = self.train_loss if self.train_loss is not None else 0.0 
            res += f"Train: Loss={loss:.4f}, Acc = {self.train_acc:.4f}, Precision = {self.train_prec:.4f}, F1 = {self.train_f1:.4f}\n"
        if self.valid_loss is not None or self.valid_f1 is not None:
            loss = self.valid_loss if self.valid_loss is not None else 0.0 
            res += f"Valid: Loss={self.valid_loss:.4f}, Acc = {self.valid_acc:.4f}, Precision = {self.valid_prec:.4f}, F1 = {self.valid_f1:.4f}"

        return res

def batch_to_device(batches, device):
    for k in batches:
        if torch.is_tensor(batches[k]):
            batches[k] = batches[k].to(device)                


class Trainer():

    def __init__(self, config, progress=None):
        self.config = config
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 1
        self.total_iters = 0
        self.not_improved = 0
        self.best_epoch_info = None 
        self.current_best_value = None
        self.checkpoint_file = None
        self.metrics_history = []
        self.labels_list = []
        self.probs_list = []
        self.preds_list = []
        self.loss_list = []
        self.lr_history = []
        self.progress = progress
        self.writer = None

    def report_epoch(self, epoch):
        LOGGER.info("-"*30)
        LOGGER.info(f"Epoch {epoch} Report")
        LOGGER.info("="*30)
        LOGGER.info(self.metrics_history[-1])
        LOGGER.info("\n\n")

        
    def print_config(self):
        # Print args
        LOGGER.info("\n" + "x" * 50 + "\n\nRunning training with the following parameters: \n")
        for key, value in self.config.items():
            LOGGER.info(f"\t\t\t {key} : {value}")
        LOGGER.info("\n" + "x" * 50 + "\n\n")

    def epoch_end(self):        
        
        lr = self.scheduler.get_last_lr()
        self.lr_history.extend(lr)

        self.train_loss = sum(self.loss_list)/len(self.loss_list)

        self.total_iters += self.iters + 1

        # Evaluate on dev set
        
        val_labels, val_preds, val_loss = self.eval_model()

        # acc, prec, recall, f1 
        metrics =  (    
                        self.compute_metrics(self.labels_list, self.preds_list),
                        self.compute_metrics(val_labels, val_preds), 
                    )
        
        epoch_metric = Metrics(
            train_loss=self.train_loss,
            train_acc=metrics[0][0],
            train_prec=metrics[0][1],
            train_recall=metrics[0][2],
            train_f1=metrics[0][3],
            valid_loss=val_loss,
            valid_acc=metrics[1][0],
            valid_prec=metrics[1][1],
            valid_recall=metrics[1][2],
            valid_f1=metrics[1][3],
        )

        log_tensorboard(self.writer, epoch_metric,self.epoch, self.iters, len(self.config['train_loader']), skip_validation=False)
        self.epoch_metric = epoch_metric
        self.metrics_history.append(epoch_metric)

        # print stats
        self.report_epoch(self.epoch)

        self.probs_list = []
        self.preds_list = []
        self.labels_list = []
        self.loss_list = []
        self.id_list = []

    def avg_gradients(self, steps):
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = param.grad / steps

    def init_scheduler(self):
        if self.config['scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_decay_step'], gamma=self.config['lr_decay_factor'])
        elif self.config['scheduler'] == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10, 15, 25, 40], gamma=self.config['lr_decay_factor'])
        elif self.config['scheduler'] == 'warmup':
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config['warmup_steps'],
                                                             num_training_steps=len(self.config['train_loader']) * self.config['max_epoch'])
        elif self.config['scheduler'] == 'warmup_cosine':
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config['warmup_steps'],
                                                             num_training_steps=len(self.config['train_loader']) * self.config['max_epoch'])


    def init_optimizer(self):
        OptimCls = {
            'adam': Adam,
            'adamax': Adamax,
            'adamw': AdamW,
            'sgd': SGD,
        }[self.config['optimizer']]

        self.optimizer = OptimCls(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
            )

    def load_model(self):

        LOGGER.info("Loading model..")
        if self.config['model_file'] or not self.config['pretrained_model_file']:
            uniter_config = UniterConfig.from_json_file(self.config['config'])
            uniter_model = UniterModel(uniter_config, img_dim=IMG_DIM)


        else:
            # Pretrained model 
            LOGGER.info(f"Starting with vanilla pretrained UNITER model")

            checkpoint = torch.load(self.config['pretrained_model_file'])
            base_model = UniterForPretraining.from_pretrained(self.config['config'],
                                                              state_dict=checkpoint,
                                                              img_dim=IMG_DIM,
                                                              img_label_dim=IMG_LABEL_DIM)
            uniter_model = base_model.uniter

# gcn_adj_matrix=gcn_adj_list,
# gcn_adj_dim=gcn_conf.vocab_size, gcn_adj_num=len(gcn_adj_list),
# gcn_embedding_dim=args.gcn_embedding_dim,
        if self.config['with_gcn']:

            datautils.dataloader.gcn_conf = GCNConfig(
                vocab_size=base_model.config.vocab_size+OBJECT_VOCAB_SIZE,
                npmi_threshold=self.config['adj_npmi_threshold'],
                tf_threshold=self.config['adj_tf_threshold'],
                vocab_adj=self.config['adj_vocab_type'],
            )


            gcn_vocab_adj_tf, gcn_vocab_adj, adj_config = pkl.load(open(self.config['matrix_file'], 'rb'))

            gcn_adj_list = get_torch_gcn(gcn_vocab_adj_tf, gcn_vocab_adj, datautils.dataloader.gcn_conf) # Scipy sparse matrix to Torch

            for i in range(len(gcn_adj_list)): gcn_adj_list[i]=gcn_adj_list[i].to(self.device) # Send to device

            self.model = VGCN_Bert(uniter_model,
                                    gcn_adj_matrix=gcn_adj_list, 
                                    gcn_adj_dim=datautils.dataloader.gcn_conf.vocab_size, 
                                    gcn_adj_num=len(gcn_adj_list),
                                    gcn_embedding_dim=self.config['gcn_embedding_dim'], 
                                    num_labels=self.config['nr_classes'])

        elif self.config['with_vgg19']:
            self.model = SemevalUniterVGG19Sentiment(uniter_model=uniter_model,
                                    dropout=self.config['dropout'],
                                    n_classes=self.config['nr_classes'])
        else:
            self.model = SemevalUniter(uniter_model=uniter_model,                                    
                                    n_classes=self.config['nr_classes'])

        if self.config['model_file']:
            # previously saved model file 
            LOGGER.info(f"Loading previously saved model {self.config['model_file']}")
            self.model.load(self.config['model_file'])
        else:
            LOGGER.info(f"Starting with fresh UNITER instance")

        self.model.to(self.device)
        
        if self.config['task']=='train':
            self.init_optimizer()
            self.init_scheduler()

        if self.config['loss_func'] == 'bce_logits':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.config['pos_wt']]).to(self.device))
        elif self.config['loss_func'] == 'bce':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def average_gradients(self, steps):
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = param.grad / steps

    def logits_to_prediction(self, logits):
        if self.config['loss_func'] == 'bce':
            probs = logits
        elif self.config['loss_func'] == 'ce':            
            # probs = F.softmax(logits, dim=1)
            probs = logits
        elif self.config['loss_func'] == 'bce_logits':
            probs = torch.sigmoid(logits)

        if self.config['nr_classes']>2:
            preds = (probs>0.5).type(torch.FloatTensor)
        else:
            preds = torch.argmax(probs, dim=1)
            preds = torch.eye(2).to(preds.device).index_select(dim=0, index=preds)
         

        return preds, probs

    
    def calculate_loss(self, logits, batch_label, is_train):

        if self.config['loss_func'] == 'bce':
            logits = torch.sigmoid(logits)
        logits = logits.squeeze(1).to(self.device) if self.config['loss_func'] == 'bce_logits' else logits.to(self.device)

        loss = self.criterion(logits, batch_label.type(torch.FloatTensor).to(self.device) if self.config['loss_func']=='ce' else batch_label.float().to(self.device))

        if is_train :
            loss.backward()
            if self.iters % self.config['gradient_accumulation'] == 0:
                self.average_gradients(steps=self.config['gradient_accumulation'])
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            preds, probs = self.logits_to_prediction(logits)


            preds = preds.cpu().detach().tolist()
            probs = probs.cpu().detach().tolist()
            batch_label = batch_label.cpu().detach().tolist()

            self.preds_list.extend(preds)
            self.probs_list.extend(probs)
            self.labels_list.extend(batch_label)
            self.loss_list.append(loss.detach().item())
        
        return loss.detach().item()

    def check_early_stopping(self):

        this_metric = self.epoch_metric["valid_"+self.config['optimize_for']]
        diff = 999 if self.current_best_value is None else abs(self.current_best_value - this_metric)
        
        self.not_improved += 1

        if  (self.current_best_value is None)or(
                this_metric < self.current_best_value if self.config['optimize_for'] == 'loss' else this_metric > self.current_best_value
            ):
            LOGGER.info("New High Score! Saving model...")

            self.current_best_value  = this_metric
            self.best_epoch_info = {
                'metrics': this_metric,
                'epoch': self.epoch,

            }


            if not self.config["no_model_checkpoints"]:
                self.model.save(Path(self.config['model_path']) / self.checkpoint_file)

            if diff >= self.config['early_stop_thresh']:
                self.not_improved = 0
    
        LOGGER.info(f"current patience: {self.not_improved}")
        return self.not_improved >= self.config['patience']
        
        
    def log_training(self):
        LOGGER.info("-----------====== Training ended ======-----------")
        LOGGER.info(f"Best Metrics on epoch {self.best_epoch_info['epoch']}:")
        LOGGER.info(self.best_epoch_info['metrics'])

    def export_test_predictions(self, preds):
        output_file = Path(self.config['model_path']) / self.checkpoint_file
        output_file = output_file.parent / ("eval_"+output_file.name)
        output_file = output_file.with_suffix(".csv")

        df_raw = self.config['test_loader'].dataset.df


        if self.config['nr_classes']>2:
            # multiclass (Task B)
            if self.config['nr_classes']==4:
                df_raw = pd.concat([df_raw, pd.DataFrame(preds, columns=["pred_shaming",
                                        "pred_stereotype","pred_objectification","pred_violence"])], axis=1)
            else:
                df_raw = pd.concat([df_raw, pd.DataFrame(preds, columns=["pred_non_misogynous","pred_shaming",
                                        "pred_stereotype","pred_objectification","pred_violence"])], axis=1)
            df_raw['pred_misogynous'] = df_raw[["pred_shaming","pred_stereotype","pred_objectification","pred_violence"]].max(axis=1)
            df_raw['pred_misogynous'] = df_raw['pred_misogynous'].astype(int)
            df_raw['pred_shaming'] = df_raw['pred_shaming'].astype(int)
            df_raw['pred_stereotype'] = df_raw['pred_stereotype'].astype(int)
            df_raw['pred_objectification'] = df_raw['pred_objectification'].astype(int)
            df_raw['pred_violence'] = df_raw['pred_violence'].astype(int)

            if self.config['nr_classes']==4:
                df_raw_negatives = pd.read_csv(Path(self.config['data_path']) / ("test.csv"), sep="\t")
                df_raw_negatives = df_raw_negatives[df_raw_negatives.misogynous==0]
                df_raw_negatives['pred_shaming'] = 0
                df_raw_negatives['pred_stereotype'] = 0
                df_raw_negatives['pred_objectification'] = 0
                df_raw_negatives['pred_violence'] = 0
                df_raw = pd.concat([df_raw, df_raw_negatives])
        else:
            # binary (Task A)
            df_raw['pred_misogynous'] = np.argmax(preds, axis=1)
            df_raw['pred_misogynous'] = df_raw['pred_misogynous'].astype(int)

        df_raw.to_csv(output_file, sep="\t", index=None)

    def compute_metrics(self, labels, preds):
        if self.config['nr_classes']<=2:
            labels = np.argmax(labels, axis=1) if len(labels[-1])>1 else labels
            preds = np.argmax(preds, axis=1) if len(preds[-1])>1 else preds

            prec=precision_score(labels, preds, zero_division=0)
            recall=recall_score(labels, preds, zero_division=0)
            f1=f1_score(labels, preds, average='weighted', zero_division=0)
        else:
            prec=precision_score(labels, preds, average='weighted', zero_division=0)
            recall=recall_score(labels, preds, average='weighted', zero_division=0)
            f1=f1_score(labels, preds, average='weighted', zero_division=0) # Weighted pt Multiclass!!

        acc=accuracy_score(labels, preds)

        return acc, prec, recall, f1

    def end_training(self):
        # Termination message
        LOGGER.info("\n" + "-"*100)
        if self.epoch < self.config['max_epoch']:
            LOGGER.info("Training terminated on epoch {} because the Validation {} did not improve for {} epochs" .format(self.epoch, self.config['optimize_for'], self.config['patience']))
        else:
            LOGGER.info("Maximum epochs of {} reached. Finished training !!".format(self.config['max_epoch']))

        if len(self.config['test_loader']) and 'misogynous' in self.config['test_loader'].dataset.df:
            LOGGER.info("-"*30)
            LOGGER.info("\t\tEvaluating on test set")
            LOGGER.info("-"*30)

            if (Path(self.config['model_path']) / self.checkpoint_file).exists():
                self.model.load(Path(self.config['model_path']) / self.checkpoint_file)
                self.model.to(self.device)
            
            
            labels, preds, loss  = self.eval_model()
            acc, prec, recall, f1 = self.compute_metrics(labels, preds)
            LOGGER.info(f"\tLoss={loss:.4f} , Acc =  {acc:.4f} , Prec = {prec:.4f} , Recall = {recall:.4f} , F1 = {f1:.4f} ")
            self.export_test_predictions(preds)

            ##  if test data has label ---> compute metrics 
            ##
            ##
        else:
            acc, prec, recall, f1, loss = [0]*5

        self.log_training()

        if self.writer:
            self.writer.close()

        return acc, prec, recall, f1, loss

    def metric_reporting(self):
        loss = sum(self.loss_list)/len(self.loss_list)

        acc, prec, recall, f1 = self.compute_metrics(self.labels_list, self.preds_list)

        metric = Metrics(
            train_loss=loss,
            train_acc=acc,
            train_prec=prec,
            train_recall=recall,
            train_f1=f1,
            valid_loss=None,
            valid_acc=None,
            valid_prec=None,
            valid_recall=None,
            valid_f1=None, 
        )

        return metric

    def do_train(self):

        formatted_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = f"{formatted_date}/{self.config['model_save_name']}_{formatted_date}.pt"
        if self.config['with_tensorboard']:
            tensor_path = Path(self.config['tensorboard_path']) / formatted_date
            self.writer = SummaryWriter(str(tensor_path))

        add_log_to_file(Path(self.config['model_path']) / f"{formatted_date}/training_{formatted_date}.log")        

        LOGGER.info("\n\n" + "="*100 + "\n\t\t\t\t\t Training Network\n" + "="*100)
        
        self.print_config()
        
        self.start = time.time()
        LOGGER.info("\nBeginning training at:  {} \n".format(datetime.now()))

        epoch_prog = self.progress.add_task("[green]Epoch...", total=self.config['max_epoch'])

        for self.epoch in range(self.start_epoch, self.config['max_epoch']+1):
            self.progress.update(epoch_prog, advance=1)
            
            step_prog = self.progress.add_task("[yellow]Batch nr ...", total=len(self.config['train_loader']))
            for self.iters, self.batch in enumerate(self.config['train_loader']):
                
                if self.config['debug'] and self.iters>10:
                    break

                # if self.config['with_tensorboard'] and self.epoch==self.start_epoch and self.iters==0:
                #     batch = dict(img_feat=self.batch['image_features'],
                #                         img_pos_feat=self.batch['image_pos_features'],
                #                         input_ids=self.batch['input_ids'],
                #                         position_ids=self.batch['position_ids'],
                #                         attention_mask=self.batch['attn_masks'],
                #                         gather_index=self.batch['gather_index'],
                #                         )
                #     # self.writer.add_graph(self.model, batch)

                self.model.train()
                self.progress.update(step_prog, advance=1)

                batch_to_device(self.batch, self.device)
                iter_time = time.time()
                if self.config['with_gcn']:
                    self.preds = self.model(img_feat=self.batch['image_features'],
                                        img_pos_feat=self.batch['image_pos_features'],
                                        input_ids=self.batch['input_ids'],
                                        position_ids=self.batch['position_ids'],
                                        attention_mask=self.batch['attn_masks'],
                                        gather_index=self.batch['gather_index'],
                                        output_all_encoded_layers=False,
                                        gcn_swop_eye=self.batch['gcn'])
                elif self.config['with_vgg19']:
                    self.preds = self.model(img_feat=self.batch['image_features'],
                                        img_pos_feat=self.batch['image_pos_features'],
                                        input_ids=self.batch['input_ids'],
                                        position_ids=self.batch['position_ids'],
                                        attention_mask=self.batch['attn_masks'],
                                        gather_index=self.batch['gather_index'],
                                        output_all_encoded_layers=False,
                                        vgg_pool=self.batch['sent_features'])
                else:
                    self.preds = self.model(img_feat=self.batch['image_features'],
                                            img_pos_feat=self.batch['image_pos_features'],
                                            input_ids=self.batch['input_ids'],
                                            position_ids=self.batch['position_ids'],
                                            attention_mask=self.batch['attn_masks'],
                                            gather_index=self.batch['gather_index'],
                                            output_all_encoded_layers=False)

                self.calculate_loss(self.preds, self.batch['labels'], is_train=True)



                if self.iters % 10 == 0:
                    metric = self.metric_reporting()
                    self.progress.update(step_prog, description=f"[yellow]Batch nr {self.iters}... loss={metric.train_loss:.2f}, acc={metric.train_acc:.2f}, f1={metric.train_f1:.2f}")
                    log_tensorboard(self.writer, metric,self.epoch, self.iters, len(self.config['train_loader']), skip_validation=True)

            self.progress.remove_task(step_prog)

            self.epoch_end()
            self.progress.update(epoch_prog, 
                description=f"[green]Epoch {self.epoch} ... loss={self.epoch_metric.train_loss:.2f}, "
                    f"f1={self.epoch_metric.train_f1:.2f}, "
                    f"val_loss={self.epoch_metric.valid_loss:.2f}, "
                    f"val_f1={self.epoch_metric.valid_f1:.2f}")
        
            if self.check_early_stopping():
                LOGGER.info(f"Early stopping on epoch {self.epoch}")
                break

        metric = self.end_training()
        return self.best_epoch_info, metric

    @torch.no_grad()
    def eval_model(self, dataset='validate_loader'):
        label_list = []
        pred_list = []
        loss_list = []

        epoch = self.epoch if hasattr(self, 'epoch') else '=None='
        eval_prog = self.progress.add_task(f"[green]Evaluation Epoch {epoch}...", total=len(self.config[dataset]))
        self.model.eval()
        for step, batch in enumerate(self.config[dataset]):
            batch_to_device(batch, self.device)
            self.progress.update(eval_prog, advance=1)
            
            if self.config['debug'] and step>10:
                break

            if self.config['with_gcn']:
                logits = self.model(img_feat=self.batch['image_features'],
                                    img_pos_feat=self.batch['image_pos_features'],
                                    input_ids=self.batch['input_ids'],
                                    position_ids=self.batch['position_ids'],
                                    attention_mask=self.batch['attn_masks'],
                                    gather_index=self.batch['gather_index'],
                                    output_all_encoded_layers=False,
                                    gcn_swop_eye=self.batch['gcn'])

            elif self.config['with_vgg19']:
                logits = self.model(img_feat=batch['image_features'],
                                    img_pos_feat=batch['image_pos_features'],
                                    input_ids=batch['input_ids'],
                                    position_ids=batch['position_ids'],
                                    attention_mask=batch['attn_masks'],
                                    gather_index=batch['gather_index'],
                                    output_all_encoded_layers=False,
                                    vgg_pool=batch['sent_features'])
            else:
                logits = self.model(img_feat=batch['image_features'],
                                        img_pos_feat=batch['image_pos_features'],
                                        input_ids=batch['input_ids'],
                                        position_ids=batch['position_ids'],
                                        attention_mask=batch['attn_masks'],
                                        gather_index=batch['gather_index'],
                                        output_all_encoded_layers=False)
            
            pred, probs = self.logits_to_prediction(logits)
            loss = self.calculate_loss(logits, batch['labels'], is_train=False)

            loss_list.append(loss)
            label_list.extend(batch['labels'].cpu().detach().tolist())
            pred_list.extend(pred.cpu().detach().tolist())

        self.progress.remove_task(eval_prog)
        eval_loss = sum(loss_list) / len(loss_list)
        return label_list, pred_list, eval_loss 

    @torch.no_grad()
    def predict_model(self, dataset='test_loader'):
        pred_list = []
        prob_list = []
        file_ids = []

        predict_prog = self.progress.add_task("[green]Predicting ...", total=len(self.config[dataset]))
        self.model.eval()
        for step, batch in enumerate(self.config[dataset]):
            batch_to_device(batch, self.device)
            self.progress.update(predict_prog, advance=1)

            if self.config['with_gcn']:
                logits = self.model(img_feat=self.batch['image_features'],
                                    img_pos_feat=self.batch['image_pos_features'],
                                    input_ids=self.batch['input_ids'],
                                    position_ids=self.batch['position_ids'],
                                    attention_mask=self.batch['attn_masks'],
                                    gather_index=self.batch['gather_index'],
                                    output_all_encoded_layers=False,
                                    gcn_swop_eye=self.batch['gcn'])
            elif self.config['with_vgg19']:
                logits = self.model(img_feat=batch['image_features'],
                                    img_pos_feat=batch['image_pos_features'],
                                    input_ids=batch['input_ids'],
                                    position_ids=batch['position_ids'],
                                    attention_mask=batch['attn_masks'],
                                    gather_index=batch['gather_index'],
                                    output_all_encoded_layers=False,
                                    vgg_pool=batch['sent_features'])
            else:
                logits = self.model(img_feat=batch['image_features'],
                                    img_pos_feat=batch['image_pos_features'],
                                    input_ids=batch['input_ids'],
                                    position_ids=batch['position_ids'],
                                    attention_mask=batch['attn_masks'],
                                    gather_index=batch['gather_index'],
                                    output_all_encoded_layers=False)
            
            pred, probs = self.logits_to_prediction(logits)
            prob_list.extend(probs.cpu().detach().tolist())
            pred_list.extend(pred.cpu().detach().tolist())
            file_ids.extend(batch['file_ids'])

        return file_ids, pred_list, prob_list

    @staticmethod
    def set_seed(seed):
        # Seeds for reproduceable runs
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def prepare_args(parser):
        # Named parameters
        parser.add_argument('--task', choices=['train', 'eval','predict'], default='train', 
                            help='Task to perform')
        # Required Paths
        parser.add_argument('--data_path', type=str, default='./dataset', required=True,
                            help='path to dataset folder that contains the processed data files')
        parser.add_argument('--nr_classes', type=int, choices=[2,5], default=2,
                            help='Number of classes for the classifier (2,5)')

        parser.add_argument('--model_path', type=str, default='./model_checkpoints',
                            help='Directory for saving trained model checkpoints')
        parser.add_argument('--tensorboard_path', type=str, default='./tensorboard',
                            help='Directory for saving tensorboard checkpoints')
        parser.add_argument("--model_save_name", type=str, default='best_model',
                            help='saved model name')
        parser.add_argument('--debug', action="store_true",
                            help='This option is intended for tests on local machines, and more output.')
        parser.add_argument('--with_cleanup', action="store_true",
                            help='Enable text Cleanup')
        parser.add_argument('--with_tensorboard', action="store_true",
                            help='Enable Tensorboard logging')

        parser.add_argument('--with_vgg19', action="store_true",
                            help='Load VGG augmented model')
        parser.add_argument('--with_gcn', action="store_true", help='Load VGCN  model')

        # Load pretrained model
        parser.add_argument('--pretrained_model_file', type=str, help='Filename for the pretrained model')
        parser.add_argument('--model_file', type=str, help='Full path to File of with the previously saved model')
        parser.add_argument('--no_model_checkpoints', action="store_true", help='Do not save model')
        parser.add_argument("--matrix_file", type=str,  help="Sparse Array files (pickled)")

        # Required Paths
        parser.add_argument('--config', type=str, default='./config/uniter-base.json',
                            help='JSON config file')
        parser.add_argument('--feature_path', type=str, default='./dataset/img_feats',
                            help='Path to image features')
        parser.add_argument('--sentiment_feature_path', type=str, 
                            help='Path to image sentiment features')

        #### Pre-processing Params ####
        parser.add_argument('--max_txt_len', type=int, default=64,
                            help='max number of tokens in text (BERT BPE)')
        parser.add_argument('--min_object_conf', type=float, default=0.2,
                            help='Confidence threshold for dynamic bounding boxes (object detection)')
        parser.add_argument('--max_bb', type=int, default=100,
                            help='max number of bounding boxes')
        parser.add_argument('--min_bb', type=int, default=10,
                            help='min number of bounding boxes')
        parser.add_argument('--num_bb', type=int, default=36,
                            help='static number of bounding boxes')
        parser.add_argument('--use_nms', action="store_true",
                            help='Select bounding boxes using Non-Maximum Suppression')
        parser.add_argument('--nms_threshold', type=float, default=0.6,
                            help='Intersection-over-union threshold for NMS')

        #### Training Params ####

        # Numerical params
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Standard dropout regularization')
        parser.add_argument("--gcn_embedding_dim", type=int, default=128, help="GCN Embedding size")
        parser.add_argument("--adj_npmi_threshold", type=float, default=0.2, help="Minimum NPMI in the Adjacency matrix")
        parser.add_argument("--adj_tf_threshold", type=float, default=0.0, help="Minimum Term Frequency (TF) in the Adjacency matrix")
        parser.add_argument("--adj_vocab_type", choices=['all', 'pmi', 'tf'], default='all', help="Build graph based on PMI, TF or both")

        # Named parameters
        parser.add_argument('--optimizer', choices=['adam', 'adamax','adamw'], default='adam', 
                            help='Optimizer to use for training: adam / adamax / adamw')
        parser.add_argument('--optimize_for', choices=['loss', 'acc','prec','recall','f1'], default='f1', 
                            help='Early stopping based on this metric')

        ## Not sure whether we should have this here. For a multi-task setup, we need our own loss functions
        parser.add_argument('--loss_func', type=str, default='bce_logits',
                            help='Loss function to use for optimization: bce / bce_logits / ce')
        parser.add_argument('--pos_wt', type=float, default=1,
                              help='Loss reweighting for the positive class to deal with class imbalance')
        parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                            help='The type of lr scheduler to use anneal learning rate: step/multi_step/warmup/warmp_cosine')

        # Numerical parameters
        parser.add_argument('--batch_size', type=int, default=8,
                            help='batch size for training')
        parser.add_argument('--gradient_accumulation', type=int, default= 1,
                            help='No. of update steps to accumulate before performing backward pass')
        parser.add_argument('--max_grad_norm', type=int, default=5,
                            help='max gradient norm for gradient clipping')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate for training')
        parser.add_argument('--warmup_steps', type=int, default=50,
                            help='No. of steps to perform linear lr warmup for')
        parser.add_argument('--weight_decay', type=float, default=1e-3,
                            help='weight decay for optimizer')
        parser.add_argument('--max_epoch', type=int, default=20,
                            help='Max epochs to train for')
        parser.add_argument('--lr_decay_step', type=float, default= 3,
                            help='No. of epochs after which learning rate should be decreased')
        parser.add_argument('--lr_decay_factor', type=float, default=0.8,
                            help='Decay the learning rate of the optimizer by this multiplicative amount')
        parser.add_argument('--patience', type=float, default= 5,
                            help='Patience no. of epochs for early stopping')
        parser.add_argument('--early_stop_thresh', type=float, default=1e-3,
                            help='Patience no. of epochs for early stopping')
        parser.add_argument('--seed', type=int, default= 42,
                            help='set seed for reproducability')

    @staticmethod
    def check_config(config):
        # Check all provided paths:
        assert Path(config['data_path']).exists(), "[!] ERROR: Dataset path does not exist"
        LOGGER.info("Data path checked..")

        Path(config['model_path']).mkdir(exist_ok=True)
        LOGGER.info("Creating checkpoint path for saved models at:  {}\n".format(config['model_path']))

        if config.get('config'):
            assert Path(config['config']).exists(), "[!] ERROR: config JSON path does not exist"

        if config.get('model_file'):
            assert Path(config['model_file']).exists(), f"Model file {config['model_file']} must exist"

        if config.get('with_tensorboard'):
            Path(config['tensorboard_path']).mkdir(parents=True, exist_ok=True)

        if config.get('with_vgg19'):
            assert config.get('sentiment_feature_path') and Path(config['sentiment_feature_path']).exists(), "[!] ERROR: Sentiment Feature path does not exist"

        if config.get('with_gcn'):
            assert config.get('matrix_file') and Path(config['matrix_file']).exists(), "[!] ERROR: Matrix File does not exist"



        Trainer.set_seed(config['seed'])
        return config


