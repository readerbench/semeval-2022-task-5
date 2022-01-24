import math
import torch
from torch import nn
from model.uniter import UniterModel
import torch.nn.init as init

class SemevalUniter(nn.Module):

    def __init__(self,
                 uniter_model: UniterModel,
                 n_classes: int):
        super().__init__()
        self.uniter_model = uniter_model
        self.n_classes = n_classes
        hidden_size=uniter_model.config.hidden_size
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, **kwargs):
        out = self.uniter_model(**kwargs)
        out = self.uniter_model.pooler(out)
        out = self.linear(out)
        return out

    def save(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
        
    def load(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))



class SemevalUniterVGG19Sentiment(nn.Module):

    def __init__(self,
                 uniter_model: UniterModel,
                 dropout:float,
                 n_classes: int):
        super().__init__()
        self.uniter_model = uniter_model
        self.n_classes = n_classes
        self.drop = nn.Dropout(p=dropout)
        hidden_size = uniter_model.config.hidden_size
        # self.sent_linear = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        # )
        self.linear = nn.Linear(hidden_size+4096, n_classes)

    def forward(self, vgg_pool, **kwargs):
        out = self.uniter_model(**kwargs)
        out = self.uniter_model.pooler(out)

        # vgg_pool_ = vgg_pool.view(vgg_pool.size(0), -1)        
        # sent_out = self.sent_linear(vgg_pool_)
        sent_out = self.drop(vgg_pool)
        sent_out = vgg_pool.view(vgg_pool.size(0), -1)
        
        out = torch.cat((out,sent_out), dim=1)
        out = self.linear(out)
        return out

    def save(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
        
    def load(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))        
        

class SemevalUniterVGG19Sentiment2(nn.Module):

    def __init__(self,
                 uniter_model: UniterModel,
                 dropout:float,
                 n_classes: int):
        super().__init__()
        self.uniter_model = uniter_model
        self.n_classes = n_classes
        hidden_size = uniter_model.config.hidden_size
        self.lstm_size = 128
        self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=self.lstm_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

        self.sent_linear = nn.Sequential(
            nn.Linear(2*self.lstm_size+4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, n_classes),
        )


    def forward(self, vgg_pool, **kwargs):
        out = self.uniter_model(**kwargs)
        # out = self.uniter_model.pooler(out)
        out, (h,c) = self.lstm(out)        
        out = torch.cat((out[:,-1, :self.lstm_size],out[:,0, self.lstm_size:]),dim=-1)
        # out = torch.cat((h[0],h[1]),dim=-1)
        # vgg_pool_ = vgg_pool.view(vgg_pool.size(0), -1)        
        # sent_out = self.sent_linear(vgg_pool_)
        sent_out = vgg_pool.view(vgg_pool.size(0), -1)
        
        out = torch.cat((out,sent_out), dim=1)
        out = self.sent_linear(out)
        return out

    def save(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
        
    def load(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))        
        


class VocabGraphConvolution(nn.Module):
    """Vocabulary GCN module.

    Params:
        `voc_dim`: The size of vocabulary graph
        `num_adj`: The number of the adjacency matrix of Vocabulary graph
        `hid_dim`: The hidden dimension after XAW
        `out_dim`: The output dimension after Relu(XAW)W
        `dropout_rate`: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `X_dv`: the feature of mini batch document, can be TF-IDF (batch, vocab), or word embedding (batch, word_embedding_dim, vocab)

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """
    def __init__(self,adj_matrix,voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super(VocabGraphConvolution, self).__init__()
        self.adj_matrix=adj_matrix
        self.voc_dim=voc_dim
        self.num_adj=num_adj
        self.hid_dim=hid_dim
        self.out_dim=out_dim

        for i in range(self.num_adj):
            setattr(self, 'W%d_vh'%i, nn.Parameter(torch.randn(voc_dim, hid_dim)))

        self.fc_hc=nn.Linear(hid_dim,out_dim) 
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n,p in self.named_parameters():
            if n.startswith('W') or n.startswith('a') or n in ('W','a','dense'):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            H_vh=self.adj_matrix[i].mm(getattr(self, 'W%d_vh'%i))
            # H_vh=self.dropout(F.elu(H_vh))
            H_vh=self.dropout(H_vh)
            H_dh=X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear=X_dv.matmul(getattr(self, 'W%d_vh'%i))
                H_linear=self.dropout(H_linear)
                H_dh+=H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out=self.fc_hc(fused_H)
        return out



class VGCN_Bert(nn.Module):
    """VGCN-BERT model for text classification. It inherits from Huggingface's BertModel.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `gcn_adj_dim`: The size of vocabulary graph
        `gcn_adj_num`: The number of the adjacency matrix of Vocabulary graph
        `gcn_embedding_dim`: The output dimension after VGCN
        `num_labels`: the number of classes for the classifier. Default = 2.
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `gcn_swop_eye`: The transform matrix for transform the token sequence (sentence) to the Vocabulary order (BoW order)
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.

    Outputs:
        Outputs the classification logits of shape [batch_size, num_labels].

    """
    def __init__(self, uniter_model: UniterModel, gcn_adj_matrix, gcn_adj_dim, gcn_adj_num, 
                    gcn_embedding_dim, num_labels,output_attentions=False):
        super().__init__()
        self.vocab_gcn=VocabGraphConvolution(gcn_adj_matrix,gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim) #192/256
        self.uniter_model = uniter_model
        self.gcn_adj_matrix=gcn_adj_matrix
        self.gcn_adj_dim =  gcn_adj_dim
        self.gcn_adj_num =  gcn_adj_num
        self.gcn_embedding_dim = gcn_embedding_dim
        self.LayerNorm = nn.LayerNorm(uniter_model.config.hidden_size, eps=uniter_model.config.layer_norm_eps)
        self.dropout = nn.Dropout(uniter_model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(uniter_model.config.hidden_size, num_labels)
        self.will_collect_cls_states=False
        self.all_cls_states=[]
        self.output_attentions=output_attentions
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.apply(self.init_bert_weights)

    def uniter_embeddings2(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index=None, img_masks=None,
                txt_type_ids=None, img_type_ids=None, **kwargs):
        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self.uniter_model._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self.uniter_model._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self.uniter_model._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, img_masks, txt_type_ids, img_type_ids)
        return embedding_output

    def uniter_embeddings(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                txt_type_ids=None, img_type_ids=None, **kwargs):
        if txt_type_ids is None:
            txt_type_ids = torch.zeros_like(input_ids)
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())


        words_embeddings = self.uniter_model.embeddings.word_embeddings(input_ids)
        position_embeddings = self.uniter_model.embeddings.position_embeddings(position_ids)
        token_type_embeddings_txt = self.uniter_model.embeddings.token_type_embeddings(txt_type_ids)
        token_type_embeddings_img = self.uniter_model.embeddings.token_type_embeddings(img_type_ids)

        transformed_im = self.uniter_model.img_embeddings.img_layer_norm(self.uniter_model.img_embeddings.img_linear(img_feat))
        transformed_pos = self.uniter_model.img_embeddings.pos_layer_norm(self.uniter_model.img_embeddings.pos_linear(img_pos_feat))

        return words_embeddings, position_embeddings, token_type_embeddings_txt, transformed_im, transformed_pos, token_type_embeddings_img
            
    def forward(self, gcn_swop_eye, gather_index, attention_mask, **kwargs):
        input_ids = kwargs['input_ids']
      
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        (words_embeddings, position_embeddings_txt, token_type_embeddings_txt, 
            image_embeddings, position_embeddings_img, token_type_embeddings_img 
            )= self.uniter_embeddings(**kwargs)
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.uniter_model.config.hidden_size)
        joint_embeddings = torch.gather(torch.cat([words_embeddings, image_embeddings], dim=1),
                                        dim=1, index=gather_index)
        vocab_input=gcn_swop_eye.matmul(joint_embeddings).transpose(1,2)

        embeddings_img = image_embeddings + position_embeddings_img + token_type_embeddings_img
        embeddings_img = self.uniter_model.img_embeddings.LayerNorm(embeddings_img)
        embeddings_img = self.uniter_model.img_embeddings.dropout(embeddings_img)

        embeddings_txt = words_embeddings + position_embeddings_txt + token_type_embeddings_txt
        embeddings_txt = self.uniter_model.embeddings.LayerNorm(embeddings_txt)
        embeddings_txt = self.uniter_model.embeddings.dropout(embeddings_txt)
       
        embedding_output = torch.gather(torch.cat([embeddings_txt, embeddings_img], dim=1),
                                        dim=1, index=gather_index)

        
        gcn_vocab_out = self.vocab_gcn(vocab_input)
        input_lens = attention_mask.sum(-1)
        gcn_vocab_out = gcn_vocab_out.transpose(1, 2)

        gcn_vocab_out = self.LayerNorm(gcn_vocab_out)

        gcn_words_embeddings = nn.ConstantPad3d((0,0,0,gcn_vocab_out.size(1),0,0), 0)(embedding_output)

        # gcn_words_embeddings=joint_embeddings.clone()
        # for i in range(self.gcn_embedding_dim):
        #     tmp_pos=(attention_mask.sum(-1)-2-self.gcn_embedding_dim+1+i)+torch.arange(0,input_ids.shape[0]).to(self.device)*input_ids.shape[1]
        #     gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos,:]=gcn_vocab_out[:,:,i]

        # De incercat si un LayerNorm pe GCN

        for i, seq_len in enumerate(input_lens.tolist()):
            gcn_words_embeddings[i, seq_len:seq_len + self.gcn_embedding_dim,:] = gcn_vocab_out[i, :, :]

        # Add the GCN embeddings to the mask 
        attention_mask = (torch.arange(gcn_words_embeddings.size(1))[None, :].to(input_lens.device)
                    < (input_lens+self.gcn_embedding_dim)[:, None])
        # Poate trebuie modificat sa adunam positional si la GCN...

        # seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

        # position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # if self.gcn_embedding_dim>0:
        #     embeddings = gcn_words_embeddings + position_embeddings + token_type_embeddings
        # else:
        #     embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = gcn_words_embeddings
        # embeddings = self.LayerNorm(embeddings)
        embedding_output = self.dropout(embeddings)
        

        # We create a 3D attention mask from a 2D tensor mask. 
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        output_all_encoded_layers=self.output_attentions
        encoded_layers = self.uniter_model.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      )

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        pooled_output = self.uniter_model.pooler(encoded_layers)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if output_all_encoded_layers:
            return encoded_layers, logits

        return logits


    def save(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
        
    def load(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))



