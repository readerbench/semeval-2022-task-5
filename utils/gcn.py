# -*- coding: utf-8 -*-

# # # #
# utils.py
# @author Zhibin.LU
# @created 2019-12-24T10:51:59.943Z-05:00
# @last-modified 2020-01-02T03:47:54.026Z-05:00
# @website: https://louis-udm.github.io
# @description 
# # # #

import torch
import numpy as np
import scipy.sparse as sp
import re

import  nltk
from nltk.corpus import stopwords
from dataclasses import dataclass

'''
General functions
'''

nltk.download('stopwords')
stopwords_pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('romanian')) + r')\b\s*')

@dataclass
class GCNConfig(object):
    vocab_size: int
    npmi_threshold: float
    tf_threshold: float
    vocab_adj: str


def clean_text(text, remove_stop_words=False, remove_numeric=False):
    # delete [ \t\n\r\f\v]
    space_pattern = r'[\s+\xa0]'
    text = re.sub(space_pattern, ' ', text)
    if remove_stop_words:
        text = stopwords_pattern.sub('', text)
    if remove_numeric:
        text = re.sub("\w*\d\w*"," ",text)
    return text

def normalize_adj(adj):
    """
        Symmetrically normalize adjacency matrix.

    """

    D_matrix = np.array(adj.sum(axis=1)) # D-degree matrix as array (Diagonal, rest is 0.)
    D_inv_sqrt = np.power(D_matrix, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(D_inv_sqrt) # array to matrix
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt) # D^(-1/2) . A . D^(-1/2)



def sparse_scipy2torch(coo_sparse):
    # coo_sparse=coo_sparse.tocoo()
    i = torch.LongTensor(np.vstack((coo_sparse.row, coo_sparse.col)))
    v = torch.from_numpy(coo_sparse.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


def get_torch_gcn(gcn_vocab_adj_tf, gcn_vocab_adj,gcn_config:GCNConfig):

    gcn_vocab_adj_tf.data *= (gcn_vocab_adj_tf.data > gcn_config.tf_threshold)
    gcn_vocab_adj_tf.eliminate_zeros()

    gcn_vocab_adj.data *= (gcn_vocab_adj.data > gcn_config.npmi_threshold)
    gcn_vocab_adj.eliminate_zeros()

    if gcn_config.vocab_adj == 'pmi':
        gcn_vocab_adj_list = [gcn_vocab_adj]
    elif gcn_config.vocab_adj == 'tf':
        gcn_vocab_adj_list = [gcn_vocab_adj_tf]
    elif gcn_config.vocab_adj == 'all':
        gcn_vocab_adj_list = [gcn_vocab_adj_tf, gcn_vocab_adj]

    norm_gcn_vocab_adj_list = []
    for i in range(len(gcn_vocab_adj_list)):
        adj = gcn_vocab_adj_list[i]
        adj = normalize_adj(adj)
        norm_gcn_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()))

    del gcn_vocab_adj_list

    return norm_gcn_vocab_adj_list

