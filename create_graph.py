from scipy.sparse.sputils import isintlike
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict, Counter
from math import log
import scipy.sparse as sp
import sys
import itertools
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import pickle as pk
from tqdm import tqdm
from joblib import Parallel, delayed
import sklearn

from utils.gcn import clean_text
OBJECT_VOCAB_SIZE = 1600

def corpus_freq(corpus, objects, tokenizer, min_word_freq, text_window_size, text_window_stride,
                    remove_stop_words, remove_numeric, no_idf, use_multiprocessing, num_workers):

    def clean_corpus(text):
        return clean_text(text, remove_stop_words, remove_numeric)
    def tokenize(texts):
        return tokenizer.batch_encode_plus(texts)['input_ids']

    print("Tokenize and Encode Corpus .... ")
    if use_multiprocessing:
        with Parallel(n_jobs=num_workers, backend="threading") as paralell:
            corpus = paralell(
                delayed(clean_corpus)(y) for x in corpus for y in x.split("\n") if len(y)>2# split in  Paragraphs
            )

        with Parallel(n_jobs=num_workers, backend="threading") as paralell:
            encoded = paralell(
                delayed(tokenize)(corpus[x*100:x*100+100]) for x in range(len(corpus) // 100 + 1)
            )
            encoded = list(itertools.chain(*encoded))
    else:
        corpus = [clean_corpus(y) for x in corpus for y in x.split("\n") if len(y)>2] # split in  Paragraphs
        encoded = tokenize(corpus)
    num_docs = len(corpus)
    
    print("Done! ")
    print(f"Got {num_docs} lines with {sum([len(x) for x in encoded])} elements")

    objects = [[int(x)+tokenizer.vocab_size for x in inner if len(x)>0] for inner in objects]
    for i in range(len(encoded)):
        del (encoded[i][0])  # remove [CLS] and [SEP] token
        del (encoded[i][-1])
        encoded[i].extend(objects[i])

    if min_word_freq > 1:
        word_list_global = [item for sublist in encoded for item in sublist]
        word_global_count = defaultdict(lambda: 0, Counter(word_list_global))  # in how many docs each word appears

        encoded = [[item for item in sublist if word_global_count[item] > min_word_freq] for sublist in
                   encoded]  # remove with lower frequency

    word_list_per_doc = [item for sublist in encoded for item in Counter(sublist)]

    word_doc_count = defaultdict(lambda: 0, Counter(word_list_per_doc))  # in how many docs each word appears

    mixt_token_list = list(tokenizer.ids_to_tokens.keys()) + list(range(tokenizer.vocab_size,tokenizer.vocab_size+OBJECT_VOCAB_SIZE))

    idf = map(lambda word: log((1.0 + num_docs) / (1.0 + word_doc_count[word])) + 1.0, mixt_token_list)
    idf = dict(zip(mixt_token_list, idf))
    del (word_doc_count[tokenizer.sep_token_id])  # remove counters for special tokens
    del (word_doc_count[tokenizer.cls_token_id])

    word_window_count = defaultdict(lambda: 0)  # In how many windows word x appears
    word_pairs_window_count = defaultdict(lambda: 0)
    doc_word_count = defaultdict(lambda: 0)
    num_windows = 0

    print("starting text window parsing")

    def generate_window_matrix(array, window_size, stride=1):
        if len(array) < window_size:
            return np.array([array])

        if stride > 1:
            result = (np.lib.stride_tricks.sliding_window_view(np.array(array), window_size))[::stride, :]
        else:
            result = np.lib.stride_tricks.sliding_window_view(np.array(array), window_size)

        return result

    def inc(d, x, val=1):
        d[x] += val

    for index, line in tqdm(enumerate(encoded), total=num_docs):
        # text window information

        window_matrix = generate_window_matrix(line, text_window_size, stride=text_window_stride) # get a matrix of sliding windows

        # Increment Word count in document for each word --> used in TF
        _ = list(map(lambda word: inc(doc_word_count, (index, word)), line))

        num_windows += window_matrix.shape[0] # Increment number of windows
        unique_in_windows = [np.unique(x) for x in window_matrix]
        unique_in_windows_flat = np.concatenate(unique_in_windows)

        _ = list(map(lambda word: inc(word_window_count, word),
                     unique_in_windows_flat))  # Increment Word count in window for each unique word
        count_word_pairs = Counter(itertools.product(unique_in_windows_flat, unique_in_windows_flat))

        _ = list(map(lambda word_pair: inc(word_pairs_window_count, word_pair, val=count_word_pairs[word_pair]),
                     count_word_pairs.keys()))  # Increment Word pair count in window for each unique word pair
    # Compute TF-IDF
    row_tf = range(num_docs)
    col_tf = idf.keys()
    print(f"Compute TF/IDF for {len(row_tf)} x {len(col_tf)}")

    row_tf, col_tf = zip(*doc_word_count.keys()) # unpack the (document, token_id) keys
    weight_tf = doc_word_count.values()
    weight_tf_idf = [doc_word_count[pair] * idf[pair[1]] for pair in doc_word_count]

    # compute pmi and npmi matrices

    row, col = zip(*word_pairs_window_count.keys())

    weight_pmi = []
    weight_npmi = []
    print("Compute PMI  and TF")

    # Normalize word_pair_count
    word_pairs_window_count_norm = np.array(list(word_pairs_window_count.values()), dtype=float) / num_windows

    # normalize Word_window_count
    word_window_count_norm = np.array(list(word_window_count.values()), dtype=float) / num_windows
    word_window_count_norm = dict(zip(word_window_count.keys(), word_window_count_norm))

    for (word_i, word_j, p_ij) in tqdm(zip(row, col, word_pairs_window_count_norm), total=len(row)):
        p_i = word_window_count_norm[word_i]
        p_j = word_window_count_norm[word_j]

        pmi = log(p_ij / (p_i * p_j + sys.float_info.epsilon))  # Pointwise mutual information
        npmi = - pmi / (log(p_ij) + sys.float_info.epsilon)  # Normalized pmi

        weight_pmi.append(pmi)
        weight_npmi.append(npmi)

        # In paper foloseste doar pmi >0 si npmi > 0 - de ce?
        # to check
    print("Compute Adjecancy ")
    vocab_adj = sp.csr_matrix((weight_npmi, (row, col)), shape=(tokenizer.vocab_size+OBJECT_VOCAB_SIZE, tokenizer.vocab_size+OBJECT_VOCAB_SIZE),
                              dtype=np.float32)
    vocab_adj.setdiag(1.0)

    # Calculate isomorphic vocab adjacency matrix using doc\'s tf-idf
    if no_idf:
        tfidf_all = sp.csr_matrix((np.array(weight_tf), (row_tf, col_tf)), shape=(num_docs, tokenizer.vocab_size+OBJECT_VOCAB_SIZE),
                                  dtype=np.float32)
    else:
        tfidf_all = sp.csr_matrix((np.array(weight_tf_idf), (row_tf, col_tf)), shape=(num_docs, tokenizer.vocab_size+OBJECT_VOCAB_SIZE),
                                  dtype=np.float32)

    print("Normalize tfidf")
    # vocab_tfidf = tfidf_all.T.tolil()
    vocab_tfidf = tfidf_all.T
    vocab_tfidf= sklearn.preprocessing.normalize(vocab_tfidf, norm='l2', axis=1)
    vocab_adj_tf = vocab_tfidf.dot(vocab_tfidf.T)

    return vocab_adj, vocab_adj_tf

def main(args):
    print("Load Bert Tokenizer (%s) .... " % args.bert_model)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    print("Got %d tokens. good.." % tokenizer.vocab_size)

    datafile = Path(args.data_folder) / args.dataset

    df = pd.read_csv(datafile)
    data = df['text'].tolist()
    
    objects = df['object_ids'].apply(lambda x: x.strip('[]').split(',')).tolist()



    a1, a2 = corpus_freq(data, objects, tokenizer, args.min_word_frequency,
                         args.text_window_size, args.text_window_stride,
                         args.remove_stop_words, args.remove_numeric,
                         args.no_idf, args.multithread, args.num_workers)

    outfile = Path(args.data_folder) / args.output_file
    pk.dump([a1, a2, {
        "bert_model": args.bert_model,
        "remove_stop_words": args.remove_stop_words,

    }], open(outfile, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert-model", type=str, default="readerbench/RoBERT-base", help="BERT Huggingface modelname")
    parser.add_argument("--dataset", type=str,  help="Corpus used for the Graph creation (located in data-folder)")
    parser.add_argument("--data-folder", type=str,  help="Data folder", required=True)
    parser.add_argument("--output-file", type=str,  help="Filename for the Pickled arrays", required=True)
    parser.add_argument("--remove-stop-words", action="store_true", default=False,
                            help="Remove stop words before computing graph")
    parser.add_argument("--remove-numeric", action="store_true", default=False,
                            help="Remove Words containing numbers - like fooball scores, dates, etc")
    parser.add_argument("--no-idf", action="store_true", default=False, help="Do not use IDF for the weights")
    parser.add_argument("--dataset-plain-text", action="store_true", default=False,
                            help="Use plain text reading instead of pandas")
    parser.add_argument("--min-word-frequency", type=int, default=1,
                            help="Minimum word frequency to get in the graph (remove low frequency words)")
    parser.add_argument("--text-window-size", type=int, default=1000,
                            help="Text Window in which we count co-occurrences ")
    parser.add_argument("--text-window-stride", type=int, default=250,
                            help="Text Window in which we count co-occurrences ")

    parser.add_argument("--multithread", action="store_true", default=False, help="Use joblib Paralell processing")
    parser.add_argument("--num-workers", type=int, default=10,
                            help="Joblib parallel workers")


    args = parser.parse_args()

    main(args)
