#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   process_data.py
@Time    :   2020/11/14 13:57:10
@Author  :   Wang Liwen
@Version :   1.0
@Contact :   w_liwen@bupt.edu.cn
@Homepage:   https://w-lw.github.io
'''

# here put the import lib
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
import numpy as np
import pickle
from collections import defaultdict, Counter
from typing import List, Dict
import tqdm

STOPWORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

def load_dataset(path):
    
    with open(path, "rb") as f:
        
        data = pickle.load(f)
    return data

def load_dictionary(path):
    
    with open(path, "r", encoding = "utf-8") as f:
        
        lines = f.readlines()
        
    k2v, v2k = {}, {}
    for line in lines:
        
        k,v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k
    
    return k2v, v2k
    
def count_profs_and_gender(data: List[dict]):
    
    counter = defaultdict(Counter)
    for entry in data:
        gender, prof = entry["g"], entry["p"]
        counter[prof][gender] += 1
        
    return counter

def load_word_vectors(fname):
    
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.vocab.keys())
    return model, vecs, words


def get_embeddings_based_dataset(data: List[dict], word2vec_model, p2i, g2i, filter_stopwords = False):
    
    X, P, G = [], [], []
    unk, total = 0., 0.
    unknown = []
    vocab_counter = Counter()
    
    for entry in tqdm.tqdm(data, total = len(data)):
        
        p = p2i[entry["p"]]
        g = g2i[entry["g"]]
        # print(entry)
        words = entry["hard_text"].split(" ")
        # words = entry["hard_text_tokenized"].split(" ")
        if filter_stopwords:
            words = [w for w in words if w.lower() not in STOPWORDS]
            
        vocab_counter.update(words) 
        bagofwords = np.sum([word2vec_model[w] if w in word2vec_model else word2vec_model["unk"] for w in words], axis = 0)
        #print(bagofwords.shape)
        X.append(bagofwords)
        P.append(p)
        G.append(g)
        total += len(words)
        
        unknown_entry = [w for w in words if w not in word2vec_model]
        unknown.extend(unknown_entry)
        unk += len(unknown_entry)
    
    X = np.array(X)
    P = np.array(P)
    G = np.array(G)
    print("% unknown: {}".format(unk/total))
    return X,P,G, unknown,vocab_counter

train = load_dataset("./data/biasbios/train.pickle")
dev = load_dataset("./data/biasbios/dev.pickle")
test = load_dataset("./data/biasbios/test.pickle")

p2i, i2p = load_dictionary("./data/biasbios/profession2index.txt")
g2i, i2g = load_dictionary("./data/biasbios/gender2index.txt")
counter = count_profs_and_gender(train+dev+test)
word2vec, vecs, words = load_word_vectors("./data/embeddings/crawl-300d-2M.vec")
X_train, P_train, G_train, unknown_train, vocab_counter_train = get_embeddings_based_dataset(train, word2vec, p2i, g2i)
X_dev, P_dev, G_dev, unknown_dev, vocab_counter_dev =  get_embeddings_based_dataset(dev, word2vec, p2i, g2i)
X_test, P_test, G_test, unknown_test, vocab_counter_test =  get_embeddings_based_dataset(test, word2vec, p2i, g2i)

np.save('./data/FastText/X_train.npy', X_train)
np.save('./data/FastText/P_train.npy', P_train)
np.save('./data/FastText/G_train.npy', G_train)

np.save('./data/FastText/X_dev.npy', X_dev)
np.save('./data/FastText/P_dev.npy', P_dev)
np.save('./data/FastText/G_dev.npy', G_dev)

np.save('./data/FastText/X_test.npy', X_test)
np.save('./data/FastText/P_test.npy', P_test)
np.save('./data/FastText/G_test.npy', G_test)


