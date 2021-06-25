#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset_reader.py
@Time    :   2020/10/10 08:30:31
@Author  :   Wang Liwen
@Version :   1.0
@Contact :   w_liwen@bupt.edu.cn
@Homepage:   https://w-lw.github.io
'''

# here put the import lib
from argparse import Namespace
from allennlp.common.util import namespace_match
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, LabelField, ArrayField
import numpy
from overrides import overrides
from torchmoji.sentence_tokenizer import SentenceTokenizer
import json
from tqdm import tqdm
import numpy as np
import os
# import torch
# from torch.utils.data import DataLoader,Dataset

@DatasetReader.register("SentimentRaceDataReader")
class SentimentRaceDataReader(DatasetReader):
    def __init__(self,
                 ratio: float = 0.5,
                 randomseed: int = 1,
                 n: int = 100_000,
                 few: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.n = n
        self.ratio = ratio
        self.maxlen = 150
        self.few = few
        self.vocabPath = "./data/mojiModel/vocabulary.json"
        self.sentceTokenizer = self.get_sentTokenizer()
        self.pos_pos, self.pos_neg, self.neg_pos, self.neg_neg = self.get_sentences()
        np.random.seed(randomseed)#之前进行的一些实验是在95上做的
        np.random.shuffle(self.pos_pos)
        np.random.shuffle(self.pos_neg)
        np.random.shuffle(self.neg_pos)
        np.random.shuffle(self.neg_neg)
        self.pos_pos = self.tokenizeSentences(self.pos_pos)#70k
        self.pos_neg = self.tokenizeSentences(self.pos_neg)#100k
        self.neg_pos = self.tokenizeSentences(self.neg_pos)#44k
        self.neg_neg = self.tokenizeSentences(self.neg_neg)#100k
        
    def get_sentTokenizer(self):
        '''
            Reuse torchMoji's sentence tokenizer;
        '''
        with open(self.vocabPath,'r') as f:
            vocab = json.load(f)
        return SentenceTokenizer(vocab, self.maxlen)

    @overrides
    def _read(self, file_path: str) :
        if file_path == "train":
            # ratios for pos / neg
            n_1 = int(self.n * self.ratio / 2)
            n_2 = int(self.n * (1 - self.ratio) / 2)
            for dataIn,sent_label,race_label,class_n in zip([self.pos_pos, self.pos_neg, self.neg_pos, self.neg_neg],
                                                        ['positive', 'positive', 'negative', 'negative'],
                                                        ['white','black','white','black'],
                                                        [n_1, n_2, n_2, n_1]):
                for vec in dataIn[:class_n]:
                    yield self.text_to_instance(vec,sent_label,race_label)
        if file_path == "train_protectedAttr_classifier":
            pos_pos = self.pos_pos[:40_000]
            pos_neg = self.pos_neg[:40_000]
            neg_pos = self.neg_pos[:40_000]
            neg_neg = self.neg_neg[:40_000]
            for dataIn,race_label in zip([pos_pos, pos_neg, neg_pos, neg_neg],['white','black','white','black']):
                if self.few:
                    for vec in dataIn[:2000]:
                        yield self.text_to_instance(vec=vec,sent_label=None,race_label=race_label)
                else:
                    for vec in dataIn:
                        yield self.text_to_instance(vec=vec,sent_label=None,race_label=race_label)

        if file_path == "dev_protectedAttr_classifier":
            pos_pos = self.pos_pos[40_000:42_000]
            pos_neg = self.pos_neg[40_000:42_000]
            neg_pos = self.neg_pos[40_000:42_000]
            neg_neg = self.neg_neg[40_000:42_000]
            for dataIn,race_label in zip([pos_pos, pos_neg, neg_pos, neg_neg],['white','black','white','black']):
                for vec in dataIn:
                    yield self.text_to_instance(vec=vec,sent_label=None,race_label=race_label)

        if file_path == "test_protectedAttr_classifier":
            pos_pos = self.pos_pos[42_000:44_000]
            pos_neg = self.pos_neg[42_000:44_000]
            neg_pos = self.neg_pos[42_000:44_000]
            neg_neg = self.neg_neg[42_000:44_000]
            for dataIn,race_label in zip([pos_pos, pos_neg, neg_pos, neg_neg],['white','black','white','black']):
                for vec in dataIn:
                    yield self.text_to_instance(vec=vec,sent_label=None,race_label=race_label)

        if file_path == "train_main_classifier":
            n_1 = int(self.n * self.ratio / 2)
            n_2 = int(self.n * (1 - self.ratio) / 2)
            for dataIn,sent_label,race_label,class_n in zip([self.pos_pos, self.pos_neg, self.neg_pos, self.neg_neg],
                                                        ['positive', 'positive', 'negative', 'negative'],
                                                        ['white','black','white','black'],
                                                        [n_1, n_2, n_2, n_1]):
                for vec in dataIn[:class_n]:
                    yield self.text_to_instance(vec,sent_label,race_label)
        if file_path == "dev_main_classifier":
            pos_pos = self.pos_pos[40_000:42_000]
            pos_neg = self.pos_neg[40_000:42_000]
            neg_pos = self.neg_pos[40_000:42_000]
            neg_neg = self.neg_neg[40_000:42_000]
            for dataIn,sent_label,race_label in zip([pos_pos, pos_neg, neg_pos, neg_neg],
                                                        ['positive', 'positive', 'negative', 'negative'],
                                                        ['white','black','white','black']):
                for vec in dataIn:
                    yield self.text_to_instance(vec,sent_label,race_label)
        if file_path == "test_main_classifier":
            pos_pos = self.pos_pos[42_000:44_000]
            pos_neg = self.pos_neg[42_000:44_000]
            neg_pos = self.neg_pos[42_000:44_000]
            neg_neg = self.neg_neg[42_000:44_000]
            for dataIn,sent_label,race_label in zip([pos_pos, pos_neg, neg_pos, neg_neg],
                                                        ['positive', 'positive', 'negative', 'negative'],
                                                        ['white','black','white','black']):
                for vec in dataIn:
                    yield self.text_to_instance(vec,sent_label,race_label)
        if file_path == "test_main_classifier_unbalanced":
            pos_pos = self.pos_pos[42_000:44_000]
            pos_neg = self.pos_neg[42_000:44_000]
            neg_pos = self.neg_pos[42_000:44_000]
            neg_neg = self.neg_neg[42_000:44_000]
            n_1 = int(5000 * self.ratio / 2)
            n_2 = int(5000 * (1 - self.ratio) / 2)
            for dataIn,sent_label,race_label, class_n in zip([pos_pos, pos_neg, neg_pos, neg_neg],
                                                        ['positive', 'positive', 'negative', 'negative'],
                                                        ['white','black','white','black'],
                                                        [n_1, n_2, n_2, n_1]):
                for vec in dataIn[:class_n]:
                    yield self.text_to_instance(vec,sent_label,race_label)
            
    
    @overrides
    def text_to_instance(self, vec: np.array, 
                            sent_label: str,
                            race_label: str=None) -> Instance:
        fields = {"vec": ArrayField(array=vec,dtype=np.int64)}
        if sent_label is not None:
            fields['sent_label'] = LabelField(sent_label,label_namespace="sent_labels")
        if race_label is not None:
            fields["race_label"] = LabelField(race_label,label_namespace="race_labels")
        return Instance(fields)

    def get_sentences(self,filePath="./data/sent_race/"):
        pos_pos = []
        pos_neg = []
        neg_pos = []
        neg_neg = []
        with open(os.path.join(filePath,"pos_pos.txt"),'r',errors="replace")as fin:
            for line in fin:
                sentnce = line.strip()
                pos_pos.append(sentnce)#70k
        with open(os.path.join(filePath,"pos_neg.txt"),'r',errors="replace")as fin:
            for line in fin:
                sentnce = line.strip()
                pos_neg.append(sentnce)#100k
        with open(os.path.join(filePath,"neg_pos.txt"),'r',errors="replace")as fin:
            for line in fin:
                sentnce = line.strip()
                neg_pos.append(sentnce)#44k
        with open(os.path.join(filePath,"neg_neg.txt"),'r',errors="replace")as fin:
            for line in fin:
                sentnce = line.strip()
                neg_neg.append(sentnce)#100k
        #to keep the size of whole testing dataset is fixed
        # print(len(pos_pos))
        # print(len(pos_neg))
        # print(len(neg_pos))
        # print(len(neg_neg))
        assert len(pos_pos)>=44000
        assert len(pos_neg)>=44000
        assert len(neg_pos)>=44000
        assert len(neg_neg)>=44000
        return pos_pos[:44000],pos_neg[:44000],neg_pos[:44000],neg_neg[:44000]

    def tokenizeSentences(self,datas):
        tokenizedData = []
        for i in tqdm(range(0,len(datas),1000)):
            tokenized, _, _ = self.sentceTokenizer.tokenize_sentences(datas[i: i+1000])
            tokenizedData.extend(tokenized)
        return np.array(tokenizedData)