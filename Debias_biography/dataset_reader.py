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
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, LabelField, ArrayField
from overrides import overrides
import json
from tqdm import tqdm
import numpy as np
import os
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

@DatasetReader.register("FastTextProfessionGenderDataReader")
class FastTextProfessionGenderDataReader(DatasetReader):
    def __init__(self,
                 randomseed: int=1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.fastTextPath = './data/FastText/'
        self.bertBiosPath = './data/bert_encode_biasbios'
        self.randomseed = randomseed
        self.g2i, self.i2g = load_dictionary('./data/biasbios/gender2index.txt')
        self.p2i, self.i2p = load_dictionary('./data/biasbios/profession2index.txt')
    
    @overrides
    def _read(self, file_path: str):
        #----------------------------bwv----------------------------------
        if file_path == "fasttext_train_protected":
            X_train = np.load(os.path.join(self.fastTextPath,'X_train.npy'))
            G_train = np.load(os.path.join(self.fastTextPath, 'G_train.npy'))
            assert len(X_train) == 255_710
            assert len(G_train) == 255_710
            # P_train = np.load(os.path.join(self.fastTextPath, 'P_train.npy'))
            for vec, gender_label in zip(X_train, G_train):
                yield self.text_to_instance(vec=vec, profession_label=None, gender_label=gender_label)
        if file_path == "fasttext_dev_protected":
            X_dev = np.load(os.path.join(self.fastTextPath,'X_dev.npy'))
            G_dev = np.load(os.path.join(self.fastTextPath, 'G_dev.npy'))
            assert len(X_dev) == 39_369
            assert len(G_dev) == 39_369
            # P_train = np.load(os.path.join(self.fastTextPath, 'P_train.npy'))
            for vec, gender_label in zip(X_dev, G_dev):
                yield self.text_to_instance(vec=vec, profession_label=None, gender_label=gender_label)
        if file_path == "fasttext_test_protected":
            X_test = np.load(os.path.join(self.fastTextPath,'X_test.npy'))
            G_test = np.load(os.path.join(self.fastTextPath, 'G_test.npy'))
            assert len(X_test) == 98_344
            assert len(G_test) == 98_344
            # P_train = np.load(os.path.join(self.fastTextPath, 'P_train.npy'))
            for vec, gender_label in zip(X_test, G_test):
                yield self.text_to_instance(vec=vec, profession_label=None, gender_label=gender_label)
        if file_path == "fasttext_main_train":
            X_train = np.load(os.path.join(self.fastTextPath,'X_train.npy'))
            G_train = np.load(os.path.join(self.fastTextPath, 'G_train.npy'))
            P_train = np.load(os.path.join(self.fastTextPath, 'P_train.npy'))
            assert len(X_train) == 255_710
            assert len(G_train) == 255_710
            assert len(P_train) == 255_710
            for vec, gender_label, profession_label in zip(X_train, G_train, P_train):
                yield self.text_to_instance(vec=vec, profession_label=profession_label, gender_label=gender_label)
        if file_path == "fasttext_main_dev":
            X_dev = np.load(os.path.join(self.fastTextPath,'X_dev.npy'))
            G_dev = np.load(os.path.join(self.fastTextPath, 'G_dev.npy'))
            P_dev = np.load(os.path.join(self.fastTextPath, 'P_dev.npy'))
            assert len(X_dev) == 39_369
            assert len(G_dev) == 39_369
            assert len(P_dev) == 39_369
            for vec, gender_label, profession_label in zip(X_dev, G_dev, P_dev):
                yield self.text_to_instance(vec=vec, profession_label=profession_label, gender_label=gender_label)
        if file_path == "fasttext_main_test":
            X_test = np.load(os.path.join(self.fastTextPath,'X_test.npy'))
            G_test = np.load(os.path.join(self.fastTextPath, 'G_test.npy'))
            P_test = np.load(os.path.join(self.fastTextPath, 'P_test.npy'))
            assert len(X_test) == 98_344
            assert len(G_test) == 98_344
            assert len(P_test) == 98_344
            for vec, gender_label, profession_label in zip(X_test, G_test, P_test):
                yield self.text_to_instance(vec=vec, profession_label=profession_label, gender_label=gender_label)
        
        #------------------------ bert ------------------------------
        if file_path == "bert_train_protected":
            X_train = np.load(os.path.join(self.bertBiosPath,'train_cls.npy'))
            G_train = np.load(os.path.join(self.fastTextPath, 'G_train.npy'))
            assert len(X_train) == 255_710
            assert len(G_train) == 255_710
            # P_train = np.load(os.path.join(self.fastTextPath, 'P_train.npy'))
            for vec, gender_label in zip(X_train, G_train):
                yield self.text_to_instance(vec=vec, profession_label=None, gender_label=gender_label)
        if file_path == "bert_dev_protected":
            X_dev = np.load(os.path.join(self.bertBiosPath,'dev_cls.npy'))
            G_dev = np.load(os.path.join(self.fastTextPath, 'G_dev.npy'))
            assert len(X_dev) == 39_369
            assert len(G_dev) == 39_369
            # P_train = np.load(os.path.join(self.fastTextPath, 'P_train.npy'))
            for vec, gender_label in zip(X_dev, G_dev):
                yield self.text_to_instance(vec=vec, profession_label=None, gender_label=gender_label)
        if file_path == "bert_test_protected":
            X_test = np.load(os.path.join(self.bertBiosPath,'test_cls.npy'))
            G_test = np.load(os.path.join(self.fastTextPath, 'G_test.npy'))
            assert len(X_test) == 98_344
            assert len(G_test) == 98_344
            # P_train = np.load(os.path.join(self.fastTextPath, 'P_train.npy'))
            for vec, gender_label in zip(X_test, G_test):
                yield self.text_to_instance(vec=vec, profession_label=None, gender_label=gender_label)
        if file_path == "bert_main_train":
            X_train = np.load(os.path.join(self.bertBiosPath,'train_cls.npy'))
            G_train = np.load(os.path.join(self.fastTextPath, 'G_train.npy'))
            P_train = np.load(os.path.join(self.fastTextPath, 'P_train.npy'))
            assert len(X_train) == 255_710
            assert len(G_train) == 255_710
            assert len(P_train) == 255_710
            for vec, gender_label, profession_label in zip(X_train, G_train, P_train):
                yield self.text_to_instance(vec=vec, profession_label=profession_label, gender_label=gender_label)
        if file_path == "bert_main_dev":
            X_dev = np.load(os.path.join(self.bertBiosPath,'dev_cls.npy'))
            G_dev = np.load(os.path.join(self.fastTextPath, 'G_dev.npy'))
            P_dev = np.load(os.path.join(self.fastTextPath, 'P_dev.npy'))
            assert len(X_dev) == 39_369
            assert len(G_dev) == 39_369
            assert len(P_dev) == 39_369
            for vec, gender_label, profession_label in zip(X_dev, G_dev, P_dev):
                yield self.text_to_instance(vec=vec, profession_label=profession_label, gender_label=gender_label)
        if file_path == "bert_main_test":
            X_test = np.load(os.path.join(self.bertBiosPath,'test_cls.npy'))
            G_test = np.load(os.path.join(self.fastTextPath, 'G_test.npy'))
            P_test = np.load(os.path.join(self.fastTextPath, 'P_test.npy'))
            assert len(X_test) == 98_344
            assert len(G_test) == 98_344
            assert len(P_test) == 98_344
            for vec, gender_label, profession_label in zip(X_test, G_test, P_test):
                yield self.text_to_instance(vec=vec, profession_label=profession_label, gender_label=gender_label)


    @overrides
    def text_to_instance(self,
                         vec: np.array,
                         profession_label: np.array,
                         gender_label: np.array) -> Instance:
        fields = {"vec":ArrayField(array= vec, dtype= np.float32)}
        if profession_label is not None:
            p = int(profession_label)
            fields['profession_labels'] = LabelField(p, label_namespace="profession_labels", skip_indexing=True)
        if gender_label is not None:
            g = self.i2g[int(gender_label)]
            fields['gender_labels'] = LabelField(g, label_namespace="gender_labels")

        return Instance(fields)