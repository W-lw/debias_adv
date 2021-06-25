#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/10/14 15:23:19
@Author  :   Wang Liwen
@Version :   1.0
@Contact :   w_liwen@bupt.edu.cn
@Homepage:   https://w-lw.github.io
'''

# here put the import lib
from typing import Iterable, Dict, Any
import logging
import os
import argparse
import math
from sys import path
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, cleanup_global_logging
from allennlp.nn import util as nn_util
from allennlp.data import DatasetReader, Instance
from allennlp.data import Vocabulary, DataIterator
from allennlp.training import Trainer
from allennlp.models import Model
from allennlp.training.util import evaluate
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.tqdm import Tqdm
from models import RaceClassifierWithMojiEncoder, MainAdversarialClassifier, SentClassifierWithMojiEncoder
from dataset_reader import SentimentRaceDataReader
# CUDA_DEVICE = 1
# def getTPR_Gap(model,test_dataset, iterator):
#     white_pos = 0.0
#     white_neg = 0.0
#     black_pos = 0.0
#     black_neg = 0.0
#     assert len(test_dataset) == 8000
#     import torch
#     import numpy as np
#     device='cuda:0'
#     for test_Instance in test_dataset:
#         # print(test_Instance) 
#         # print(test_Instance['vec'].array)
#         if test_Instance['race_label'].label=="white":
#             if test_Instance["sent_label"].label=="positive":
#                 # print('###########')
#                 vec = torch.tensor([test_Instance['vec'].array.astype(np.int64)],device=device)
#                 sent_hat = model(vec,torch.tensor([0],device=device),None)['y_hat']
#                 # print(sent_hat)
#                 if sent_hat == 0:
#                     white_pos += 1
#             else:
#                 vec = torch.tensor([test_Instance['vec'].array.astype(np.int64)],device=device)
#                 sent_hat = model(vec,torch.tensor([1],device=device),None)['y_hat']
#                 # print(sent_hat)
#                 if sent_hat == 1:
#                     white_neg += 1
#         else:
#             if test_Instance["sent_label"].label=="positive":
#                 vec = torch.tensor([test_Instance['vec'].array.astype(np.int64)],device=device)
#                 sent_hat = model(vec,torch.tensor([0],device=device),None)['y_hat']
#                 # print(sent_hat)
#                 if sent_hat == 0:
#                     black_pos += 1
#             else:
#                 vec = torch.tensor([test_Instance['vec'].array.astype(np.int64)],device=device)
#                 sent_hat = model(vec,torch.tensor([1],device=device),None)['y_hat']
#                 # print(sent_hat)
#                 if sent_hat == 1:
#                     black_neg += 1
#     # TPR_white = (white_pos + white_neg)/4000.0
#     # TPR_black = (black_pos + black_neg)/4000.0
#     white_pos /= 2000.0
#     white_neg /= 2000.0
#     black_pos /= 2000.0
#     black_neg /= 2000.0
#     GAP_pos = math.fabs(white_pos - black_pos)
#     GAP_neg = math.fabs(white_neg - black_neg)
#     GAP_rms = math.sqrt((math.pow(GAP_pos,2)+math.pow(GAP_neg,2))/2)
#     print(white_pos,white_neg, black_pos, black_neg)
#     return GAP_pos, GAP_neg, GAP_rms


def main(args):
    params = Params.from_file(args.config_path)
    print(args.output_dir)
    stdout_handler = prepare_global_logging(args.output_dir,False)
    prepare_environment(params)

    reader = DatasetReader.from_params(params["dataset_reader"])
    train_dataset = reader.read(params.pop("train_protected"))
    valid_dataset = reader.read(params.pop("validate_protected"))
    test_dataset = reader.read(params.pop("test_protected"))
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    #set model from config file
    model_params = params.pop("model", None)
    model = Model.from_params(model_params.duplicate(), vocab=None)#, cuda_device=-1)
    #save config file
    with open(args.config_path,'r',encoding='utf-8')as f_in:
        with open(os.path.join(args.output_dir,"config.json"),'w',encoding='utf-8')as f_out:
            f_out.write(f_in.read())
    iterator = DataIterator.from_params(params.pop("iterator", vocab))
    iterator.index_with(vocab)
    trainer_params = params.pop("trainer",None)
    # print(list(trainer_params.pop("optimizer")))
    # print(model.state_dict().keys())
    # print(vocab)
    # i=0
    # for i in range(8):
    #     print(train_dataset[i])
    trainer = Trainer.from_params(model=model,
                                  serialization_dir=args.output_dir,
                                  iterator=iterator,
                                  train_data=train_dataset,
                                  validation_data=valid_dataset,
                                  params=trainer_params.duplicate())
    trainer.train()

    if test_dataset:
        logging.info("Evaluating on the test set")
        import torch
        model.load_state_dict(torch.load(os.path.join(args.output_dir,"best.th")))#"model_state_epoch_2.th")))#
        test_metrics = evaluate(model,test_dataset,iterator,
                                cuda_device=trainer_params.pop("cuda_device", 0),
                                batch_weight_key=None)
        logging.info(f"Metrics on the test set: {test_metrics}")
        with open(os.path.join(args.output_dir, "test_metrics.txt"), "w", encoding="utf-8") as f_out:
            f_out.write(f"Metrics on the test set: {test_metrics}")
        model.saveFeature(args.output_dir)
    # model.eval()
    # GAP_pos, GAP_neg, GAP_rms=getTPR_Gap(model,test_dataset,iterator)
    # print(GAP_pos, GAP_neg, GAP_rms)

    cleanup_global_logging(stdout_handler)



if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    
    # dataset type
    arg_parser.add_argument("--data_type",type=str,default="train_protectedAttr_classifier")
    # path of config file 
    arg_parser.add_argument("--config_path",type=str,default="./config/train_Main_embed_adv.json")
    # output path of the model
    output_path = "./output_baselines/1100_0.9_acc"
    arg_parser.add_argument("--output_dir",type=str,default=output_path)
    args = arg_parser.parse_args()
    main(args)


