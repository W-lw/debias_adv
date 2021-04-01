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
def getTPR_Gap(model,test_dataset, iterator):
    white_pos = 0.0
    white_neg = 0.0
    black_pos = 0.0
    black_neg = 0.0
    assert len(test_dataset) == 8000
    import torch
    import numpy as np
    device='cuda:0'
    for test_Instance in test_dataset:
        # print(test_Instance) 
        # print(test_Instance['vec'].array)
        if test_Instance['race_label'].label=="white":
            if test_Instance["sent_label"].label=="positive":
                # print('###########')
                vec = torch.tensor([test_Instance['vec'].array.astype(np.int64)],device=device)
                sent_hat = model(vec,torch.tensor([0],device=device),None)['y_hat']
                # print(sent_hat)
                if sent_hat == 0:
                    white_pos += 1
            else:
                vec = torch.tensor([test_Instance['vec'].array.astype(np.int64)],device=device)
                sent_hat = model(vec,torch.tensor([1],device=device),None)['y_hat']
                # print(sent_hat)
                if sent_hat == 1:
                    white_neg += 1
        else:
            if test_Instance["sent_label"].label=="positive":
                vec = torch.tensor([test_Instance['vec'].array.astype(np.int64)],device=device)
                sent_hat = model(vec,torch.tensor([0],device=device),None)['y_hat']
                # print(sent_hat)
                if sent_hat == 0:
                    black_pos += 1
            else:
                vec = torch.tensor([test_Instance['vec'].array.astype(np.int64)],device=device)
                sent_hat = model(vec,torch.tensor([1],device=device),None)['y_hat']
                # print(sent_hat)
                if sent_hat == 1:
                    black_neg += 1
    # TPR_white = (white_pos + white_neg)/4000.0
    # TPR_black = (black_pos + black_neg)/4000.0
    white_pos /= 2000.0
    white_neg /= 2000.0
    black_pos /= 2000.0
    black_neg /= 2000.0
    GAP_pos = math.fabs(white_pos - black_pos)
    GAP_neg = math.fabs(white_neg - black_neg)
    GAP_rms = math.sqrt((math.pow(GAP_pos,2)+math.pow(GAP_neg,2))/2)
    print(white_pos,white_neg, black_pos, black_neg)
    return GAP_pos, GAP_neg, GAP_rms


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

# def evaluate(model: Model,
#              instances: Iterable[Instance],
#              data_iterator: DataIterator,
#              cuda_device: int,
#              batch_weight_key: str) -> Dict[str, Any]:
#     check_for_gpu(cuda_device)
#     model.eval()

#     iterator = data_iterator(instances,
#                                 num_epochs=1,
#                                 shuffle=False)
#     generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

#     # Number of batches in instances.
#     batch_count = 0
#     # Number of batches where the model produces a loss.
#     loss_count = 0
#     # Cumulative weighted loss
#     total_loss = 0.0
#     # Cumulative weight across all batches.
#     total_weight = 0.0

#     for batch in generator_tqdm:
#         batch_count += 1
#         batch = nn_util.move_to_device(batch, cuda_device)
#         output_dict = model(**batch)
#         loss = output_dict.get("loss")

#         metrics = model.get_metrics()

#         if loss is not None:
#             loss_count += 1
#             if batch_weight_key:
#                 weight = output_dict[batch_weight_key].item()
#             else:
#                 weight = 1.0

#             total_weight += weight
#             total_loss += loss.item() * weight
#             # Report the average loss so far.
#             metrics["loss"] = total_loss / total_weight

#         description = ', '.join(["%s: %.2f" % (name, value) for name, value
#                                     in metrics.items() if not name.startswith("_")]) + " ||"
#         generator_tqdm.set_description(description, refresh=False)

#     final_metrics = model.get_metrics(reset=True)
#     if loss_count > 0:
#         # Sanity check
#         if loss_count != batch_count:
#             raise RuntimeError("The model you are trying to evaluate only sometimes " +
#                                 "produced a loss!")
#         final_metrics["loss"] = total_loss / total_weight

#     return final_metrics

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    #
    arg_parser.add_argument("--data_type",type=str,default="train_protectedAttr_classifier")
    # arg_parser.add_argument("--config_path",type=str,default="./config/train_protectedAttr_classifier_0.5.json")
    # arg_parser.add_argument("--config_path",type=str,default="./config/sent.json")
    # arg_parser.add_argument("--config_path",type=str,default="./config/test_on_unbalanced_data.json")
    arg_parser.add_argument("--config_path",type=str,default="./config/train_Main_embed_adv.json")
    # arg_parser.add_argument("--config_path",type=str,default="./config/extract_feature.json")
    # arg_parser.add_argument("--config_path",type=str,default="./config/train_Main_with_adv.json")
    # arg_parser.add_argument("--config_path",type=str,default="./config/random_adv.json")
    # output_path = "./outputs_embed/seed@1-1-1-1-ratio_0.5_1110_EntropyLoss_0.08"
    # output_path = "./outputs_embed/seed@1-1-1-1-ratio_0.8_0010_CrossEntropyLoss_1.4"
    output_path = "./output_baselines/1100_0.9_acc"
    # output_path = "./outputs_random/r_0.5_n_1.5"#默认是0010
    # output_path = "./outputs_random/r_0.5_0011_n_1.5"#默认是0010
    # output_path = "./outputs_random/r_0.6_1010_n_1.5"
    # output_path = "./outputs_base/seed@1-1-1-2-ratio_0.8_1100_embed"
    # output_path = "./output_baselines/0100_0.5_gap"
    # output_path = "./output/0.8_0110_0.8_E_few"
    # output_path = "./outputs_embed/seed@1-1-1-1-ratio_0.8_0010_CrossEntropyLoss_0.8"
    # print(output_path)
    arg_parser.add_argument("--output_dir",type=str,default=output_path)
    # arg_parser.add_argument("--config_path",type=str,default="./config/train_Main_with_adv.json")
    # arg_parser.add_argument("--output_dir",type=str,default="./output/old_debug3_")
    # arg_parser.add_argument("--output_dir",type=str,default="./output/protected_few")
    args = arg_parser.parse_args()
    # print(args.output_dir)
    # exit()
    main(args)