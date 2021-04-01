#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   TPR_GAP.py
@Time    :   2020/10/27 22:24:30
@Author  :   Wang Liwen
@Version :   1.0
@Contact :   w_liwen@bupt.edu.cn
@Homepage:   https://w-lw.github.io
'''

# here put the import lib
import torch
from overrides import overrides
from allennlp.training.metrics.metric import Metric

@Metric.register("tpr_gap")
class TPRGAPMetric(Metric):
    def __init__(self) -> None:
        self._white_pos = 0.
        self._white_neg = 0.
        self._black_pos = 0.
        self._black_neg = 0.
        self._white_pos_num = 0.
        self._white_neg_num = 0.
        self._black_pos_num = 0.
        self._black_neg_num = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 sent_label: torch.Tensor,
                 race_label: torch.Tensor):
        predictions, sent_label, race_label = self.unwrap_to_tensors(predictions, sent_label, race_label)
        if predictions.size() != sent_label.size() or sent_label.size() != race_label.size():
            raise ValueError("##### Label size is not same as predictions")
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, -1)
        sent_label = sent_label.view(batch_size, -1)
        race_label = race_label.view(batch_size, -1)
        #white:0 pos:0 balck:1 neg:1
        wp = ((1-race_label)*(1-sent_label)).float()
        wn = ((1-race_label)*(sent_label)).float()
        bp = ((race_label)*(1-sent_label)).float()
        bn = ((race_label)*(sent_label)).float()
        #calculate the num of four class
        self._white_pos_num += wp.sum()
        self._white_neg_num += wn.sum()
        self._black_pos_num += bp.sum()
        self._black_neg_num += bn.sum()
        #calculate the num of the correct predictions
        self._white_pos += (wp*(1-predictions)).float().sum()
        self._white_neg += (wn*(predictions)).float().sum()
        self._black_pos += (bp*(1-predictions)).float().sum()
        self._black_neg += (bn*(predictions)).float().sum()
    
    def get_metric(self, reset: bool=False):
        wp_acc = torch.tensor(0.0)
        wn_acc = torch.tensor(0.0)
        bp_acc = torch.tensor(0.0)
        bn_acc = torch.tensor(0.0)
        if self._white_pos_num > 0:
            wp_acc = self._white_pos/self._white_pos_num
        # else:
        #     wp_acc = 0.0

        if self._white_neg_num > 0:
            wn_acc = self._white_neg/self._white_neg_num
        # else:
        #     wn_acc = 0.0

        if self._black_pos_num > 0:
            bp_acc = self._black_pos/self._black_pos_num
        # else:
        #     bp_acc = 0.0

        if self._black_neg_num > 0:
            bn_acc = self._black_neg/self._black_neg_num
        # else:
        #     bn_acc = 0.0
        GAP_POS = wp_acc - bp_acc
        GAP_NEG = wn_acc - bn_acc
        # print(GAP_POS)
        # print(GAP_NEG)
        # print(GAP_POS)
        # print(GAP_NEG)
        GAP_RMS = torch.sqrt(torch.div(torch.pow(GAP_POS,2) + torch.pow(GAP_NEG,2),2))
        if reset:
            self.reset()
        # print(GAP_POS, GAP_NEG, GAP_RMS)
        return GAP_POS.item(), GAP_NEG.item(), GAP_RMS.item()

    @overrides
    def reset(self):
        self._white_pos = 0.
        self._white_neg = 0.
        self._black_pos = 0.
        self._black_neg = 0.
        self._white_pos_num = 0.
        self._white_neg_num = 0.
        self._black_pos_num = 0.
        self._black_neg_num = 0.