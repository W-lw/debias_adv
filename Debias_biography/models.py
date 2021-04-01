#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Time    :   2020/10/14 13:58:01
@Author  :   Wang Liwen
@Version :   1.0
@Contact :   w_liwen@bupt.edu.cn
@Homepage:   https://w-lw.github.io
'''

# here put the import lib
from typing import Optional
from overrides.overrides import overrides
from torch.autograd import Variable
from torch.nn.modules import loss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from typing import Dict
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import BooleanAccuracy, F1Measure
from allennlp.models import Model
import torch
import torch.nn as nn
# from torch.nn import Module
import copy
from TPR_GAP import TPRGAPMetric

@Model.register("gender_classify")
class GenderClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 inputSize: int,
                 classNum: int) -> None:
        super().__init__(vocab)
        linear = nn.Linear(inputSize, classNum)
        ac = nn.LogSoftmax(dim=1)
        self.logistic = nn.Sequential(*[linear, ac])
        self.loss = nn.NLLLoss()
        self.metrics = {'accuracy': BooleanAccuracy(),
                        'f1': F1Measure(positive_label=1)}

    @overrides
    def forward(self, 
                vec: torch.Tensor,
                gender_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.logistic(vec)
        y_hat = torch.argmax(out,dim=1)
        output = {'y_hat': y_hat}
        self.metrics['accuracy'](y_hat, gender_labels)
        self.metrics['f1'](nn.functional.softmax(out, dim=1), gender_labels)
        output['loss'] = self.loss(out, gender_labels)
        return output
    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        return {"accuracy": self.metrics['accuracy'].get_metric(reset=reset),
                "p": p,
                "r": r,
                "f1": f1}
    
@Model.register("profession_classify")
class ProfessionClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 inputSize: int,
                 classNum: int) -> None:
        super().__init__(vocab)
        # self.linear1 = nn.Linear(inputSize, int(inputSize/2))
        self.linear = nn.Linear(inputSize, classNum)
        # self.ac = nn.Sigmoid()
        self.ac = nn.LogSoftmax(dim=1)
        self.logistic = nn.Sequential(*[self.linear, self.ac])
        self.loss = nn.NLLLoss()
        # self.loss = nn.CrossEntropyLoss()
        self.metrics = {'accuracy': BooleanAccuracy(),
                        'f1': F1Measure(positive_label=1),
                        'TPR_GAP': TPRGAPMetric()}

    
    @overrides
    def forward(self, 
                vec: torch.Tensor,
                gender_labels: torch.Tensor,
                profession_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        # print(vec.shape)
        out = self.logistic(vec)
        y_hat = torch.argmax(out,dim=1)
        output = {'y_hat': y_hat}
        self.metrics['accuracy'](y_hat, profession_labels)
        self.metrics['TPR_GAP'](y_hat, profession_labels=profession_labels, gender_labels=gender_labels)
        # self.metrics['f1'](nn.functional.softmax(out, dim=1), profession_labels)
        output['loss'] = self.loss(out, profession_labels)
        # output['TPR_GAP'] = 
        return output

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        # p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        outputs, GAP_RMS = self.metrics['TPR_GAP'].get_metric(reset=reset)
        return {"accuracy": self.metrics['accuracy'].get_metric(reset=reset),
                "GAP_RMS": GAP_RMS}

@Model.register("main_adv")
class MainAdversarial(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 protectedClassifierPath: str,
                 fixProtectedParams: bool,
                 add_attack_noise: bool,
                 if_test_adv:bool,
                 loss_type: str,
                 inputSize: int,
                 do_noise_normalization: bool = True,
                 noise_norm: Optional[float] = None,
                 classNum: int=28) -> None:
        super().__init__(vocab)
        # self.linear1 = nn.Linear(inputSize, int(inputSize/2))
        # self.ac1 = nn.Tanh()
        # self.linear = nn.Linear(int(inputSize/2), classNum)
        self.fixProtectedParams = fixProtectedParams
        self.add_attack_noise = add_attack_noise
        self.if_test_adv = if_test_adv
        self.loss_type = loss_type
        self.do_noise_normalization = do_noise_normalization
        self.noise_norm = noise_norm

        linear = nn.Linear(inputSize, classNum)
        ac = nn.LogSoftmax(dim=1)
        logistic = [linear, ac]
        # self.logistic = nn.Sequential(*[self.linear1, self.ac1, self.linear])#, self.ac])
        self.logistic = nn.Sequential(*logistic)
        self.protected_logistic = nn.Sequential(*[nn.Linear(inputSize,2)])
        # self.protected_logistic = nn.Sequential(*[nn.Linear(inputSize,2), nn.LogSoftmax(dim=1)])
        weights = torch.load(protectedClassifierPath)
        for key, weight in weights.items():
            if key[9:] in self.protected_logistic.state_dict().keys():
                print(f'load weight of {key[9:]}')
                self.protected_logistic.state_dict()[key[9:]].copy_(weight)

        # print(self.protected_logistic.state_dict().keys())
        # exit()
        self.loss = nn.NLLLoss()
        self.metrics = {'profession_accuracy': BooleanAccuracy(),
                        'f1': F1Measure(positive_label=1),
                        'gender_accuracy_pre':BooleanAccuracy(),
                        'gender_accuracy_post': BooleanAccuracy(),
                        'TPR_GAP': TPRGAPMetric()}
        
    
    @overrides
    def forward(self, 
                vec: torch.Tensor,
                gender_labels: torch.Tensor,
                profession_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = {}
        # print(vec)
        # print(vec.shape)
        # # print(gender_labels.shape)
        # print(gender_labels)
        # # print(profession_labels.shape)
        # print(profession_labels)
        # print('--------------')
        # exit()
        if self.training:
            if self.add_attack_noise:
                vec.requires_grad = True
                loss_protected = self._inner_forward(vec, loopType='adv', gender_labels=gender_labels)['loss']
                vec.retain_grad()
                loss_protected.backward(retain_graph=True)
                attack_disturb = vec.grad.detach_()
                if self.fixProtectedParams:
                    for p in self.protected_logistic.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                if self.do_noise_normalization:
                    norm = attack_disturb.norm(dim=-1,p=2)
                    # print('--------------')
                    # print(norm)
                    norm_disturb = attack_disturb/(norm.unsqueeze(dim=-1)+1e-10)
                    disturb = self.noise_norm * norm_disturb
                    # print('--------------')
                    # print(vec.norm(dim=-1,p=2))
                    # # print(vec)
                    # print('--------------')
                    # print(disturb.norm(dim=-1,p=2))
                    # print('--------------')
                    # exit()
                    vec = vec + disturb
            output.update(self._inner_forward(vec, loopType='normal',gender_labels=gender_labels,profession_labels=profession_labels))
        else:
            if self.if_test_adv:
                vec.requires_grad = True
                loss_protected = self._inner_forward(vec, loopType='adv', gender_labels=gender_labels)['loss']
                vec.retain_grad()
                loss_protected.backward(retain_graph=True)
                attack_disturb = vec.grad.detach_()
                for p in self.protected_logistic.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                if self.do_noise_normalization:
                    norm = attack_disturb.norm(dim=-1,p=2)
                    norm_disturb = attack_disturb/(norm.unsqueeze(dim=-1)+1e-30)
                    disturb = self.noise_norm * norm_disturb
                    vec = vec + disturb
                output.update(self._inner_forward(vec, loopType='normal',gender_labels=gender_labels,profession_labels=profession_labels))
            else:
                output.update(self._inner_forward(vec, loopType='normal',gender_labels=gender_labels,profession_labels=profession_labels))
        return output
        # out = self.logistic(vec)
        # y_hat = torch.argmax(out,dim=1)
        # output = {'y_hat': y_hat}
        # self.metrics['profession_accuracy'](y_hat, profession_labels)
        # self.metrics['TPR_GAP'](y_hat,profession_labels=profession_labels, gender_labels=gender_labels)
        # # self.metrics['f1'](nn.functional.softmax(out, dim=1), profession_labels)
        # output['loss'] = self.loss(out, profession_labels)
        # # output['TPR_GAP'] = 
        # return output

    def _inner_forward(self, vec: torch.Tensor, loopType:str, gender_labels: torch.Tensor=None, profession_labels: torch.Tensor=None):
        output = {}
        if loopType == 'adv':
            res_protected = self.protected_logistic(vec)
            with torch.no_grad():
                gender_hat = torch.argmax(res_protected, dim=1)
            if gender_labels is not None:
                self.metrics['gender_accuracy_pre'](gender_hat, gender_labels)
            if self.loss_type == 'Supervised':
                # loss = nn.NLLLoss()
                loss = nn.CrossEntropyLoss()
                loss_protected = loss(res_protected, gender_labels)
                output['loss'] = loss_protected
                return output
            elif self.loss_type == 'Unsupervised':
                print('-----------------------')
                print(res_protected)
                res_softmax = nn.functional.softmax(res_protected, dim=1)
                loss = EntropyLoss()
                print('-----------------------')
                print(res_softmax)
                loss_protected = loss(res_softmax)
                print('-----------------------')
                print(loss_protected)
                print('-----------------------')
                output['loss'] = loss_protected
                return output
        elif loopType == 'normal':
            out = self.logistic(vec)
            with torch.no_grad():
                gender_hat = torch.argmax(self.protected_logistic(vec), dim=1)
            p_hat = torch.argmax(out, dim=1)
            output['p_hat'] = p_hat
            output['loss'] = self.loss(out, profession_labels)
            self.metrics['profession_accuracy'](p_hat, profession_labels)
            self.metrics['gender_accuracy_post'](gender_hat, gender_labels)
            self.metrics['TPR_GAP'](p_hat, profession_labels=profession_labels, gender_labels=gender_labels)
            return output
            

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        # p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        outputs, GAP_RMS = self.metrics['TPR_GAP'].get_metric(reset=reset)
        return {"profession_accuracy": self.metrics['profession_accuracy'].get_metric(reset=reset),
                "gender_accuracy1": self.metrics['gender_accuracy_pre'].get_metric(reset=reset),
                "gender_accuracy2": self.metrics['gender_accuracy_post'].get_metric(reset=reset),
                "GAP_RMS": GAP_RMS}
    
class EntropyLoss(nn.Module):
    def __init__(self) -> None:
        super(EntropyLoss, self).__init__()
    
    def forward(self, output):
        """
            Calculate the entropy of output as Loss to make the prediction approach uniform distribution. 
        """

        # loss = torch.mean(torch.sum(torch.log(output)*output))
        loss = torch.mean(torch.sum(torch.log(output)))
        return loss
