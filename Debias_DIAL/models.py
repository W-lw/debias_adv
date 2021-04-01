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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.lstm import LSTMHardSigmoid
from torchmoji.attlayer import Attention
from torchmoji.global_variables import NB_TOKENS, NB_EMOJI_CLASSES
from typing import Dict
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import BooleanAccuracy, F1Measure
from allennlp.models import Model
import torch
import torch.nn as nn
import numpy as np
# from torch.nn import Module
import copy
from TPR_GAP import TPRGAPMetric
import os

@Model.register("race_classify")
class RaceClassifierWithMojiEncoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 TorchMojiPath: str,
                 fixMojiParams: bool,
                 hidden_size: int,
                 emb_size: int = 2304) -> None:
        super().__init__(vocab)
        self.mojiEncoder = torchmoji_feature_encoding(TorchMojiPath)
        if fixMojiParams:
            for p in self.mojiEncoder.parameters():
                p.requires_grad=False
            # self.mojiEncoder.eval()
        layers = []
        layers.append(nn.Linear(emb_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        self.scorer = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.metrics = {'accuracy': BooleanAccuracy(),
                        'f1': F1Measure(positive_label=1)}

    @overrides
    def forward(self, 
                vec: torch.Tensor,
                race_label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        vec: torch.Tensor, required
            The input vector.
        label: torch.Tensor, optional (default = None)
            A variable of the correct label.

        Returns
        -------
        An output dictionary consisting of:
        y_hat: torch.FloatTensor
            the predicted values
        loss: torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        #print(vec.is_leaf)#true
        enc_moji = self.mojiEncoder(vec)
        #print(enc_moji.shape)  #32 x 2304
        scores = self.scorer(enc_moji)
        # print(scores.shape)#32 x 2
        y_hat = torch.argmax(scores, dim=1)

        output = {"y_hat": y_hat}
        if race_label is not None:
            self.metrics['accuracy'](y_hat, race_label)
            self.metrics['f1'](torch.nn.functional.softmax(scores, dim=1), race_label)
            output["loss"] = self.loss(scores, race_label)

        return output
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        return {"accuracy": self.metrics['accuracy'].get_metric(reset=reset),
                "p": p,
                "r": r,
                "f1": f1}

@Model.register("sent_classify")
class SentClassifierWithMojiEncoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 TorchMojiPath: str,
                 fixMojiParams: bool,
                 hidden_size: int,
                 emb_size: int = 2304) -> None:
        super().__init__(vocab)
        self.mojiEncoder = torchmoji_feature_encoding(TorchMojiPath)
        if fixMojiParams:
            for p in self.mojiEncoder.parameters():
                p.requires_grad = False
        layers = []
        layers.append(nn.Linear(emb_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        self.scorer = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.metrics = {'accuracy': BooleanAccuracy(),
                        'f1': F1Measure(positive_label=1)}
    @overrides
    def forward(self, 
                vec: torch.Tensor,
                sent_label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        vec: torch.Tensor, required
            The input vector.
        label: torch.Tensor, optional (default = None)
            A variable of the correct label.

        Returns
        -------
        An output dictionary consisting of:
        y_hat: torch.FloatTensor
            the predicted values
        loss: torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        enc_moji = self.mojiEncoder(vec)
        scores = self.scorer(enc_moji)
        y_hat = torch.argmax(scores, dim=1)

        output = {"y_hat": y_hat}
        if sent_label is not None:
            self.metrics['accuracy'](y_hat, sent_label)
            self.metrics['f1'](torch.nn.functional.softmax(scores, dim=1), sent_label)
            output["loss"] = self.loss(scores, sent_label)

        return output
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        return {"accuracy": self.metrics['accuracy'].get_metric(reset=reset),
                "p": p,
                "r": r,
                "f1": f1}

@Model.register("main_adv_classify")
class MainAdversarialClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 TorchMojiPath: str,
                 ProtectClassifierPath: str,
                 fixMojiParams: bool,
                 fixProtectParams: bool,
                 add_attack_noise: bool,
                 if_test_adv: bool,
                 hidden_size: int,
                 loss_type: str,
                 adv_layer: str = "hidden",
                 do_noise_normalization: bool = True,
                 noise_norm: Optional[float] = None,
                 emb_size: int=2304) -> None:
        super().__init__(vocab)
        self.add_attack_noise = add_attack_noise
        self.do_noise_normalization = do_noise_normalization
        self.noise_norm = noise_norm
        self.if_test_adv = if_test_adv
        self.loss_type = loss_type
        self.mojiEncoder = torchmoji_feature_encoding(TorchMojiPath)
        self.fixMojiParams = fixMojiParams
        self.fixProtectParams = fixProtectParams
        self.adv_layer = adv_layer
        if self.fixMojiParams:
            #fix the parameters of mojiEncoder
            for p in self.mojiEncoder.parameters():
                p.requires_grad = False
        layers = []
        layers.append(nn.Linear(emb_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        main_layers = copy.deepcopy(layers)
        self.protectClassifier = nn.Sequential(*layers)
        weights = torch.load(ProtectClassifierPath)#,map_location=lambda storage, loc: storage)
        
        #load weight for protected classifier
        for key, weight in weights.items():
            #the key in loaded weights is like "scorer.0.weight" instead of "0.weight"
            if key[7:] in self.protectClassifier.state_dict().keys():
                print(f"load weight of {key[7:]}")
                self.protectClassifier.state_dict()[key[7:]].copy_(weight)
        # if self.fixProtectParams:
        #     #fix the parameters of protected Attribution classifier
        #     for p in self.protectClassifier.parameters():
        #         p.requires_grad = False

        self.scorer = nn.Sequential(*main_layers)
        self.loss = nn.CrossEntropyLoss()
        self.metrics = {'sent_accuracy': BooleanAccuracy(),
                        'race_accuracy_pre': BooleanAccuracy(),
                        'race_accuracy_post': BooleanAccuracy(),
                        'TPR_GAP': TPRGAPMetric(),
                        'f1': F1Measure(positive_label=1)}
    def forward(self, 
                vec: torch.Tensor,
                sent_label: torch.Tensor,
                race_label: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        #encode sentence 
        enc_moji1 = self.mojiEncoder(vec)
        with torch.no_grad():
            race_hat1 = torch.argmax(self.protectClassifier(enc_moji1),dim=1)
        
        enc_moji2=None

        if self.training:
            if self.add_attack_noise:
                if self.adv_layer=="hidden":
                    #if we fix the parameters of mojiEncoder, we should set the requires_grad to true manually.
                    if self.fixMojiParams:
                        enc_moji1.requires_grad=True
                    loss_protected = self._inerforward(enc_moji1,race_label)
                    enc_moji1.retain_grad()
                    loss_protected.backward(retain_graph=True)
                    attack_disturb = enc_moji1.grad.detach_()
                    #if we want to fix the parameters of protected classifier, we should clear the grad of protectClassifier manually
                    if self.fixProtectParams:
                        for p in self.protectClassifier.parameters():
                            if p.grad is not None:
                                p.grad.detach_()
                                p.grad.zero_()
                    #clear the grad of mojiEncoder
                    for p in self.mojiEncoder.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                    enc_moji2 = self.mojiEncoder(vec)
                    # attack_disturb = torch.randn(enc_moji1.shape).to(device=enc_moji1.device)
                    if self.do_noise_normalization:
                        norm = attack_disturb.norm(dim=-1,p=2)
                        norm_disturb = attack_disturb/(norm.unsqueeze(dim=-1)+1e-10)
                        disturb = self.noise_norm * norm_disturb
                        enc_moji2 = enc_moji2 + disturb
                    
                elif self.adv_layer=="embed":
                    # if self.fixMojiParams:
                    #     enc_moji1.requires_grad=True
                    loss_protected = self._inerforward(enc_moji1, race_label)
                    loss_protected.backward(retain_graph=True)
                    attack_disturb = self.mojiEncoder.embed.weight.grad.detach_()
                    #if we want to fix the parameters of protected classifier, we should clear the grad of protectClassifier manually
                    if self.fixProtectParams:
                        for p in self.protectClassifier.parameters():
                            if p.grad is not None:
                                p.grad.detach_()
                                p.grad.zero_()
                    #clear the grad of mojiEncoder
                    for p in self.mojiEncoder.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                    #disturb the embedding layer
                    if self.do_noise_normalization:
                        norm = attack_disturb.norm(dim=-1,p=2)
                        norm_disturb = attack_disturb/(norm.unsqueeze(dim=-1)+1e-10)
                        disturb = self.noise_norm * norm_disturb
                        
                        self.mojiEncoder.embed.weight = self.mojiEncoder.embed.weight + disturb
                    enc_moji2 = self.mojiEncoder(vec)
                    print(enc_moji1.norm(dim=1,p=2))
                    print(enc_moji2.norm(dim=1,p=2))
                    exit()
            else:
                enc_moji2 = enc_moji1
        else:
            if self.if_test_adv:
                if self.fixMojiParams:
                    enc_moji1.requires_grad = True
                loss_protected = self._inerforward(enc_moji1, race_label)
                enc_moji1.retain_grad()
                loss_protected.backward(retain_graph=True)
                attack_disturb = enc_moji1.grad.detach_()
                #in the test, we must clear the grad of classifier
                for p in self.protectClassifier.parameters():
                    if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                #clear the grad of mojiEncoder
                for p in self.mojiEncoder.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                enc_moji2 = self.mojiEncoder(vec)
                if self.do_noise_normalization:
                    norm = attack_disturb.norm(dim=-1,p=2)
                    norm_disturb = attack_disturb/(norm.unsqueeze(dim=-1)+1e-10)
                    disturb = self.noise_norm * norm_disturb
                    enc_moji2 = enc_moji2 + disturb
            else:
                enc_moji2=enc_moji1
        # enc_moji2=enc_moji1
        scores = self.scorer(enc_moji2)
        y_hat = torch.argmax(scores, dim=1)
        output = {"y_hat": y_hat}
        with torch.no_grad():
            race_hat2 = torch.argmax(self.protectClassifier(enc_moji2),dim=1)
        # loss_tmp = self.loss(scores, sent_label)
        # loss_tmp.backward()
        # for p in self.mojiEncoder.parameters():
        #     print(p.requires_grad)
        #     print(p.grad)
        #     print(p.is_leaf)
        # print('-------------------')
        # for p in self.scorer.parameters():
        #     print(p.requires_grad)
        #     print(p.grad)
        #     print(p.is_leaf)
        # exit()
        # print("----------------y_hat-------------------")
        # print(y_hat)
        # print("----------------race_hat-------------------")
        # print(race_hat)
        # print("----------------sent_label-------------------")
        # print(sent_label)
        # print(sent_label.shape)
        # print("----------------race_label-------------------")
        # print(race_label)
        # print(race_label.shape)
        # print("----------------vec-------------------")
        # print(vec)
        # print(vec.dtype)
        # exit()
        # print('-----------------------------------------------------------------')
        # print((enc_moji1-enc_moji2).norm(p=2))
        # print('-----------------------------------------------------------------')
        # exit()
        # print(sent_label)
        # print(race_label)
        # print(race_hat1.size())
        # print(sent_label.size())
        # print(race_label.size())
        # exit()
        if sent_label is not None:
            self.metrics['sent_accuracy'](y_hat, sent_label)
            self.metrics['f1'](torch.nn.functional.softmax(scores, dim=1), sent_label)
            output["loss"] = self.loss(scores, sent_label)
        if race_label is not None:
            self.metrics['race_accuracy_pre'](race_hat1,race_label)
        if race_label is not None:
            self.metrics['race_accuracy_post'](race_hat2,race_label)
        if race_label is not None and  sent_label is not None:
            self.metrics['TPR_GAP'](y_hat, sent_label, race_label)

        return output

    def _inerforward(self,
                     enc_moji: torch.Tensor,
                     race_label: torch.Tensor):
        '''
            Forward encoded embedding to the protected attribute classifier and get the loss or grad;
        '''
        res_protected = self.protectClassifier(enc_moji)
        if self.loss_type == 'CrossEntropyLoss':
            loss = nn.CrossEntropyLoss()
            loss_protected = loss(res_protected, race_label)
            return loss_protected
        if self.loss_type == "EntropyLoss":
            res_softmax = nn.functional.softmax(res_protected,dim=1)
            entropyloss = EntropyLoss()
            loss_protected = entropyloss(res_softmax)
            return loss_protected
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        '''
        
        '''
        p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        GAP_POS, GAP_NEG, GAP_RMS = self.metrics['TPR_GAP'].get_metric(reset=reset)
        return {"sent_accuracy": self.metrics['sent_accuracy'].get_metric(reset=reset),
                "race_accuracy1": self.metrics['race_accuracy_pre'].get_metric(reset=reset),
                "race_accuracy2": self.metrics['race_accuracy_post'].get_metric(reset=reset),
                "GAP_POS": GAP_POS,
                "GAP_NEG": GAP_NEG,
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
@Model.register("main_adv_embed_classify")
class AdversarialOnBottomEmbed(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 TorchMojiPath: str,
                 ProtectClassifierPath: str,
                 fixMojiParams: bool,
                 fixProtectParams: bool,
                 add_attack_noise: bool,
                 if_test_adv: bool,
                 hidden_size: int,
                 loss_type: str,
                 random_disturb:bool = False,
                 do_noise_normalization: bool = True,
                 noise_norm: Optional[float] = None,
                 if_extract_feature: bool = False,
                 emb_size: int=2304) -> None:
        super().__init__(vocab)
        self.add_attack_noise = add_attack_noise
        self.do_noise_normalization = do_noise_normalization
        self.noise_norm = noise_norm
        self.if_test_adv = if_test_adv
        self.loss_type = loss_type
        self.random_disturb = random_disturb
        # self.embeder = nn.Embedding(50000,256)
        # self.
        # self.mojiEncoder = torchmoji_feature_encoding(TorchMojiPath)
        self.if_extract_feature = if_extract_feature
        if if_extract_feature:
            self.encode_sentence = {}

        embedding_dim = 256
        self.feature_output = True
        self.embed_dropout_rate = 0
        self.final_dropout_rate = 0
        self.return_attention = False
        self.hidden_size = 512
        self.attention_size = 4*self.hidden_size+embedding_dim
        self.output_logits = False
        self.nb_classes = None
        self.embed = nn.Embedding(50000, embedding_dim)
        self.embed_dropout = nn.Dropout2d(self.embed_dropout_rate)
        self.lstm_0 = LSTMHardSigmoid(embedding_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_1 = LSTMHardSigmoid(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)
        self.attention_layer = Attention(attention_size=self.attention_size, return_attention=self.return_attention)
        self.init_encoder_weight(TorchMojiPath)
        self.mojiEncoder = [self.embed,self.lstm_0, self.lstm_1, self.attention_layer]
        self.fixMojiParams = fixMojiParams
        self.fixProtectParams = fixProtectParams
        # self.adv_layer = adv_layer
        if self.fixMojiParams:
            #fix the parameters of mojiEncoder
            for layer in self.mojiEncoder:
                for p in layer.parameters():
                    p.requires_grad = False
        
        layers = []
        layers.append(nn.Linear(emb_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        main_layers = copy.deepcopy(layers)
        self.protectClassifier = nn.Sequential(*layers)
        weights = torch.load(ProtectClassifierPath)#, map_location=lambda storage, loc: storage)
        
        #load weight for protected classifier
        for key, weight in weights.items():
            #the key in loaded weights is like "scorer.0.weight" instead of "0.weight"
            if key[7:] in self.protectClassifier.state_dict().keys():
                print(f"load weight of {key[7:]}")
                self.protectClassifier.state_dict()[key[7:]].copy_(weight)

        self.scorer = nn.Sequential(*main_layers)
        self.loss = nn.CrossEntropyLoss()
        self.metrics = {'sent_accuracy': BooleanAccuracy(),
                        'race_accuracy_pre': BooleanAccuracy(),
                        'race_accuracy_post': BooleanAccuracy(),
                        'TPR_GAP': TPRGAPMetric(),
                        'f1': F1Measure(positive_label=1)}

    def init_encoder_weight(self,TorchMojiPath):
        weights = torch.load(TorchMojiPath)#, map_location=lambda storage, loc: storage)
        for key, weight in weights.items():
            if key in self.state_dict().keys():
                print(f"load weight of {key} for mojiEncoder.")
                self.state_dict()[key].copy_(weight)
        
    def forward(self, 
                vec: torch.Tensor,
                sent_label: torch.Tensor,
                race_label: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        output = {}
        # print('-------------------------------')
        # print('vec:',vec.shape)#(batch, 150)
        # print('-------------------------------')
        # exit()

        ho = self.lstm_0.weight_hh_l0.data.new(2, vec.size()[0], self.hidden_size).zero_()
        co = self.lstm_0.weight_hh_l0.data.new(2, vec.size()[0], self.hidden_size).zero_()

        # Reorder batch by sequence length
        input_lengths = torch.LongTensor([torch.max(vec[i, :].data.nonzero()) + 1 for i in range(vec.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_seqs = vec[perm_idx][:, :input_lengths.max()]
        # print('-------------')
        # print(input_lengths)
        # print(input_seqs.shape)
        # Pack sequence and work on data tensor to reduce embeddings/dropout computations
        packed_input = pack_padded_sequence(input_seqs, input_lengths.cpu().numpy(), batch_first=True)
        # reorder_output = True
        # hidden = (Variable(ho, requires_grad=True), Variable(co, requires_grad=True))
        hidden = (Variable(ho, requires_grad=False), Variable(co, requires_grad=False))
        embedding_sequence = self.embed(packed_input.data)
        embedding_sequence = self.embed_dropout(nn.Tanh()(embedding_sequence))
        if self.training:
            if self.add_attack_noise:
                if self.fixMojiParams:
                    embedding_sequence.requires_grad = True
                loss_protected = self._inner_forward(embedding_sequence, packed_input.batch_sizes, hidden, 'adv', input_lengths, perm_idx, race_label=race_label)['loss']
                embedding_sequence.retain_grad()
                loss_protected.backward(retain_graph=True)
                attack_disturb = embedding_sequence.grad.detach_()
                if self.random_disturb:
                    attack_disturb = torch.randn(embedding_sequence.shape).to(device=embedding_sequence.device)
                #if we want to fix the parameters of protected classifier, we should clear the grad of protectClassifier manually
                if self.fixProtectParams:
                    for p in self.protectClassifier.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                for layer in self.mojiEncoder:
                    for p in layer.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                if self.do_noise_normalization:
                    norm = attack_disturb.norm(dim=-1,p=2)
                    norm_disturb = attack_disturb/(norm.unsqueeze(dim=-1)+1e-10)
                    disturb = self.noise_norm * norm_disturb
                    embedding_sequence = embedding_sequence + disturb
            output.update(self._inner_forward(embedding_sequence, packed_input.batch_sizes,hidden, 'normal', input_lengths, perm_idx, sent_label=sent_label,race_label=race_label))
        else:  
            if self.if_test_adv:
                if self.fixMojiParams:
                    embedding_sequence.requires_grad = True
                loss_protected = self._inner_forward(embedding_sequence, packed_input.batch_sizes, hidden, 'adv', input_lengths, perm_idx, race_label=race_label)['loss']
                embedding_sequence.retain_grad()
                loss_protected.backward(retain_graph=True)
                attack_disturb = embedding_sequence.grad.detach_()
                if self.random_disturb:
                    attack_disturb = torch.randn(embedding_sequence.shape).to(device=embedding_sequence.device)
                for p in self.protectClassifier.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                for layer in self.mojiEncoder:
                    for p in layer.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                if self.do_noise_normalization:
                    norm = attack_disturb.norm(dim=-1,p=2)
                    norm_disturb = attack_disturb/(norm.unsqueeze(dim=-1)+1e-10)
                    disturb = self.noise_norm * norm_disturb
                    
                    embedding_sequence = embedding_sequence + disturb
                output.update(self._inner_forward(embedding_sequence, packed_input.batch_sizes,hidden, 'normal', input_lengths, perm_idx, sent_label=sent_label,race_label=race_label))

            else:
                output.update(self._inner_forward(embedding_sequence, packed_input.batch_sizes,hidden, 'normal', input_lengths, perm_idx, sent_label=sent_label,race_label=race_label))
        return output

    def _inner_forward(self,
                       embedding_sequence,
                       batch_sizes,
                       hidden,
                       loopType: str,
                       input_lengths,
                       perm_idx,
                       sent_label: torch.Tensor=None,
                       race_label: torch.Tensor=None,
                       ):
        output = {}
        # Update packed sequence data for RNN
        packed_input = PackedSequence(embedding_sequence, batch_sizes)
        # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
        # ordering of the way the merge is done is important for consistency with the pretrained model
        lstm_0_output, _ = self.lstm_0(packed_input, hidden)
        lstm_1_output, _ = self.lstm_1(lstm_0_output, hidden)

        # Update packed sequence data for attention layer
        packed_input = PackedSequence(torch.cat((lstm_1_output.data,
                                                 lstm_0_output.data,
                                                 packed_input.data), dim=1),
                                      packed_input.batch_sizes)

        input_seqs, _ = pad_packed_sequence(packed_input, batch_first=True)
        x, att_weights = self.attention_layer(input_seqs, input_lengths)
        reorered = Variable(x.data.new(x.size()))
        reorered[perm_idx] = x
        enc_moji = reorered
        if loopType == 'adv':
            res_protected = self.protectClassifier(enc_moji)
            with torch.no_grad():
                race_hat = torch.argmax(res_protected,dim=1)
            if race_label is not None:
                self.metrics['race_accuracy_pre'](race_hat,race_label)
            if self.loss_type == 'CrossEntropyLoss':
                loss = nn.CrossEntropyLoss()
                loss_protected = loss(res_protected, race_label)
                output['loss'] = loss_protected
                return output
            if self.loss_type == "EntropyLoss":
                res_softmax = nn.functional.softmax(res_protected,dim=1)
                entropyloss = EntropyLoss()
                loss_protected = entropyloss(res_softmax)
                output['loss'] = loss_protected
                return output
        elif loopType == 'normal':
            if self.if_extract_feature:
                scores = enc_moji
                for i in range(len(self.scorer)):
                    scores = self.scorer[i](scores)
                    if i == 0:
                        enc_moji_np = copy.deepcopy(scores).cpu().numpy()
                        try:
                            self.encode_sentence['enc_moji'] = np.concatenate((self.encode_sentence['enc_moji'],enc_moji_np))
                        except KeyError:
                            self.encode_sentence['enc_moji'] = enc_moji_np

                        sent_label_np = copy.deepcopy(sent_label).cpu().numpy()
                        try:
                            self.encode_sentence['sent_label'] = np.concatenate((self.encode_sentence['sent_label'], sent_label_np))
                        except KeyError:
                            self.encode_sentence['sent_label'] = sent_label_np

                        race_label_np = copy.deepcopy(race_label).cpu().numpy()
                        try:
                            self.encode_sentence['race_label'] = np.concatenate((self.encode_sentence['race_label'], race_label_np))
                        except KeyError:
                            self.encode_sentence['race_label'] = race_label_np
            else:
                scores = self.scorer(enc_moji)
            # if self.if_extract_feature:
            if False:
                enc_moji_np = copy.deepcopy(enc_moji).cpu().numpy()
                try:
                    self.encode_sentence['enc_moji'] = np.concatenate((self.encode_sentence['enc_moji'],enc_moji_np))
                except KeyError:
                    self.encode_sentence['enc_moji'] = enc_moji_np
                
                sent_label_np = copy.deepcopy(sent_label).cpu().numpy()
                try:
                    self.encode_sentence['sent_label'] = np.concatenate((self.encode_sentence['sent_label'], sent_label_np))
                except KeyError:
                    self.encode_sentence['sent_label'] = sent_label_np

                race_label_np = copy.deepcopy(race_label).cpu().numpy()
                try:
                    self.encode_sentence['race_label'] = np.concatenate((self.encode_sentence['race_label'], race_label_np))
                except KeyError:
                    self.encode_sentence['race_label'] = race_label_np

            with torch.no_grad():
                race_hat = torch.argmax(self.protectClassifier(enc_moji),dim=1)
            y_hat = torch.argmax(scores, dim=1)
            output = {"y_hat": y_hat}
            self.metrics['sent_accuracy'](y_hat, sent_label)
            self.metrics['f1'](torch.nn.functional.softmax(scores, dim=1), sent_label)
            output["loss"] = self.loss(scores, sent_label)
            self.metrics['race_accuracy_post'](race_hat,race_label)
            self.metrics['TPR_GAP'](y_hat, sent_label, race_label)
            return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        GAP_POS, GAP_NEG, GAP_RMS = self.metrics['TPR_GAP'].get_metric(reset=reset)
        return {"sent_accuracy": self.metrics['sent_accuracy'].get_metric(reset=reset),
                "race_accuracy1": self.metrics['race_accuracy_pre'].get_metric(reset=reset),
                "race_accuracy2": self.metrics['race_accuracy_post'].get_metric(reset=reset),
                "GAP_POS": GAP_POS,
                "GAP_NEG": GAP_NEG,
                "GAP_RMS": GAP_RMS}

    def saveFeature(self, savePath):
        if self.if_extract_feature:
            np.save(os.path.join(savePath,'enc_moji_bftanh.npy'),self.encode_sentence['enc_moji'])
            np.save(os.path.join(savePath,'race_label.npy'),self.encode_sentence['race_label'])
            np.save(os.path.join(savePath,'sent_label.npy'),self.encode_sentence['sent_label'])
            print("Saved sentence feature!")
        else:
            print("Done!")
