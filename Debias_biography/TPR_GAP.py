# here put the import lib
from enum import Flag
import torch
from overrides import overrides
from allennlp.training.metrics.metric import Metric

@Metric.register("tpr_gap")
class TPRGAPMetric(Metric):
    def __init__(self) -> None:
        self.results = {}
        for i in range(28):
            self.results[str(i)+'male'] = 0.
            self.results[str(i)+'female'] = 0.
            self.results[str(i)+'male_num'] = 0.
            self.results[str(i)+'female_num'] = 0.

    
    def __call__(self,
                 predictions: torch.Tensor,
                 profession_labels: torch.Tensor,
                 gender_labels: torch.Tensor):# m:0 f:1
        predictions, profession_labels, gender_labels = self.unwrap_to_tensors(predictions, profession_labels, gender_labels)
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, -1)
        profession_labels = profession_labels.view(batch_size, -1)
        gender_labels = gender_labels.view(batch_size, -1)
        # print('---------------')
        # print(predictions)
        # print(profession_labels)
        # print(gender_labels)
        # print('---------------')
        # exit()

        for i in range(28):
            self.results[str(i)+'male'] += ((profession_labels==i)*(predictions==i)*(gender_labels==0)).float().sum()
            self.results[str(i)+'female'] += ((profession_labels==i)*(predictions==i)*(gender_labels==1)).float().sum()
            self.results[str(i)+'male_num'] += ((profession_labels==i)*(gender_labels==0)).float().sum()
            self.results[str(i)+'female_num'] += ((profession_labels==i)*(gender_labels==1)).float().sum()

        #     print(str(i)+'male:',self.results[str(i)+'male'])
        #     print(str(i)+'female:',self.results[str(i)+'female'])
        #     print(str(i)+'male_num:',self.results[str(i)+'male_num'])
        #     print(str(i)+'female_num:',self.results[str(i)+'female_num'])
        # exit()
    def get_metric(self, reset: bool=False):
        outputs = {}
        for i in range(28):
            m_acc = 0.0
            f_acc = 0.0
            if self.results[str(i)+'male_num'] > 0:
                m_acc = self.results[str(i)+'male']/self.results[str(i)+'male_num']
            if self.results[str(i)+'female_num'] > 0:
                f_acc = self.results[str(i)+'female']/self.results[str(i)+'female_num']
            outputs[str(i)+'_gap'] = m_acc - f_acc
        gaps = torch.tensor(list(outputs.values()))
        GAP_RMS = torch.sqrt(torch.mean(gaps**2))
        if GAP_RMS>1:
            print(outputs)
        if reset:
            self.reset()
        return outputs, GAP_RMS.item()
    
    @overrides
    def reset(self) -> None:
        x=0
        for i in range(28):
            # print(str(i)+'male:',self.results[str(i)+'male'])
            # print(str(i)+'female:',self.results[str(i)+'female'])
            # print(str(i)+'male_num:',self.results[str(i)+'male_num'])
            # print(str(i)+'female_num:',self.results[str(i)+'female_num'])
            x+=self.results[str(i)+'female_num']+self.results[str(i)+'male_num']
            self.results[str(i)+'male'] = 0.
            self.results[str(i)+'female'] = 0.
            self.results[str(i)+'male_num'] = 0.
            self.results[str(i)+'female_num'] = 0.
        print(x)