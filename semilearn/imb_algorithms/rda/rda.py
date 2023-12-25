# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument
from .utils import consistency_loss_rda

class RDANet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        # auxiliary classifier
        self.reverse_classifier = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_reverse'] = self.reverse_classifier(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@IMB_ALGORITHMS.register('rda')
class RDA(ImbAlgorithmBase):
    """
        RDA algorithm (https://arxiv.org/abs/2208.04619).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):

        super(RDA, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        # distri & distri_reverse
        self.distri = torch.ones((128, self.num_classes)).cuda()
        self.distri_reverse = torch.ones((128, self.num_classes)).cuda()
        self.count = 0
        
        # TODO: better ways
        self.model = RDANet(self.model, num_classes=self.num_classes)
        self.ema_model = RDANet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, *args, **kwargs):
        x_lb, y_lb, x_ulb_w, x_ulb_s = kwargs['x_lb'], kwargs['y_lb'], kwargs['x_ulb_w'], kwargs['x_ulb_s']
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                logits_reverse_separate = outputs['logits_reverse'][:num_lb]
                logits_x_ulb_w_reverse, logits_x_ulb_s_reverse = outputs['logits_reverse'][num_lb:].chunk(2)
            else:
                raise ValueError('self.use_cat should be True')
                            
            pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
            pseudo_label_cls_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)
            
            # construct complementary label
            res = [0] * len(y_lb)
            all_possible_labels = set(range(self.num_classes))
            for each in range(len(res)):
                potential_anws = all_possible_labels - set([y_lb[each].item()])
                res[each] = np.random.choice(list(potential_anws))
            res_torch = torch.cuda.LongTensor(res)

            self.distri[self.count] = pseudo_label.detach().mean(0)
            self.distri_reverse[self.count] = pseudo_label_cls_reverse.detach().mean(0)
            self.count = (self.count + 1) % 128
            
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            reverse_loss = self.ce_loss(logits_reverse_separate, res_torch, reduction='mean')
            
            unsup_loss_cd, unsup_loss_ca = consistency_loss_rda(
                                            logits_x_ulb_w_reverse,
                                            logits_x_ulb_s_reverse,
                                            logits_x_ulb_w, 
                                            logits_x_ulb_s, 
                                            self.distri,
                                            self.distri_reverse)
                                              
            total_loss = sup_loss + reverse_loss + self.lambda_u * unsup_loss_cd + self.lambda_u * unsup_loss_ca

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss_cd.item(), 
                                         total_loss=total_loss.item())

        log_dict['unsup_loss_cd'] = unsup_loss_cd.item()
        log_dict['unsup_loss_ca'] = unsup_loss_ca.item()
        
        return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='logits', return_logits=return_logits)
