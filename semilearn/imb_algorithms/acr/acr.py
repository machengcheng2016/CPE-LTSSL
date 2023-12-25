# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS, get_optimizer, get_cosine_schedule_with_warmup
from semilearn.algorithms.utils import SSL_Argument
from .utils import interleave, de_interleave


class ACRNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        # auxiliary classifier
        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_aux'] = self.aux_classifier(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@IMB_ALGORITHMS.register('acr')
class ACR(ImbAlgorithmBase):
    """
        ACR algorithm (https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Towards_Realistic_Long-Tailed_Semi-Supervised_Learning_Consistency_Is_All_You_Need_CVPR_2023_paper.pdf).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - py_con (`float`):
                ...
            - py_uni (`float`):
                ...
            - py_rev (`float`):
                ...
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super(ACR, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(lb_class_dist / lb_class_dist.sum())
        
        self.imb_init(args)
        
        # TODO: better ways
        self.model = ACRNet(self.model, num_classes=self.num_classes)
        self.ema_model = ACRNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()
        

    def imb_init(self, args):
        self.py_con = self.lb_class_dist.cuda()
        self.py_uni = torch.ones(self.num_classes).cuda() / self.num_classes
        self.py_rev = torch.flip(self.py_con, dims=[0])
        
        self.est_step = 0
        
        self.adjustment_l1 = torch.log(self.py_con ** args.tau1 + 1e-12)
        self.adjustment_l12 = torch.log(self.py_con ** args.tau12 + 1e-12)
        self.adjustment_l2 = torch.log(self.py_con ** args.tau2 + 1e-12)
        
        self.taumin = 0
        self.taumax = args.tau1
        
        self.est_epoch = args.est_epoch
        self.num_eval_iter = args.num_eval_iter
        self.KL_div = nn.KLDivLoss(reduction='sum')        
        
        self.ema_u = args.ema_u
        self.u_py = torch.ones(self.num_classes).cuda() / self.num_classes

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")
            
            if self.epoch > self.est_epoch:
                self.count_KL = self.count_KL / self.num_eval_iter
                KL_softmax = (torch.exp(self.count_KL[0])) / (torch.exp(self.count_KL[0])+torch.exp(self.count_KL[1])+torch.exp(self.count_KL[2]))
                self.tau = self.taumin + (self.taumax - self.taumin) * KL_softmax
                if not torch.isnan(self.tau):
                    self.adjustment_l1 = torch.log(self.py_con ** self.tau + 1e-12)
            else:
                self.tau = self.taumax
                    
            self.count_KL = torch.zeros(3)

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
        
    def train_step(self, *args, **kwargs):
        x_lb, y_lb, x_ulb_w, x_ulb_s = kwargs['x_lb'], kwargs['y_lb'], kwargs['x_ulb_w'], kwargs['x_ulb_s']
        
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                #inputs = interleave(torch.cat((x_lb, x_ulb_w, x_ulb_s)), 1+2*2)
                outputs = self.model(inputs)
                logits_x = outputs['logits'][:num_lb]
                logits_u_w, logits_u_s = outputs['logits'][num_lb:].chunk(2)
                #logits = de_interleave(outputs['logits'], 1+2*2)
                #logits_x = logits[:num_lb]
                #logits_u_w, logits_u_s = logits[num_lb:].chunk(2)
                logits_x_b = outputs['logits_aux'][:num_lb]
                logits_u_w_b, logits_u_s_b = outputs['logits_aux'][num_lb:].chunk(2)
                #logits_b = de_interleave(outputs['logits_aux'], 1+2*2)
                #logits_x_b = logits_b[:num_lb]
                #logits_u_w_b, logits_u_s_b = logits_b[num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x = outs_x_lb['logits']
                logits_x_b = outs_x_lb['logits_aux']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_u_s = outs_x_ulb_s['logits']
                logits_u_s_b = outs_x_ulb_s['logits_aux']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_u_w = outs_x_ulb_w['logits']
                    logits_u_w_b = outs_x_ulb_w['logits_aux']

            Lx = self.ce_loss(logits_x, y_lb, reduction='mean')
            Lx_b = self.ce_loss(logits_x_b + self.adjustment_l2, y_lb, reduction='mean')
            
            pseudo_label = self.compute_prob(logits_u_w.detach() - self.adjustment_l1)
            pseudo_label_h2 = self.compute_prob(logits_u_w.detach() - self.adjustment_l12)
            pseudo_label_b = self.compute_prob(logits_u_w_b.detach())
            pseudo_label_t = self.compute_prob(logits_u_w.detach())

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            max_probs_h2, targets_u_h2 = torch.max(pseudo_label_h2, dim=-1)
            max_probs_b, targets_u_b = torch.max(pseudo_label_b, dim=-1)
            max_probs_t, _ = torch.max(pseudo_label_t, dim=-1)

            mask = max_probs.ge(self.p_cutoff)
            mask_h2 = max_probs_h2.ge(self.p_cutoff)
            mask_b = max_probs_b.ge(self.p_cutoff)
            mask_t = max_probs_t.ge(self.p_cutoff)

            mask_ss_b_h2 = (mask_b + mask_h2).float()
            mask_ss_t = (mask + mask_t).float()
            
            now_mask = torch.zeros(self.num_classes).to(y_lb.device)
            
            if self.epoch > self.est_epoch:
                now_mask[targets_u_b] += mask_b.float()
                self.est_step += 1
                
                if now_mask.sum() > 0:
                    now_mask = now_mask / now_mask.sum()
                    self.u_py = self.ema_u * self.u_py + (1-self.ema_u) * now_mask
                    KL_con = 0.5 * self.KL_div(self.py_con.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_con)
                    KL_uni = 0.5 * self.KL_div(self.py_uni.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_uni)
                    KL_rev = 0.5 * self.KL_div(self.py_rev.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_rev)
                    self.count_KL[0] = self.count_KL[0] + KL_con
                    self.count_KL[1] = self.count_KL[1] + KL_uni
                    self.count_KL[2] = self.count_KL[2] + KL_rev
            
            # targets_u is computed with dynamic self.adjustment_l1 
            Lu = self.consistency_loss(logits_u_s,
                                       targets_u,
                                       'ce',
                                       mask=mask_ss_t)

            # targets_u_h2 is computed with fixed self.adjustment_l12
            Lu_b = self.consistency_loss(logits_u_s_b,
                                         targets_u_h2,
                                         'ce',
                                         mask=mask_ss_b_h2)

            total_loss = Lx + Lx_b + Lu + Lu_b
            
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=Lx.item(), 
                                         unsup_loss=Lu.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())

        log_dict['train/Lx_b'] = Lx_b.item()
        log_dict['train/Lu_b'] = Lu_b.item()
        log_dict['train/tau'] = self.tau

        return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='logits_aux', return_logits=return_logits)
        
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--tau1', float, 2),
            SSL_Argument('--tau12', float, 2),
            SSL_Argument('--tau2', float, 2),
            SSL_Argument('--est_epoch', int, 5),
            SSL_Argument('--ema_u', float, 0.9),
        ]        
