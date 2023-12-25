# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import numpy as np
from sklearn.metrics import precision_score, recall_score
from collections import Counter

@ALGORITHMS.register('fixmatchdebug')
class FixMatchDebug(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, loss_type=args.loss_type)
        # added to debug
        self.est_epoch = 4 #args.est_epoch
        self.ema_u = 0.9 #args.ema_u
        self.u_py = torch.ones(self.num_classes).cuda() / self.num_classes
        self.u_cnt = torch.zeros(self.num_classes).cuda()
        self.au_py = torch.ones(self.num_classes).cuda() / self.num_classes
        self.au_cnt = torch.zeros(self.num_classes).cuda()
        self.g_py = torch.ones(self.num_classes).cuda() / self.num_classes
        self.g_cnt = torch.zeros(self.num_classes).cuda()
        self.ag_py = torch.ones(self.num_classes).cuda() / self.num_classes
        self.ag_cnt = torch.zeros(self.num_classes).cuda()
        self.precisions_all = [[] for _ in range(self.num_classes)]
        self.recalls_all = [[] for _ in range(self.num_classes)]
        self.aprecisions_all = [[] for _ in range(self.num_classes)]
        self.arecalls_all = [[] for _ in range(self.num_classes)]
    
    def init(self, T, p_cutoff, hard_label=True, loss_type='ce'):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        assert loss_type in ['ce', 'mse']
        self.loss_type = loss_type
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        if self.it % self.num_eval_iter == 0:
            self.precisions_all = [[] for _ in range(self.num_classes)]
            self.recalls_all = [[] for _ in range(self.num_classes)]
            self.aprecisions_all = [[] for _ in range(self.num_classes)]
            self.arecalls_all = [[] for _ in range(self.num_classes)]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                '''
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
                '''
            if self.use_cat:
                feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            else:
                feat_dict = {'x_lb':feats_x_lb}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            if self.use_cat:
                # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
                
                # if distribution alignment hook is registered, call it 
                # this is implemented for imbalanced algorithm - CReST
                if self.registered_hook("DistAlignHook"):
                    probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

                # compute mask
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False).bool()

                # generate unlabeled targets using pseudo label hook
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                              logits=probs_x_ulb_w,
                                              use_hard_label=self.use_hard_label,
                                              T=self.T,
                                              softmax=False)
                if self.loss_type == 'mse':
                    if self.use_hard_label:
                        pseudo_label = F.one_hot(pseudo_label, num_classes=self.num_classes).float()
                    else:
                        pseudo_label = probs_x_ulb_w

                now_mask = torch.zeros(self.num_classes).to(y_lb.device)
                if self.epoch > self.est_epoch:
                    for k, v in Counter(pseudo_label[mask]).items():
                        now_mask[k] += v
                    if now_mask.sum() > 0:
                        now_mask = now_mask / now_mask.sum()
                        self.u_py = self.ema_u * self.u_py + (1-self.ema_u) * now_mask
                        self.u_cnt += now_mask

                gt_mask = torch.zeros(self.num_classes).to(y_lb.device)
                if self.epoch > self.est_epoch:
                    for k, v in Counter(y_ulb[mask]).items():
                        gt_mask[k] += v
                    if gt_mask.sum() > 0:
                        gt_mask = gt_mask / gt_mask.sum()
                        self.g_py = self.ema_u * self.g_py + (1-self.ema_u) * gt_mask
                        self.g_cnt += gt_mask
                
                anow_mask = torch.zeros(self.num_classes).to(y_lb.device)
                if self.epoch > self.est_epoch:
                    for k, v in Counter(pseudo_label).items():
                        anow_mask[k] += v
                    if anow_mask.sum() > 0:
                        anow_mask = anow_mask / anow_mask.sum()
                        self.au_py = self.ema_u * self.au_py + (1-self.ema_u) * anow_mask
                        self.au_cnt += anow_mask
                        
                agt_mask = torch.zeros(self.num_classes).to(y_lb.device)
                if self.epoch > self.est_epoch:
                    for k, v in Counter(y_ulb).items():
                        agt_mask[k] += v
                    if agt_mask.sum() > 0:
                        agt_mask = agt_mask / agt_mask.sum()
                        self.ag_py = self.ema_u * self.ag_py + (1-self.ema_u) * agt_mask
                        self.ag_cnt += agt_mask
                
                if self.epoch > self.est_epoch:
                    if mask.float().sum() > 0:
                        precision_all = [None for _ in range(self.num_classes)]
                        recall_all = [None for _ in range(self.num_classes)]
                        precisions = precision_score((y_ulb[mask]).tolist(), (pseudo_label[mask]).tolist(), average=None, zero_division=np.nan)
                        recalls = recall_score((y_ulb[mask]).tolist(), (pseudo_label[mask]).tolist(), average=None, zero_division=np.nan)
                        for i, c in enumerate(sorted(list(set(y_ulb[mask].tolist()).union(set(pseudo_label[mask].tolist()))))):
                            if not np.isnan(precisions[i]):
                                precision_all[c] = precisions[i]
                            if not np.isnan(recalls[i]):
                                recall_all[c] = recalls[i]
                        for c in range(self.num_classes):
                            if not precision_all[c] is None:
                                self.precisions_all[c].append(precision_all[c])
                            if not recall_all[c] is None:
                                self.recalls_all[c].append(recall_all[c])
                    
                aprecision_all = [None for _ in range(self.num_classes)]
                arecall_all = [None for _ in range(self.num_classes)]
                aprecisions = precision_score(y_ulb.tolist(), pseudo_label.tolist(), average=None, zero_division=np.nan)
                arecalls = recall_score(y_ulb.tolist(), pseudo_label.tolist(), average=None, zero_division=np.nan)
                for i, c in enumerate(sorted(list(set(y_ulb.tolist()).union(set(pseudo_label.tolist()))))):
                    if not np.isnan(aprecisions[i]):
                        aprecision_all[c] = aprecisions[i]
                    if not np.isnan(arecalls[i]):
                        arecall_all[c] = arecalls[i]
                for c in range(self.num_classes):
                    if not aprecision_all[c] is None:
                        self.aprecisions_all[c].append(aprecision_all[c])
                    if not arecall_all[c] is None:
                        self.arecalls_all[c].append(arecall_all[c])
                        
                unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                                   pseudo_label,
                                                   self.loss_type,
                                                   mask=mask.float())
            else:
                mask = torch.zeros(y_ulb.shape[0])
                unsup_loss = torch.zeros_like(sup_loss)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
                                         
        log_dict['train/u_py'] = self.u_py.tolist()
        log_dict['train/u_cnt'] = (self.u_cnt / (self.u_cnt.sum() + 1e-12)).tolist()
        log_dict['train/au_py'] = self.au_py.tolist()
        log_dict['train/au_cnt'] = (self.au_cnt / (self.au_cnt.sum() + 1e-12)).tolist()
        log_dict['train/g_py'] = self.g_py.tolist()
        log_dict['train/g_cnt'] = (self.g_cnt / (self.g_cnt.sum() + 1e-12)).tolist()
        log_dict['train/ag_py'] = self.ag_py.tolist()
        log_dict['train/ag_cnt'] = (self.ag_cnt / (self.ag_cnt.sum() + 1e-12)).tolist()
        
        log_dict['train/precisions_all'] = [np.mean(item) if len(item) else np.nan for item in self.precisions_all]
        log_dict['train/aprecisions_all'] = [np.mean(item) if len(item) else np.nan for item in self.aprecisions_all]
        log_dict['train/recalls_all'] = [np.mean(item) if len(item) else np.nan for item in self.recalls_all]
        log_dict['train/arecalls_all'] = [np.mean(item) if len(item) else np.nan for item in self.arecalls_all]
        
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--loss_type', str, 'ce'),
            #SSL_Argument('--est_epoch', int, 4),
            #SSL_Argument('--ema_u', float, 0.9),
        ]
