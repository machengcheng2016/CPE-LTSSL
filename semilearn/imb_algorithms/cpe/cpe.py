# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument
from sklearn.metrics import precision_score, recall_score
from collections import Counter


class CBNNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        # auxiliary classifier
        self.classifier2 = nn.Linear(self.backbone.num_features, num_classes)
        self.classifier3 = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        feat = results_dict['feat']
        
        # additional 2 (wrn_cbn does not have additional heads, so we should add them here)
        logits2_all = self.classifier2(feat)
        c = logits2_all.shape[0] // 3
        logitsH2 = logits2_all[:c,:]
        logitsM2 = logits2_all[c:c*2,:]
        logitsT2 = logits2_all[c*2:,:]
        logits_2 = (logitsH2 + logitsM2 + logitsT2) / 3
        
        results_dict['logitsH2'] = logitsH2
        results_dict['logitsM2'] = logitsM2
        results_dict['logitsT2'] = logitsT2
        results_dict['logits_2'] = logits_2
        
        # additional 3 (wrn_cbn does not have additional heads, so we should add them here)
        logits3_all = self.classifier3(feat)
        c = logits3_all.shape[0] // 3
        logitsH3 = logits3_all[:c,:]
        logitsM3 = logits3_all[c:c*2,:]
        logitsT3 = logits3_all[c*2:,:]
        logits_3 = (logitsH3 + logitsM3 + logitsT3) / 3
        
        results_dict['logitsH3'] = logitsH3
        results_dict['logitsM3'] = logitsM3
        results_dict['logitsT3'] = logitsT3
        results_dict['logits_3'] = logits_3
        
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@IMB_ALGORITHMS.register('cpe')
class CPE(ImbAlgorithmBase):
    """
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
        super(CPE, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        # TODO: better ways
        self.model = CBNNet(self.model, num_classes=self.num_classes)
        self.ema_model = CBNNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()
        
        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.p_hat_lb = torch.from_numpy(lb_class_dist / lb_class_dist.sum()).cuda()
        self.tau_lb1 = args.la_tau_lb1
        self.tau_lb2 = args.la_tau_lb2
        self.tau_lb3 = args.la_tau_lb3
        self.est_epoch = args.est_epoch
        self.ema_u = args.ema_u
        self.cut1 = args.cut1
        self.cut2 = args.cut2
        # For recording frequencies, precisions, and recalls
        self.u_py1 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.u_py2 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.u_py3 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.au_py1 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.au_py2 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.au_py3 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.g_py1 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.g_py2 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.g_py3 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.ag_py1 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.ag_py2 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.ag_py3 = torch.ones(self.num_classes).cuda() / self.num_classes
        self.precisions_all1 = [[] for _ in range(self.num_classes)]
        self.precisions_all2 = [[] for _ in range(self.num_classes)]
        self.precisions_all3 = [[] for _ in range(self.num_classes)]
        self.recalls_all1 = [[] for _ in range(self.num_classes)]
        self.recalls_all2 = [[] for _ in range(self.num_classes)]
        self.recalls_all3 = [[] for _ in range(self.num_classes)]
        self.aprecisions_all1 = [[] for _ in range(self.num_classes)]
        self.aprecisions_all2 = [[] for _ in range(self.num_classes)]
        self.aprecisions_all3 = [[] for _ in range(self.num_classes)]
        self.arecalls_all1 = [[] for _ in range(self.num_classes)]
        self.arecalls_all2 = [[] for _ in range(self.num_classes)]
        self.arecalls_all3 = [[] for _ in range(self.num_classes)]

    def update_stuff(self, stuff, pseudo_label, mask=None):
        now_mask = torch.zeros(self.num_classes).cuda()
        if mask is None:
            thing = pseudo_label
        else:
            thing = pseudo_label[mask]
        for k, v in Counter(thing).items():
            now_mask[k] += v
        if now_mask.sum() > 0:
            now_mask = now_mask / now_mask.sum()
            stuff = self.ema_u * stuff + (1-self.ema_u) * now_mask
        return stuff
            
    def update_precision_recall(self, precisions_all, recalls_all, aprecisions_all, arecalls_all, pseudo_label, y_ulb, mask):
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
                    precisions_all[c].append(precision_all[c])
                if not recall_all[c] is None:
                    recalls_all[c].append(recall_all[c])
                            
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
                aprecisions_all[c].append(aprecision_all[c])
            if not arecall_all[c] is None:
                arecalls_all[c].append(arecall_all[c])
            
        
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        if self.it % self.num_eval_iter == 0:
            self.precisions_all1 = [[] for _ in range(self.num_classes)]
            self.precisions_all2 = [[] for _ in range(self.num_classes)]
            self.precisions_all3 = [[] for _ in range(self.num_classes)]
            self.recalls_all1 = [[] for _ in range(self.num_classes)]
            self.recalls_all2 = [[] for _ in range(self.num_classes)]
            self.recalls_all3 = [[] for _ in range(self.num_classes)]
            self.aprecisions_all1 = [[] for _ in range(self.num_classes)]
            self.aprecisions_all2 = [[] for _ in range(self.num_classes)]
            self.aprecisions_all3 = [[] for _ in range(self.num_classes)]
            self.arecalls_all1 = [[] for _ in range(self.num_classes)]
            self.arecalls_all2 = [[] for _ in range(self.num_classes)]
            self.arecalls_all3 = [[] for _ in range(self.num_classes)]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb1 = outputs['logits'][:num_lb]
                logits_x_ulb_w1, logits_x_ulb_s1 = outputs['logits'][num_lb:].chunk(2)
                _, logits_x_ulb_sH1 = outputs['logitsH'][num_lb:].chunk(2)
                _, logits_x_ulb_sM1 = outputs['logitsM'][num_lb:].chunk(2)
                _, logits_x_ulb_sT1 = outputs['logitsT'][num_lb:].chunk(2)
                
                logits_x_lb2 = outputs['logits_2'][:num_lb]
                logits_x_ulb_w2, logits_x_ulb_s2 = outputs['logits_2'][num_lb:].chunk(2)
                _, logits_x_ulb_sH2 = outputs['logitsH2'][num_lb:].chunk(2)
                _, logits_x_ulb_sM2 = outputs['logitsM2'][num_lb:].chunk(2)
                _, logits_x_ulb_sT2 = outputs['logitsT2'][num_lb:].chunk(2)
                
                logits_x_lb3  = outputs['logits_3'][:num_lb]
                logits_x_ulb_w3, logits_x_ulb_s3 = outputs['logits_3'][num_lb:].chunk(2)
                _, logits_x_ulb_sH3 = outputs['logitsH3'][num_lb:].chunk(2)
                _, logits_x_ulb_sM3 = outputs['logitsM3'][num_lb:].chunk(2)
                _, logits_x_ulb_sT3 = outputs['logitsT3'][num_lb:].chunk(2)
            else:
                raise ValueError('use_cat should be True')
                
            feat_dict = {}
            
            # First Head: FixMatch w/ tau1 * Logit Adjustment
            sup_loss1 = self.ce_loss(logits_x_lb1 + self.tau_lb1 * torch.log(self.p_hat_lb), y_lb, reduction='mean')
            probs_x_ulb_w1 = self.compute_prob(logits_x_ulb_w1.detach())
            mask1 = probs_x_ulb_w1.amax(dim=-1).ge(self.p_cutoff)
            pseudo_label1 = probs_x_ulb_w1.argmax(dim=-1)
            pseudo_label1H = F.one_hot(pseudo_label1, self.num_classes).sum(dim=1) * mask1.float()
            pseudo_label1M = F.one_hot(pseudo_label1, self.num_classes)[:,self.cut1:].sum(dim=1) * mask1.float()
            pseudo_label1T = F.one_hot(pseudo_label1, self.num_classes)[:,self.cut2:].sum(dim=1) * mask1.float()
            unsup_loss1 =  (self.ce_loss(logits_x_ulb_sH1, pseudo_label1, reduction='none') * pseudo_label1H).sum()
            unsup_loss1 += (self.ce_loss(logits_x_ulb_sM1, pseudo_label1, reduction='none') * pseudo_label1M).sum()
            unsup_loss1 += (self.ce_loss(logits_x_ulb_sT1, pseudo_label1, reduction='none') * pseudo_label1T).sum()
            unsup_loss1 /= (pseudo_label1H.sum() + pseudo_label1M.sum() + pseudo_label1T.sum() + 1e-12)
            
            # Second Head: FixMatch w/ tau2 * Logit Adjustment
            sup_loss2 = self.ce_loss(logits_x_lb2 + self.tau_lb2 * torch.log(self.p_hat_lb), y_lb, reduction='mean')
            probs_x_ulb_w2 = self.compute_prob(logits_x_ulb_w2.detach())
            mask2 = probs_x_ulb_w2.amax(dim=-1).ge(self.p_cutoff)
            pseudo_label2 = probs_x_ulb_w2.argmax(dim=-1)
            pseudo_label2H = F.one_hot(pseudo_label2, self.num_classes).sum(dim=1) * mask2.float()
            pseudo_label2M = F.one_hot(pseudo_label2, self.num_classes)[:,self.cut1:].sum(dim=1) * mask2.float()
            pseudo_label2T = F.one_hot(pseudo_label2, self.num_classes)[:,self.cut2:].sum(dim=1) * mask2.float()
            unsup_loss2 =  (self.ce_loss(logits_x_ulb_sH2, pseudo_label2, reduction='none') * pseudo_label2H).sum()
            unsup_loss2 += (self.ce_loss(logits_x_ulb_sM2, pseudo_label2, reduction='none') * pseudo_label2M).sum()
            unsup_loss2 += (self.ce_loss(logits_x_ulb_sT2, pseudo_label2, reduction='none') * pseudo_label2T).sum()
            unsup_loss2 /= (pseudo_label2H.sum() + pseudo_label2M.sum() + pseudo_label2T.sum() + 1e-12)
            
            # Third Head: FixMatch w/ tau3 * Logit Adjustment 
            sup_loss3 = self.ce_loss(logits_x_lb3 + self.tau_lb3 * torch.log(self.p_hat_lb), y_lb, reduction='mean')
            probs_x_ulb_w3 = self.compute_prob(logits_x_ulb_w3.detach())
            mask3 = probs_x_ulb_w3.amax(dim=-1).ge(self.p_cutoff)
            pseudo_label3 = probs_x_ulb_w3.argmax(dim=-1)
            pseudo_label3H = F.one_hot(pseudo_label3, self.num_classes).sum(dim=1) * mask3.float()
            pseudo_label3M = F.one_hot(pseudo_label3, self.num_classes)[:,self.cut1:].sum(dim=1) * mask3.float()
            pseudo_label3T = F.one_hot(pseudo_label3, self.num_classes)[:,self.cut2:].sum(dim=1) * mask3.float()
            unsup_loss3 =  (self.ce_loss(logits_x_ulb_sH3, pseudo_label3, reduction='none') * pseudo_label3H).sum()
            unsup_loss3 += (self.ce_loss(logits_x_ulb_sM3, pseudo_label3, reduction='none') * pseudo_label3M).sum()
            unsup_loss3 += (self.ce_loss(logits_x_ulb_sT3, pseudo_label3, reduction='none') * pseudo_label3T).sum()
            unsup_loss3 /= (pseudo_label3H.sum() + pseudo_label3M.sum() + pseudo_label3T.sum() + 1e-12)
            
            # To compute frequencies, precision, and recalls
            if self.epoch > self.est_epoch and self.use_cat:
                self.u_py1 = self.update_stuff(self.u_py1, pseudo_label1, mask1)
                self.u_py2 = self.update_stuff(self.u_py2, pseudo_label2, mask2)
                self.u_py3 = self.update_stuff(self.u_py3, pseudo_label3, mask3)
                self.g_py1 = self.update_stuff(self.g_py1, y_ulb, mask1)
                self.g_py2 = self.update_stuff(self.g_py2, y_ulb, mask2)
                self.g_py3 = self.update_stuff(self.g_py3, y_ulb, mask3)
                self.au_py1 = self.update_stuff(self.au_py1, pseudo_label1, None)
                self.au_py2 = self.update_stuff(self.au_py2, pseudo_label2, None)
                self.au_py3 = self.update_stuff(self.au_py3, pseudo_label3, None)
                self.ag_py1 = self.update_stuff(self.ag_py1, y_ulb, None)
                self.ag_py2 = self.update_stuff(self.ag_py2, y_ulb, None)
                self.ag_py3 = self.update_stuff(self.ag_py3, y_ulb, None)
                self.update_precision_recall(self.precisions_all1, self.recalls_all1, self.aprecisions_all1, self.arecalls_all1, pseudo_label1, y_ulb, mask1)
                self.update_precision_recall(self.precisions_all2, self.recalls_all2, self.aprecisions_all2, self.arecalls_all2, pseudo_label2, y_ulb, mask2)
                self.update_precision_recall(self.precisions_all3, self.recalls_all3, self.aprecisions_all3, self.arecalls_all3, pseudo_label3, y_ulb, mask3)
            
            total_loss = (sup_loss1 + self.lambda_u * unsup_loss1) + (sup_loss2 + self.lambda_u * unsup_loss2) + (sup_loss3 + self.lambda_u * unsup_loss3)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss2.item(), 
                                         unsup_loss=unsup_loss2.item(), 
                                         total_loss=(sup_loss2 + self.lambda_u * unsup_loss2).item(), 
                                         util_ratio=mask2.float().mean().item())
                                         
        log_dict['train/u_py1'] = self.u_py1.tolist()
        log_dict['train/u_py2'] = self.u_py2.tolist()
        log_dict['train/u_py3'] = self.u_py3.tolist()
        log_dict['train/au_py1'] = self.au_py1.tolist()
        log_dict['train/au_py2'] = self.au_py2.tolist()
        log_dict['train/au_py3'] = self.au_py3.tolist()
        log_dict['train/g_py1'] = self.g_py1.tolist()
        log_dict['train/g_py2'] = self.g_py2.tolist()
        log_dict['train/g_py3'] = self.g_py3.tolist()
        log_dict['train/ag_py1'] = self.ag_py1.tolist()
        log_dict['train/ag_py2'] = self.ag_py2.tolist()
        log_dict['train/ag_py3'] = self.ag_py3.tolist()
        
        log_dict['train/precisions_all1'] = [np.mean(item) if len(item) else np.nan for item in self.precisions_all1]
        log_dict['train/precisions_all2'] = [np.mean(item) if len(item) else np.nan for item in self.precisions_all2]
        log_dict['train/precisions_all3'] = [np.mean(item) if len(item) else np.nan for item in self.precisions_all3]
        log_dict['train/aprecisions_all1'] = [np.mean(item) if len(item) else np.nan for item in self.aprecisions_all1]
        log_dict['train/aprecisions_all2'] = [np.mean(item) if len(item) else np.nan for item in self.aprecisions_all2]
        log_dict['train/aprecisions_all3'] = [np.mean(item) if len(item) else np.nan for item in self.aprecisions_all3]
        log_dict['train/recalls_all1'] = [np.mean(item) if len(item) else np.nan for item in self.recalls_all1]
        log_dict['train/recalls_all2'] = [np.mean(item) if len(item) else np.nan for item in self.recalls_all2]
        log_dict['train/recalls_all3'] = [np.mean(item) if len(item) else np.nan for item in self.recalls_all3]
        log_dict['train/arecalls_all1'] = [np.mean(item) if len(item) else np.nan for item in self.arecalls_all1]
        log_dict['train/arecalls_all2'] = [np.mean(item) if len(item) else np.nan for item in self.arecalls_all2]
        log_dict['train/arecalls_all3'] = [np.mean(item) if len(item) else np.nan for item in self.arecalls_all3]
        
        return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='logits_2', return_logits=return_logits)
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--la_tau_lb1', float, 0.0),
            SSL_Argument('--la_tau_lb2', float, 2.0),
            SSL_Argument('--la_tau_lb3', float, 4.0),
            SSL_Argument('--est_epoch', int, 0),
            SSL_Argument('--ema_u', float, 0.9),
            SSL_Argument('--cut1', float, 2),
            SSL_Argument('--cut2', float, 4),
        ]        
