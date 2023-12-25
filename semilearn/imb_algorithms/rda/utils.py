from __future__ import print_function
import torch
import torch.nn.functional as F

__all__ = ['consistency_loss_rda']

def normalize_d(x):
    x_sum = torch.sum(x)
    x = x / x_sum
    return x.detach()
        
def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss
        
def consistency_loss_rda(logits_x_ulb_w_reverse, logits_x_ulb_s_reverse, logits_w, logits_s, distri, distri_reverse):
    logits_w = logits_w.detach()
    logits_x_ulb_w_reverse = logits_x_ulb_w_reverse.detach()
    distri = distri.detach()
    distri_reverse = distri_reverse.detach()
    pseudo_label = torch.softmax(logits_w, dim=-1)
    pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)

    distri_ = torch.ones_like(distri) - distri 
    distri_ = normalize_d(distri_)      
    pseudo_label_reverse_da = normalize_d(pseudo_label_reverse * (torch.mean(distri_, dim=0) / torch.mean(distri_reverse, dim=0)))  
    distri_reverse_ = torch.ones_like(distri_reverse) - distri_reverse 
    distri_reverse_ = normalize_d(distri_reverse_)  
    pseudo_label_da = normalize_d(pseudo_label * (torch.mean(distri_reverse_, dim=0) / torch.mean(distri, dim=0)))
    max_probs, max_idx = torch.max(pseudo_label_da, dim=-1)
 
    loss_cd = ce_loss(logits_s, max_idx, use_hard_labels=True, reduction='none') 
    loss_ca = ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse_da, use_hard_labels=False, reduction='none')   

    return loss_cd.mean(), loss_ca.mean()
