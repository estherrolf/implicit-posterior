
import torch as T
import numpy as np

def ce_loss(preds, prior):
    return -(preds * prior).sum(1), T.Tensor([0])

def comb_loss(preds, prior):
    return -(preds.softmax(1) * prior / prior.sum(1).unsqueeze(1)).sum(1).log(), T.Tensor([0])

def qr_loss(preds, prior):
    eps = 0.000000001
    logprior = prior.clamp(min=eps).log()
    r = (preds.log_softmax(0) + logprior).log_softmax(1)
    return -(r * preds.exp()).sum(1), -(preds * preds.exp()).sum(1)

def rq_loss(preds, prior):
    eps = 0.000000001
    logprior = prior.clamp(min=eps).log()
    r = (preds.log_softmax(0) + logprior).log_softmax(1)
    return -(preds * r.exp()).sum(1), -(r * r.exp()).sum(1)