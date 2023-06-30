#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-09
'''

import torch


def compute_mask(v, padding_idx=0):
    """
    compute mask on given tensor v
    :param v:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(v, padding_idx).float()
    return mask


def masked_flip(v, mask, flip_dim=0):
    """
    flip a tensor
    :param v: (batch, ..., ...), batch first, input batch with padding values
    :param mask: (batch, seq_len), show whether padding index
    :param flip_dim: dim to flip on
    :return:
    """
    length = mask.data.eq(1).long().sum(1)
    batch_size = v.shape[0]

    flip_list = []
    for i in range(batch_size):
        cur_tensor = v[i, :, :]
        cur_length = length[i]

        idx = list(range(cur_length - 1, -1, -1)) + list(range(cur_length, v.shape[flip_dim]))
        idx = v.new_tensor(idx, dtype=torch.long)

        cur_inv_tensor = cur_tensor.unsqueeze(0).index_select(flip_dim, idx).squeeze(0)
        flip_list.append(cur_inv_tensor)

    inv_tensor = torch.stack(flip_list, dim=0)

    return inv_tensor


def masked_softmax(x, mask=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param mask:
    :param dim:
    :return:
    """
    if mask is not None:
        mask = mask.float()
        x = x * mask
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if mask is not None:
        e_x = e_x * mask
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax

def masked_logits(target, mask):
    """
    Logits with mask
    :param target:
    :param mask:
    :return:
    """
    mask = mask.type(torch.float32)
    return target + (1 - mask) * (-1e30)