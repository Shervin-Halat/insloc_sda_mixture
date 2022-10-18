# from __future__ import division

# import torch
import torch.nn as nn

# from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
#                         build_sampler, merge_aug_bboxes, merge_aug_masks,
#                         multiclass_nms)
# from .. import builder
# from ..registry import DETECTORS
# from .base import BaseDetector
# from .test_mixins import RPNTestMixin
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import os
from mmdet.ops.context_block import ContextBlock        ###################

class channel_attention(nn.Module):
    def __init__(self):
        super(channel_attention,self).__init__()
        M = 4
        d = 128
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(256, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, 256, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, num_levels):
        feats_U = torch.sum(inputs, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors,dim=1)

        attention_vectors = attention_vectors.view(-1, num_levels, 256, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        return attention_vectors

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention,self).__init__()
        M = 4
        d = 128
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.ModuleList([])
        for i in range(M):
            self.conv.append(
                 nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, num_levels):
        inputs = torch.sum(inputs, dim=1)
        avg_out = torch.mean(inputs, dim=1, keepdim=True)
        max_out, _ = torch.max(inputs, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        attention_vectors = [conv(x) for conv in self.conv]
        attention_vectors = torch.cat(attention_vectors,dim=1)
        attention_vectors = self.softmax(attention_vectors)
        return attention_vectors.unsqueeze(2)


class selective_attention(nn.Module):

    def __init__(self, refine_level):
        super(selective_attention,self).__init__()
        self.refine_level = refine_level
        self.channel_att = channel_attention()
        self.spatial_att = spatial_attention()
        self.refine = ContextBlock(256, 1./16)
        
    def forward(self, inputs):
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        num_levels = len(inputs)
        for i in range(num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)
        feats = torch.cat(feats, dim=1)
        feats = feats.view(feats.shape[0], num_levels, 256, feats.shape[2], feats.shape[3])
        
        channel_attention_vectors = self.channel_att(feats,num_levels)
        feats_C = torch.sum(feats*channel_attention_vectors, dim=1)
        
        spatial_attention_vectors = self.spatial_att(feats,num_levels)
        feats_S = torch.sum(feats*spatial_attention_vectors, dim=1)

        feats_sum = feats_C + feats_S
        bsf = self.refine(feats_sum)
        
        residual = F.adaptive_max_pool2d(bsf, output_size=gather_size)
        return residual + inputs[self.refine_level]
'''
def attention(fpn_outputs):
    .
    .
    .
    return ...

def fpn2rpn(fpn_outputs):

    rpn_inputs = attention(fpn_output)

    return rpn_inputs
'''
