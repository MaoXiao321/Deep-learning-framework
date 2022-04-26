# -*- coding: utf-8 -*-
"""
使用预训练的模型，并学会更改模型结构
"""
# 公众号：土堆碎念
import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)  # 加载预训练模型的网络结构
# vgg16_true = torchvision.models.vgg16(pretrained=True)  # 加载预训练模型
# print(vgg16_false)

vgg16_false.classifier.add_module('add_linear', nn.Linear(1000, 10))  # 在classifier的最后加一层，名字是add_linear
# print(vgg16_false)

vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 更改指定层
print(vgg16_false)
