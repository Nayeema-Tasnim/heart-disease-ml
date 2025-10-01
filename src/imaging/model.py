import torch.nn as nn
from torchvision import models
class ResNetBinary(nn.Module):
 def __init__(self,arch='resnet18',pretrained=False):
  super().__init__()
  if arch=='resnet18':
   self.backbone=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
   in_f=self.backbone.fc.in_features; self.backbone.fc=nn.Identity()
  else: raise ValueError('Unsupported arch')
  self.head=nn.Linear(in_f,1)
 def forward(self,x):
  feat=self.backbone(x); logit=self.head(feat).squeeze(1); return logit,feat
