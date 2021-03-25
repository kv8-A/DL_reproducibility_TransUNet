import torch.nn as nn

""" 
From paper:

Input resolution = 224x224
Patch size = 16

Encoder design:
ResNet-50 and ViT, pretrained in ImageNet

Decoder design: Cascaded Upsampler CUP
4 2x upsampling blocks

Skip connections at all 3 intermediate upsampling steps except the output layer
so at following resolution scales:
    - 1/2
    - 1/4
    - 1/8

"""

class TransUNet(nn.Module):
    ...