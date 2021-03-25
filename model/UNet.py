import torch.nn as nn

""" 
For reconstructung the UNet part, we look at the decoder structure defined in the paper.
Ref code UNet: https://github.com/milesial/Pytorch-UNet

We can see from the paper that the UNet architecture has:
    - 4 upsampling blocks
    - always kernel size 3, so padding is always 1, stride is assumed 1(not mentioned in paper)
    - 512 channels at the bottom
    - the following decoder channel sizes are: 256 -> 128 -> 64 -> 16
    - the channels for the skip connections are: 512 -> 256 -> 64 -> 0
    - 3 skip connections

so the decoder blocks are:
    - inoutres  | ksize | inoutch  | skipch
    - 1/16->1/8 | 3x3   | 512->256 | 512
    - 1/8 ->1/4 | 3x3   | 256->128 | 256
    - 1/4 ->1/2 | 3x3   | 128->64  | 64
    - 1/2 ->1/1 | 3x3   | 64->16   | 0

according to UNet, a decoder block consists of: upsample->conv->bn->relu->conv->bn->relu
according to the paper, the upsampling is done bilinearly

so a single decoder block consists of: (ref. UNet)
    - upsample : bilinear | resolution x2
    - (add skip to input if skip is not None (using torch.cat([x, skip, dim=1])))
    - conv     : (inch+skipch)->outch | 3x3
    - bn       : outch
    - ReLU
    - conv     : outch->outch | 3x3
    - bn       : outch
    - ReLU

the segmentation head definition is taken from UNet:
    - conv : 16->2 | 1x1 NOTE: standard UNet uses 1x1 kernel for segmentation head while code fot TransUNet uses 3x3 (not mentioned in paper) !!

"""
class UNet(nn.Module):
