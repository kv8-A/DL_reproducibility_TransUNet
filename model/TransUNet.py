""" 
For reconstructung the UNet part, we look at the decoder structure defined in the paper, and existing pytorch UNet implementations and paper.
Ref UNet paper: https://arxiv.org/abs/1505.04597
Ref UNet code: https://github.com/milesial/Pytorch-UNet

We can see from the paper that the UNet architecture has:
    - 4 upsampling blocks
    - always kernel size 3, so padding is always 1, stride is assumed 1(not mentioned in paper)
    - 512 channels at the bottom
    - the following decoder channel sizes are: 256 -> 128 -> 64 -> 16
    - the channels for the skip connections are: 512 -> 256 -> 64 -> 0 NOTE: Why?
    - 3 skip connections

so the decoder blocks are:
    - inoutres  | ksize | inoutch  | skipch
    - 1/16->1/8 | 3x3   | 512->256 | 512
    - 1/8 ->1/4 | 3x3   | 256->128 | 256
    - 1/4 ->1/2 | 3x3   | 128->64  | 64 NOTE: Why not 128?
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

the segmentation head definition is taken from UNet as well:
    - conv : 16->2 | 1x1 NOTE: standard UNet uses 1x1 kernel for segmentation head while code fot TransUNet uses 3x3 (not mentioned in paper) !!

"""

import torch
import torch.nn as nn

"""
UNet Decoder block

- upsample : bilinear | resolution x2
- (add skip to input if skip is not None (using torch.cat([x, skip, dim=1])))
- conv     : (inch+skipch)->outch | 3x3
- bn       : outch
- ReLU
- conv     : outch->outch | 3x3
- bn       : outch
- ReLU
"""
class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        
        # 2x Bilinear upsampling
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear'
        )
        # 3x3 Convolutional, also accepting skip connection
        self.conv1 = nn.Conv2d(
            in_channels=(in_channels+skip_channels), 
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )
        # 3x3 Convolutional
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )
        # BatchNorm
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        # Add skip connection
        x = torch.cat([x, skip], dim=1)
        # Conv 1
        x = self.conv1(x)
        # BatchNorm 1
        x = self.bn(x)
        # ReLU 1
        x = self.relu(x)
        # Conv 2
        x = self.conv2(x)
        # BatchNorm 2
        x = self.bn(x)
        # ReLU 2
        x = self.relu(x)

""" 
The segmentation head
converts back to final output image with segmented labels

definition is taken from UNet
- conv : 16->2 | 1x1 NOTE: standard UNet uses 1x1 kernel for segmentation head while code fot TransUNet uses 3x3 (not mentioned in paper) !!
"""
class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Single convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1
        )

    def forward(self, x):
        x = self.conv(x)

        return x

"""
Reshapes the output of the encoder

from (n_patch, D) -> (D, H/16, W/16) -> (512, H/16, W/16)
"""
class ReshapeBlock(nn.Module):
    ...

class TransUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Reshape block
        self.reshapeBlock = ReshapeBlock()

        """
        Decoder blocks according to paper:
        - inoutres  | ksize | inoutch 
        - 1/16->1/8 | 3x3   | 512->256
        - 1/8 ->1/4 | 3x3   | 256->128
        - 1/4 ->1/2 | 3x3   | 128->64 
        - 1/2 ->1/1 | 3x3   | 64->16
        NOTE: what are the skip connection channels? Get it from Resnet output?
        """
        # Decoder blocks
        self.decoderBlock1 = DecoderBlock(
            in_channels=512,
            out_channels=256,
            skip_channels=...
        ),
        self.decoderBlock2 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            skip_channels=...
        ),
        self.decoderBlock3 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            skip_channels=...
        ),
        self.decoderBlock4 = DecoderBlock(
            in_channels=64,
            out_channels=16,
            skip_channels=0
        )

        # Segmentation head
        self.segmentationHead = SegmentationHead(
            in_channels=16,
            out_channels=... # 2 channels?
        )
    
    def forward(self, x):
        # Encoder
        ...

        # Reshape
        x = self.reshapeBlock(x)

        # Decoder
        x = decoderBlock1(x, skip=...)
        x = decoderBlock2(x, skip=...)
        x = decoderBlock3(x, skip=...)
        x = decoderBlock4(x, skip=...)
        
        # Segmentation head
        x = self.segmentationHead(x)

        return x