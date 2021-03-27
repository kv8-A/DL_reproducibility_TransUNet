""" 
For reconstructung the UNet part, we look at the decoder structure defined in the paper, and existing pytorch UNet implementations and paper.
Ref UNet paper: https://arxiv.org/abs/1505.04597 # [paper]
Ref UNet code: https://github.com/milesial/Pytorch-UNet # [self search for pytorch impl.]

We can see from the paper that the UNet architecture has:
    - 4 upsampling blocks
    - always kernel size 3, so padding is always 1, stride is assumed 1(not mentioned in paper)
    - 512 channels at the bottom
    - the following decoder channel sizes are: 256 -> 128 -> 64 -> 16
    - 3 skip connections

according to UNet ref code, a decoder block consists of: upsample->conv->bn->relu->conv->bn->relu
however the paper suggests a decoder block of: upsample->conv_relu
according to the paper, the upsampling is done bilinearly

the segmentation head definition is taken from UNet ref code as well
as it is not discussed in the paper:
    - conv : 16->2 | 1x1

"""

import torch
import torchvision
import torch.nn as nn

"""
Decoder block

According to paper, made clear in text and by Fig1, they use for 1 of the 4 blocks:
1. upsample : bilinear | resolution x2
2. (add skip connection) # [UNet, paper Fig1]
3. conv     : (inch+skipch)->outch | 3x3
4. ReLU

NOTE: deviation from UNet paper: UNet adds batchnorm before relu [UNet code]
      and has an intermediate step resulting in another conv->bn->relu added to the sequence
"""
class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        
        # 2x Bilinear upsampling
        self.upsample = nn.Upsample(
            scale_factor=2, # [paper]
            mode='bilinear' # [paper]
        )
        # 3x3 Convolutional, accepting the skip connection
        self.conv = nn.Conv2d(
            in_channels=(in_channels+skip_channels),
            out_channels=out_channels,
            kernel_size=3, # [paper]
            padding=1,
            stride=1
        )
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        # Add skip connection
        if skip != None: x = torch.cat([x, skip], dim=1)
        # Conv
        x = self.conv(x)
        # ReLU
        x = self.relu(x)

""" 
The segmentation head
converts input back to final output image with segmented labels

- conv : 16->2 | 1x1 # [UNet code]
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

"""

"""
class TransUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Reshape block
        self.reshapeBlock = ReshapeBlock()

        """
        Decoder block channels according to paper:
        - inoutch 
        - 512->256
        - 256->128
        - 128->64 
        - 64->16
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
            out_channels=3 # assumtopn that output is rgb
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
        x = decoderBlock4(x, skip=None)
        
        # Segmentation head
        x = self.segmentationHead(x)

        return x