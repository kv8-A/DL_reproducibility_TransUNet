"""
Ref UNet paper: https://arxiv.org/abs/1505.04597 [paper]
Ref UNet code: https://github.com/milesial/Pytorch-UNet [self search for pytorch impl.]
Ref ResNet paper: https://arxiv.org/pdf/1512.03385.pdf [paper]
"""

import torch
import torchvision
import torch.nn as nn
import numpy as np

class DecoderBlock(nn.Module):
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


class SegmentationHead(nn.Module):
    """ 
    The segmentation head
    converts input back to final output image with segmented labels

    - conv : 16->2 | 1x1 [UNet]
    """

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


class ReshapeBlock(nn.Module):
    """
    Reshapes the output of the transformer encoder
    from (n_patch, D) -> (D, H/P, W/P) -> (512, H/P, W/P) [paper]
    This block includes the first 3x3 convolution

    output of encoder is (N, n_patch, D) with N batch size, D hidden size
    input has to be square images, so H = W
    we know n_patch = HW/P^2 = H^2/P^2 => H/P = sqrt(n_patch)

    using a 1x1 convolution [paper]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 3x3 Convolution
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )
        # ReLU activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape
        N, n_patch, D = x.size() # [paper Fig1]
        h_P = int(np.sqrt(n_patch)) # H/P [paper 3.1]
        x = x.view(N, h_P, h_P, D) # reshape using value for H/P (W/P = H/P)
        x = x.permute(0, 2, 3, 1) # reorder dimensions according to [paper Fig1]
        # Conv
        x = self.conv(x)
        # ReLU
        x = self.relu(x)

        return x

class TransUNet(nn.Module):
    """
    TransUNet Network

    Consisting of hybrid CNN-Transformer encoder
    and UNet cascaded decoder with skip connections

    Network architectire derived from [paper Fig1]

    The paper mentions some dimensions chosen for the "base" model, being the one the experiments were performed on.
    - input res       = 224x224
    - patch size P    = 16
    - hidden size D   = 768    NOTE: is there a typo in [paper 4.4]? According to [paper Fig1] nr of layer = 12
    - nr of layers    = 12           so D = 768 (also makes more sense), but 4.4 has them switched around
    - MLP size        = 3072
    - number of heads = 12
    """

    def __init__(self):
        super().__init__()

        # Encoder blocks: ResNet-50
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet = list(resnet.children())
        # conv(stride=2)->bn->relu
        self.resnetBlock1 = nn.Sequential(*resnet[0:3]) # -> 64 channels
        # maxpool(stride=2)->resnetLayer1
        self.resnetBlock2 = nn.Sequential(*resnet[3:5]) # -> 256 channels
        # resnetLayer2(stride=2)
        self.resnetBlock3 = nn.Sequential(*resnet[5])   # -> 512 channels
        # resnetLayer3(stride=2)->resnetLayer4(stride=2)
        self.resnetBlock4 = nn.Sequential(*resnet[6:8])

        # Transformer
        # self.transformer = TransformerBlock(...)

        # Reshape block
        self.reshapeBlock = ReshapeBlock(
            in_channels=768,
            out_channels=512
        )

        """
        Decoder block channels according to paper:
        - inoutch 
        - 512->256
        - 256->128
        - 128->64 
        - 64->16
        NOTE: what are the skip connection channels? Get it from Resnet output? Not clear form paper
        """
        # Decoder blocks
        self.decoderBlock1 = DecoderBlock(
            in_channels=512,
            out_channels=256,
            skip_channels=512 # from resnetBlock3
        ),
        self.decoderBlock2 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            skip_channels=256 # from resnetBlock2
        ),
        self.decoderBlock3 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            skip_channels=64 # from resnetBlock1
        ),
        self.decoderBlock4 = DecoderBlock(
            in_channels=64,
            out_channels=16,
            skip_channels=0 # no skip connection
        )

        # Segmentation head
        self.segmentationHead = SegmentationHead(
            # in_channels=16,
            in_channels=2048,
            out_channels=9 # NOTE: assumption because using 9 classes
        )

    
    def forward(self, x):
        if x.shape[1] == 1: # if grayscale
            x = x.repeat(1, 3, 1, 1) # always have 3 channels

        # Encoder
        x1 = self.resnetBlock1(x)
        x2 = self.resnetBlock2(x1)
        x3 = self.resnetBlock3(x2)
        x  = self.resnetBlock4(x3) 

        # # Transformer
        # x = self.transformer(x)

        # # Reshape
        # x = self.reshapeBlock(x)

        # # Decoder
        # x = self.decoderBlock1(x, skip=x3)
        # x = self.decoderBlock2(x, skip=x2)
        # x = self.decoderBlock3(x, skip=x1)
        # x = self.decoderBlock4(x, skip=None)

        test = nn.Upsample(
            scale_factor=32,
            mode='bilinear',
            align_corners=False
        )
        x = test(x)
        
        # Segmentation head
        x = self.segmentationHead(x)

        return x