""" 
For reconstructing the ResNet-50 part
Ref ResNet paper: https://arxiv.org/pdf/1512.03385.pdf [paper]

The paper does not go in a lot of detail about the ResNet implementation
apart from saying the 50 layer resnet is used, pretrained on ImageNet.
Pytorch contains a pretrained ResNet-50 on ImageNet, which we can use.

The skip connections from the paper are indicated to originate from the
1/2, 1/4 and 1/8 resolution scales.

======================================================================================

For reconstructung the UNet part, we look at the decoder structure defined in the paper, and existing pytorch UNet implementations and paper.
Ref UNet paper: https://arxiv.org/abs/1505.04597 [paper]
Ref UNet code: https://github.com/milesial/Pytorch-UNet [self search for pytorch impl.]

The paper mentions some dimensions chosen for the "base" model, being the one the experiments were performed on.
- input res       = 224x224, has to be square
- patch size P    = 16
- hidden size D   = 768    NOTE: is this a typo in [paper 4.4]? According to [paper Fig1] nr of layer = 12
- nr of layers    = 12           so D = 768 (also makes more sense), but 4.4 has them switched
- MLP size        = 3072
- number of heads = 12

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
import numpy as np

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

- conv : 16->2 | 1x1 [UNet]
"""
class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Single convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, # NOTE: kernel size not mentioned in paper, assuming 1 from UNet ref code
            padding=0,
            stride=1
        )

    def forward(self, x):
        x = self.conv(x)

        return x

"""
Reshapes the output of the transformer encoder
from (n_patch, D) -> (D, H/P, W/P) -> (512, H/P, W/P) [paper]
This block includes the first 3x3 convolution

output of encoder is (N, n_patch, D) with N batch size, D hidden size
input has to be square images, so H = W
we know n_patch = HW/P^2 = H^2/P^2 => H/P = sqrt(n_patch)

using a 1x1 convolution [paper]
"""
class ReshapeBlock(nn.Module):

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
class TransUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder blocks: ResNet-50

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
            out_channels=9 # NOTE: assumption because using 9 classes
        )
    
    def forward(self, x):
        # Encoder
        ...

        # Reshape
        x = self.reshapeBlock(x)

        # Decoder
        x = self.decoderBlock1(x, skip=...)
        x = self.decoderBlock2(x, skip=...)
        x = self.decoderBlock3(x, skip=...)
        x = self.decoderBlock4(x, skip=None)
        
        # Segmentation head
        x = self.segmentationHead(x)

        return x


""" 
The CNN part of the hybrid encoder is ResNet50 pretrained on ImageNet
This is conveniently available in pytorch torchvision

To extract the correct layers from the pretrained resnet we look at [paper Fig1],
which is the only limited detail we have for the ResNet part.

From [paper Fig1]: 3 hight level layers from ResNet, each containing a 1/2 downsampling
Downsampling in ResNet is done by means of stride=2 convolutions [resnet]
Looking at the structure of torchvision ResNet

"""
resnet = torchvision.models.resnet50(pretrained=True)
resnet = list(resnet.children())

# conv(stride=2)->bn->relu
resnetBlock1 = nn.Sequential(*resnet[0:3])
# maxpool->resnetLayer1->resnetLayer2(stride=2)
resnetBlock2 = nn.Sequential(*resnet[3:6])

resnetBlock3 = nn.Sequential(*resnet[6:7])

# print(resnetBlock1)

# ===

resnet50 = torchvision.models.resnet50(pretrained=True)

data = np.load('dataset/Synapse/train_npz/case0005_slice040.npz')
data_image = data['image']
data_label = data['label']


output = resnet50(torch.tensor([data_image, data_image]))