# TransUNet Reproducibility Report

## Introduction

## TransUNet Architecture
![](/figs/architecture.png)

## Reproducing The Cascaded Decoder (UNet)
TransUNet’s decoder is presented in the paper as the decoder part of UNet, shown in Fig1 of the paper as the right hand side of the network. Its task is to decode and upsample the encoded features that come out of the hybrid CNN-Transformer encoder. Just like in UNet skip connections are introduced in order to preserve information across the resolution scales. For reproducing this part of the network, we solely use the information provided in the paper, the reference paper of UNet [^unet] and reference code of UNet implementation in PyTorch [^unetcode].

From Fig1 of the paper the UNet architecture can be derived as follows:
- The decoder consists of 4 main upsampling blocks.
- The kernel size of the convolutions is always 3x3, with a ReLU nonlinearity.
- The decoder starts with 512 input channels, which result from a reshaping step at the bottom of the encoder.
- The output channels of the decoder blocks are: 256 - 128 - 64 - 16 respectively.
- There are 3 skip connections that originate from the CNN encoder (ResNet) at resolution scales ½, ¼ and ⅛ .

Using this information and [^unetcode] a pytorch module can be coded that represents one decoder block. This decoder block then consists of:
- Bilinear 2x upsample
- 3x3 Conv2d
- Cascading the skip connection
- ReLU

There is however an inconsisticy between the UNet paper and TransUNet paper. In UNet every decoder block has an intermediate step resulting in two convoultions per block. Also UNet uses a batchnorm layer after every convolution. None of this is mentioned in [^transunet], so only the layers mentioned in 3.2 of [^transunet] are used in the `DecoderBlock`.

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        
        # 2x Bilinear upsampling
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear'
        )
        # 3x3 Convolutional, accepting the skip connection
        self.conv = nn.Conv2d(
            in_channels=(in_channels+skip_channels),
            out_channels=out_channels,
            kernel_size=3,
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
```

After these upsampling block, a final convolution acts as the segmentation head, just like in UNet. The TransUNet paper mentions a 1x1 convolution that outputs the number of classes as output channels. This corresponds with a standard UNet segmentation head and is thus implemented accordingly in `SegmentationHead`.

```python
class SegmentationHead(nn.Module):
    """ 
    The segmentation head
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
```

Next to the decoder blocks, a reshaping step takes place before passing the data to the cascaded decoder. This reshape step is described in the paper as reshaping the feature from one dimension of `n_patches` to two dimensions of `H/P x W/P` with `P` the patch size. The `ReshapeBlock` also implements the additional convolution shown in Fig1 of the paper.

```python
class ReshapeBlock(nn.Module):
    """
    Reshapes the output of the transformer encoder
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
```


## Reproducing The CNN Encoder (ResNet)

## Reproducing Transformer Encoder

## Final Reproduced TransUNet

```python
```

## Bibliography

[^transunet] Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo,
Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, and Yuyin Zhou (2021). TransUNet: Transformers Make Strong
Encoders for Medical Image Segmentation. 	arXiv:2102.04306

[^unet] Ronneberger, O., Fischer, P., Brox, T. (2015) U-Net: Convolutional Networks for Biomedical
Image Segmentation. arXiv:1505.04597

[^unetcode] https://github.com/milesial/Pytorch-UNet