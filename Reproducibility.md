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
1. Bilinear 2x upsample
2. 3x3 Conv2d
3. Cascading the skip connection
4. ReLU

There is however an inconsisticy between the UNet paper and TransUNet paper. In UNet every decoder block has an intermediate step resulting in two convoultions per block. Also UNet uses a batchnorm layer after every convolution. None of this is mentioned in [^transunet], so only the layers mentioned in 3.2 of [^transunet] are used in the `DecoderBlock`. See the pseudocode underneath for the structure that is used for the decoder block.

```python
# model/TransUNet.py

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        self.upsample = 2x bilinear upsampling
        self.conv = 3x3 2d convolutional, in_channels -> out_channels + skip_channels
        self.relu = ReLU

    def forward(self, x, skip):
        x = self.upsample(x)
        concatenate skip connection to x
        x = self.conv(x)
        x = self.relu(x)

        return x
```

After these upsampling block, a final convolution acts as the segmentation head, just like in UNet. The TransUNet paper mentions a 1x1 convolution that outputs the number of classes as output channels. This corresponds with a standard UNet segmentation head and is thus implemented accordingly in `SegmentationHead`. See the pseudocode underneath for the structure that is used for the segmentation head.

```python
# model/TransUNet.py

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv = 1x1 2d convolution, in_channels -> out_channels

    def forward(self, x):
        x = self.conv(x)

        return x
```

Next to the decoder blocks, a reshaping step takes place before passing the data to the cascaded decoder. This reshape step is described in the paper as reshaping the feature from one dimension of `n_patches` to two dimensions of `H/P x W/P` with `P` the patch size. The `ReshapeBlock` also implements the additional convolution shown in Fig1 of the paper. The pseudocode underneath shows the structure that is used for the reshape block.

```python
# model/TransUNet.py

class ReshapeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv = 3x3 2d convolution, in_channels -> out_channels
        self.relu = ReLU
    
    def forward(self, x):
        get dimensions of x -> N, n_patch, D
        get value of new dimension H/P = sqrt(n_patch)
        reshape x using H/P
        reorder dimensions of x according to Fig1 of the paper 
        
        x = self.conv(x)
        x = self.relu(x)

        return x
```

## Reproducing The CNN Encoder (ResNet)
The first part of the TransUNet encoder is a CNN, in this case it is ResNet-50 [^transunet]. The paper mentiones that ResNet-50 is pretrained on ImageNet. Conveniently, PyTroch contains a pretrained ResNet50 on ImageNet, which will be used here. This CNN also needs to provide the skip connections to the decoder. Fig1 of the paper shows 3 blocks in the CNN encoder part which represent 3 high level layers of the ResNet from which the skip connections are output.
The paper does not go in a lot of detail about the ResNet implementation, so the only information we have to determine which layers of ResNet these are is that every skip connection has a certain resolution scale. ResNet uses stride=2 convolution layers to downsample, so looking at the structure of the PyTroch ResNet the correct high level layers should be derived.

First of all, only the feature extractors of ResNet are needed, so the last 2 layers which act as the classifiers, `AdaptiveAvgPool2d` and `Linear`, can be discarded. Next, the suitable layers of ResNet need to be split up in blocks so the skip output can be passed to the decoder later. When looking at the layers, there are more downsampling steps in ResNet-50 than are needed for the skip connections, 5 to be excact: `Conv2d(s=2)`, `MaxPool2d(s=2)`, `Conv2d(s=2)`, `Conv2d(s=2)` and `Conv2d(s=2)` in that order. Assumptions need to be made on how to split up the layers. Here is is assumed that the entire ResNet-50 layer count needs to be preserved, so that results in 4 blocks in total. The first 3 blocks contain one downsampling step each and are the blocks presented in Fig1 of the paper. The last block contains the remaining two downsampling steps and will be included in the CNN, but will not provide skip connections. The blocks are splitted right before the downsamplign step of the next block.

With the CNN blocks defined, also the input channels for the cascaded decoder can be determined by looking at the output channel count of the three first blocks. This results in the following skip channels: 64 - 256 - 512

```python
# model/TransUNet.py

# Encoder blocks: ResNet-50
resnet = torchvision.models.resnet50(pretrained=True)
# Observe the location of the downsampling layers
print(resnet)
resnet = list(resnet.children())
# conv(stride=2)->bn->relu
resnetBlock1 = nn.Sequential(*resnet[0:3]) # -> 64 channels
# maxpool(stride=2)->resnetLayer1
resnetBlock2 = nn.Sequential(*resnet[3:5]) # -> 256 channels
# resnetLayer2(stride=2)
resnetBlock3 = nn.Sequential(*resnet[5])   # -> 512 channels
# resnetLayer3(stride=2)->resnetLayer4(stride=2)
resnetBlock4 = nn.Sequential(*resnet[6:8])
```

## Reproducing Transformer Encoder

From the CNN encoder that was reproduced in the section before, the output of this CNN can now be used to create the embedded space which is used as input to the transformer.
 
The transformer encoder from the TransUnet is implimented from the paper An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929 . 
This paper is therefore also studied and used for the reproducability of the TransUNet paper, as the original code is written for Tensorflow, it is now to us to convert it to PyTorch.
For this Pytorch implementation a github libary was found '''ref this'''. This libary is used as a the main help for this implementation. 


## Final Reproduced TransUNet

```python
```

## The Dataset
The dataset the paper uses for the Table1 experiments is the Synapse multi-organ segmentation dataset [^synapse]. This dataset consist of 30 abdominal CT scans, every scan consisting of a number of (85 ~ 198) axial contrast-enhanced abdominal clinical CT images [^transunet]. This results in a total of 3779 images with segmentation labels. The labels span across 8 different abdominal organs (aorta, gallbladder, spleen, left kidney, right kidney, liver, pancreas, spleen, stomach).

The dataset is randomly split into 18 scans (2211 slices) for training and 12 scans (1568 slices) for validation. The raw dataset is available from Synapse directly [^synapse], however this paper performed preprocessing on the data which makes it workable in pytorch. The preprocedd data was provided by the authors and consists of `.npz` files for the training, being one singel channel image and label per file. The validation data are `.npy.h5` files being a batch of singel channel images and labels per file (one scan).

The dataset is implemented in PyTorch and can read the numpy compatible files from a specified directory. Note that the paper also specifies that random transformations are performed on the data, these are random roration and random flipping, the latter is assumed to be both horizontal and vertical flipping. These transformations are also implemented in the dataset. The pseudocode below shows how the dataset class is constructed.

```python
# dataset/Synapse.py

class Synapse(data.Dataset):
    def __init__(self, data_dir, mode):
        self.transforms = [
            random rotation,
            random vertical flip,
            random horizontal flip,
            resize to 224, 224
        ])
        self.data_list = get list of filenames from data_dir (to not load everything in memory at once)

    def __getitem__(self, i):
        get the image and label at the self.data_list[i] 
        apply self.transforms to image and label

        return image, label
```

## Training and Validation

## Bibliography

[^transunet] Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo,
Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, and Yuyin Zhou (2021). TransUNet: Transformers Make Strong
Encoders for Medical Image Segmentation. 	arXiv:2102.04306

[^unet] Ronneberger, O., Fischer, P., Brox, T. (2015) U-Net: Convolutional Networks for Biomedical
Image Segmentation. arXiv:1505.04597

[^unetcode] https://github.com/milesial/Pytorch-UNet

[^synapse] https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

[^pytorch_implementation_transformers] https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py 

