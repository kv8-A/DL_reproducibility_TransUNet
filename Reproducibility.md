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

First of all, only the feature extractors of ResNet are needed, so the last 2 layers which act as the classifiers, `AdaptiveAvgPool2d` and `Linear`, can be discarded. Next, the suitable layers of ResNet need to be split up in blocks so the skip output can be passed to the decoder later. When looking at the layers, there are more downsampling steps in ResNet-50 than are needed for the skip connections, 5 to be excact: `Conv2d(s=2)`, `MaxPool2d(s=2)`, `Conv2d(s=2)`, `Conv2d(s=2)` and `Conv2d(s=2)` in that order. Assumptions need to be made on how to split up the layers. Here is is assumed that the entire ResNet-50 layer count needs to be preserved, so that results in 4 blocks in total. The first 3 blocks contain one downsampling step each and are the blocks presented in Fig1 of the paper. The last block contains the remaining two downsampling steps and will be included in the CNN, but will not provide skip connections. The blocks are splitted right before the downsamplign step of the next block. Note that incompatible dimensions arised when running the network. This was due to too many downsampming that is happening in the ResNet. The reference from the paper, only Figure 1, does not provide enough inmformation to debug this incompatibility as it only shows the 3 blocks of resnet, bot also mention the ResNet-50 using 50 layers is used. To solve this, the assumption is made to ommit the last high level layer of the ResNet, containing the extra downsampling step causing the problems. This results in compatible down and later upsampling.

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
# resnetLayer3(stride=2)
resnetBlock4 = nn.Sequential(*resnet[6:7])
```

## Reproducing Transformer Encoder

From the CNN encoder that was reproduced in the section before, the output of this CNN can now be used to create the embedded space which is used as input to the transformer.
 
The transformer encoder from the TransUnet is implimented from the paper An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929 . 
This paper is therefore also studied and used for the reproducability of the TransUNet paper, as the original code is written for Tensorflow, it is now to us to convert it to PyTorch.
For this Pytorch implementation a github libary was found '''ref this'''. This libary is used as a the main help for this implementation. 


## Final Reproduced TransUNet
The several building block of the 3 main netowrk parts: UNet, ResNet and ViT, are put together in the main TransUNet network class. The pseudocode for this class can be seen below and shows how the blocks are defined and used together. First, ResNet accepts the image input and generates the feature maps from it. Then this gets patch embedded and the transformer takes over. After the transformer steps, the reshape block reshapes after which the UNet upsample produces the final output of 9 classes/channels. The main reference for this architecture is the Figure 1 of the paper, which mentiones both the dimensions and structure of the network.

```python
# model/TransUNet.py

class TransUNet(nn.Module):
    def __init__(self):
        self.resnetBlock1 = resnetBlock1
        self.resnetBlock2 = resnetBlock1
        self.resnetBlock3 = resnetBlock1
        self.resnetBlock4 = resnetBlock1

        self.transformer = VisionTransformer

        self.reshapeBlock

        self.decoderBlock1 = DecoderBlock(
            in_channels=512,
            out_channels=256,
            skip_channels=512
        )
        self.decoderBlock2 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            skip_channels=256
        )
        self.decoderBlock3 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            skip_channels=64
        )
        self.decoderBlock4 = DecoderBlock(
            in_channels=64,
            out_channels=16,
            skip_channels=0
        )

        self.segmentationHead = SegmentationHead

    
    def forward(self, x):
        x1 = self.resnetBlock1(x)
        x2 = self.resnetBlock2(x1)
        x3 = self.resnetBlock3(x2)
        x  = self.resnetBlock4(x3)

        x = self.transformer(x)

        x = self.reshapeBlock(x)

        x = self.decoderBlock1(x, skip=x3)
        x = self.decoderBlock2(x, skip=x2)
        x = self.decoderBlock3(x, skip=x1)
        x = self.decoderBlock4(x, skip=None)

        x = self.segmentationHead(x)

        return x
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
To check the workings of the reproduced model, it is trained according to the parameters in the paper. For the training procedure, the paper mentiones a number of hyperparameters. The SGD optimiser is used with learning rate 0.01, momentum 0.9 and weight decay 1e-4. The loss function was not mentioned in the paper, so here, the standard of crossentropy loss is used for the training. Finally a batch size of 24 is mentioned in the paper, and the training happens on 9 classes, 8  organs and the zero class. The training procedure has then been setup according to this as can be seen in `train.py`.

### Training difficulties
When initially trying to train the model, our hardware seemed to struggle with the dataset. Out of memory erros were occuring when using the batch size of 24 stated in the paper. In order to train the model, the batch size had to be reduced to 10 on the 2080 super max-Q GPU that was avilable for training. Training using the full training dataset of 2211 samples could then be performed in around 2 hours on 30 epochs.

Due to time limits, not a lot of in depth development could be done on reproducing the excact training procedure of the paper and dubugging the outcome of the trained model. The focus of this reproduction was put on understanding the architectures of the three main parts of this network, UNet, ResNet and ViT.

### Training results
Due to time limits, the valudation run could not be performed using the validation samples, so no validation loss can be given. To judge the outcome of the model, the segmentation on a set of the training images can be analysed and compared to the labels.

Here, the loss progression of the training procudure can be seen over the 30 epochs. As can be seen, the loss does decrease and indicates a working optimization implementation. However, this can not be directly compared to the results of the paper as there they do not give the training loss curves but give more elaborate metrics using the Dice score coefficient and Hausdorff Distance. These metrics could not be implemented in the reproduction as it requires more focus on the training implementation.



To look at the segmentation, the softmax and argmax of the 9 channel outcome is used, as commonly done with segmentation outputs (see `test.py`). This should give the same dimension as the labels and can be plotted for comparison. Below, 4 samples can be seen, with the input image on the left, the label on the right, and the reproduced network outcome in the middle. As can be seen from these images, the classification has failed for these images. The network seems to mereg all the segmented areas in one class. The reason for this could be a mistake in the network architecture, like the segmentation head, were the 9 classes are extracted. Or also the training precure could be elaborated upan and dubugged more to find the cause of the segmentation errors. However it can also be seen that the network does somewhat succesfully identify the location of the organs in the scans, aside from some missing organs in the segmentation. The contours match somewhat closely with the organ labels, indicating an error in mainly the classification, but also still in the segmentaion due to some missing organs in the output.

![](/figs/Figure_1.png)

![](/figs/Figure_2.png)

![](/figs/Figure_3.png)

![](/figs/Figure_4.png)

## Bibliography

[^transunet] Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo,
Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, and Yuyin Zhou (2021). TransUNet: Transformers Make Strong
Encoders for Medical Image Segmentation. 	arXiv:2102.04306

[^unet] Ronneberger, O., Fischer, P., Brox, T. (2015) U-Net: Convolutional Networks for Biomedical
Image Segmentation. arXiv:1505.04597

[^unetcode] https://github.com/milesial/Pytorch-UNet

[^synapse] https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

[^pytorch_implementation_transformers] https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py 

