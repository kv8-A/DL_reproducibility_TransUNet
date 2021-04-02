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
The first part of the TransUNet encoder is a CNN, in this case it is ResNet-50 [^transunet]. The paper mentiones that ResNet-50 is pretrained on ImageNet. Conveniently, PyTroch contains a pretrained ResNet50 on ImageNet, which will be used here. This CNN also needs to provide the skip connections to the decoder. Fig1 of the paper shows 3 blocks in the CNN encoder part which represent 3 high level layers of the ResNet from which the skip connections are output.
The paper does not go in a lot of detail about the ResNet implementation, so the only information we have to determine which layers of ResNet these are is that every skip connection has a certain resolution scale. ResNet uses stride=2 convolution layers to downsample, so looking at the structure of the PyTroch ResNet the correct high level layers should be derived.

First of all, only the feature extractors of ResNet are needed, so the last 2 layers which act as the classifiers, `AdaptiveAvgPool2d` and `Linear`, can be discarded. Next, the suitable layers of ResNet need to be split up in blocks so the skip output can be passed to the decoder later. When looking at the layers, there are more downsampling steps in ResNet-50 than are needed for the skip connections, 5 to be excact: `Conv2d(s=2)`, `MaxPool2d(s=2)`, `Conv2d(s=2)`, `Conv2d(s=2)` and `Conv2d(s=2)` in that order. Assumptions need to be made on how to split up the layers. Here is is assumed that the entire ResNet-50 layer count needs to be preserved, so that results in 4 blocks in total. The first 3 blocks contain one downsampling step each and are the blocks presented in Fig1 of the paper. The last block contains the remaining two downsampling steps and will be included in the CNN, but will not provide skip connections. The blocks are splitted right before the downsamplign step of the next block.

With the CNN blocks defined, also the input channels for the cascaded decoder can be determined by looking at the output channel count of the three first blocks. This results in the following skip channels: 64 - 256 - 512

```python
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

## Final Reproduced TransUNet

```python
```

## The Dataset
The dataset the paper uses for the Table1 experiments is the Synapse multi-organ segmentation dataset [^synapse]. This dataset consist of 30 abdominal CT scans, every scan consisting of a number of (85 ~ 198) axial contrast-enhanced abdominal clinical CT images [^transunet]. This results in a total of 3779 images with segmentation labels. The labels span across 8 different abdominal organs (aorta, gallbladder, spleen, left kidney, right kidney, liver, pancreas, spleen, stomach).

The dataset is randomly split into 18 scans (2211 slices) for training and 12 scans (1568 slices) for validation. The raw dataset is available from Synapse directly [^synapse], however this paper performed preprocessing on the data which makes it workable in pytorch. The preprocedd data was provided by the authors and consists of `.npz` files for the training, being one singel channel image and label per file. The validation data are `.npy.h5` files being a batch of singel channel images and labels per file (one scan).

The dataset is implemented in PyTorch and can read the numpy compatible files from a specified directory. Note that the paper also specifies that random transformations are performed on the data, these are random roration and random flipping, the latter is assumed to be both horizontal and vertical flipping. These transformations are also implemented in the dataset.

```python
class Synapse(data.Dataset):
    """Dataset loader.
    - root_dir (``string``): Data root directory path.
    - mode (``string``): train or test
    """

    def __init__(self, data_dir, mode):
        # Data root dir
        self.data_dir = data_dir
        # Mode
        self.mode = mode
        # Transforms
        """ Random roration and flipping as mentioned in [paper 4.2] """
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)) # also resize to 224x224
        ])
        # Get the data filepaths
        if self.mode.lower() == 'train':
            self.data_list = [f for f in glob.glob(data_dir+'/*.npz')]
        elif self.mode.lower() == 'test':
            self.data_list = [f for f in glob.glob(data_dir+'/*.npy.h5')]

    def __getitem__(self, i):
        """
        - index (``int``): index of the item in the dataset
        """

        if self.mode.lower() == 'train':
            # Load the images at the filepath
            data = np.load(self.data_list[i])
            img = data['image']
            label = data['label']
            # Apply transforms
            rng_state = torch.get_rng_state()
            img = self.transforms(img)
            torch.set_rng_state(rng_state)
            label = self.transforms(label)
            
        elif self.mode.lower() == 'test':
            # Load the images at the filepath
            data = h5py.File(self.data_list[i], 'r')
            img = data['image'][:]
            label = data['label'][:]
            # Apply transforms on every slice
            rng_state = torch.get_rng_state()
            img = torch.cat(
                [self.transforms(img_slice) for img_slice in img]
            ).unsqueeze(1) # unsqueeze to add the 1 channel
            torch.set_rng_state(rng_state)
            label = torch.cat(
                [self.transforms(label_slice) for label_slice in label]
            ).unsqueeze(1) # unsqueeze to add the 1 channel

        return img, label
        
    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data_list)
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