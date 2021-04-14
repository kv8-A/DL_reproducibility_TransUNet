import torch.utils.data as data
from torchvision import transforms
import torch
import glob
import numpy as np
import h5py
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import random

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
            # self.data_list = self.data_list[0:10] # for debugging
        elif self.mode.lower() == 'test':
            self.data_list = [f for f in glob.glob(data_dir+'/*.npy.h5')]
            # self.data_list = self.data_list[4:5] # for debugging

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

        return {'image': img, 'label': label}
        
    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data_list)

