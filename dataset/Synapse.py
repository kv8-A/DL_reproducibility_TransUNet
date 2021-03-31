import torch.utils.data as data
from torchvision import transforms
import glob
import numpy as np

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
            transforms.RandomHorizontalFlip()
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

        # Load the image at the filepath
        if self.mode.lower() == 'train':
            data = np.load(self.data_list[i])
            img = data['image']
            label = data['label']
        elif self.mode.lower() == 'test':
            # Load the image at the filepath
            img = ...
            label = ...
        # Apply transformations
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data_list)

