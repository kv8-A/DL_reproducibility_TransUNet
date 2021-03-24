import torch.utils.data as data

# TODO This is just exemplary skeleton of the custom Dataset class. Read up the documentation on that
# TODO and tailor it to your own needs.


class YourDataset(data.Dataset):
    """Dataset loader.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that takes in
    an PIL image and returns a transformed version of it. Default: None.
    """

    def __init__(self,
                 root_dir,
                 mode='train',
                 transforms=None,
                 config=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        self.train_size = config['train_size']
        self.val_size = config['val_size']
        self.test_size = config['test_size']

        if self.mode.lower() == 'train':
            # Get the training data FILEPATHS (NOT THE IMAGES -> whole dataset might be too big to fit into memory)
            self.train_data = ...

            # Labels you can probably load all at once
            self.train_labels = ...
        elif self.mode.lower() == 'val':
            # Get the validation data FILEPATHS (NOT THE IMAGES -> whole dataset might be too big to fit into memory)
            self.val_data = ...

            # Labels you can probably load all at once
            self.val_labels = ...

        elif self.mode.lower() == 'test':
            # Get the test data FILEPATHS (NOT THE IMAGES -> whole dataset might be too big to fit into memory)
            self.test_data = ...

            # Labels you can probably load all at once
            self.test_labels = ...
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val")

    def __getitem__(self, i):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """

        if self.mode.lower() == 'train':
            # Load the image at the filepath
            img = ...
            label = ...
        elif self.mode.lower() == 'val':
            # Load the image at the filepath
            img = ...
            label = ...

        elif self.mode.lower() == 'test':
            # Load the image at the filepath
            img = ...
            label = ...
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        # Apply transformations if necessary
        if self.transforms is not None:
            img = ...

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return ...
        elif self.mode.lower() == 'val':
            return ...
        elif self.mode.lower() == 'test':
            return ...
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

