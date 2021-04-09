"""Profile mode Dataset (PyTorch) object

This module contains Dataset object with the triples information 
represented as (ui, pi, ni), where each is an identifier.
To this triplet, we append the item image: (ui, pi, ni, ii)
"""
import torch
import errno
import os

import numpy as np
import pandas as pd
from skimage import io, transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UserModeImgDataset(Dataset):
    """Represents the Dataset as a PyTorch Dataset that yields tuples
    of 5 items: (ui, pi, ni, pimg, nimg).
    This mode represents users as an id.

    Attributes:
        ui, pi, ni, pimg, nimg: Dataset tuples (in different arrays).
        transform: Transforms for each sample.
    """

    def __init__(self, csv_file, img_path, id2index, index2fn, transform=None, img_size=224):
        """Inits a Dataset.

        Args:
            csv_file: Path (string) to the triplets file.
            img_path: Path (string) to the images
            id2index: Dict. Keys are img name, values are indexes
            index2fn: Dict. Keys are indexes, values are file names
            transform: Optional. Torchvision like transforms.
            
        """
        # Data sources
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), csv_file,
            )
        if not os.path.isdir(img_path):
            raise NotADirectoryError(
                errno.ENOENT, os.strerror(errno.ENOENT), img_path
            )

        self.__source_file = csv_file
        self.__images_path = img_path
        self.id2index = id2index
        self.index2fn = index2fn
        
        # Load triples from dataframe
        triples = pd.read_csv(self.__source_file)

        # Keep important attributes
        self.ui = triples["ui"].to_numpy(copy=True)
        self.pi = triples["pi"].to_numpy(copy=True)
        self.ni = triples["ni"].to_numpy(copy=True)
        # Common setup
        if transform is None:
            self.transform = TransformTuple(img_size)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.ui)

    def __getitem__(self, idx):
        pimgpath = os.path.join(self.__images_path, self.index2fn[self.pi[idx]])
        pimg = Image.open(pimgpath)

        nimgpath = os.path.join(self.__images_path, self.index2fn[self.ni[idx]])
        nimg = Image.open(nimgpath)
        tuple = self.transform(self.ui[idx], pimg, nimg, self.pi[idx], self.ni[idx])

        if tuple[1] is not None and tuple[2] is not None:
           return tuple


class TransformTuple(object):
    def __init__(self, img_size):
        assert isinstance(img_size, (int, tuple))
        self.to_tensor = ToTensor()
    
    def __call__(self, ui, pimg, nimg, pi, ni):
        pimg = self.to_tensor(pimg)
        nimg = self.to_tensor(nimg)
        return (ui, pimg, nimg, pi, ni)



class Rescale(object):
    """Rescale the image in a sample to a given size.
    output_size (tuple or int): Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        img = transforms.ToTensor()(image)
        return img
