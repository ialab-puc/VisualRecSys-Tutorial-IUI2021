"""Profile mode Dataset (PyTorch) object

This module contains Dataset object with the triples information 
represented as (ui, pi, ni), where each is an identifier.
To this triplet, we append the item image: (ui, pi, ni, ii)
"""
import errno
import os

import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset


class UserModeImgDataset(Dataset):
    """Represents the Dataset as a PyTorch Dataset that yields tuples
    of 4 items: (ui, pi, ni, ii). This mode, represents users as an id.

    Attributes:
        ui, pi, ni, ii: Dataset tuples (in different arrays).
        transform: Transforms for each sample.
    """

    def __init__(self, csv_file, img_path, id2index, index2fn, transform=None):
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

        # TODO: this crashes, why?
        # Process profile elements
        # if id2index:
        #     # Note: Assumes id is str and index is int
        #     def map_id2index(element):
        #         if type(element) is list:
        #             return [id2index[e] for e in element]
        #         else:
        #             return id2index[str(element)]
        #     triples[["pi", "ni"]] = triples[["pi", "ni"]].applymap(map_id2index)
        
        # Keep important attributes
        self.ui = triples["ui"].to_numpy(copy=True)
        self.pi = triples["pi"].to_numpy(copy=True)
        self.ni = triples["ni"].to_numpy(copy=True)
        # Common setup
        self.transform = transform

    def __len__(self):
        return len(self.ui)

    def __getitem__(self, idx):
        path = os.path.join(self.__images_path, self.index2fn[idx])
        img = io.imread(path)
        return (
            self.ui[idx],
            self.pi[idx],
            self.ni[idx],
            img
        )
