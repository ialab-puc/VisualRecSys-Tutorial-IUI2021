import errno
import os

import numpy as np
import pandas as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset


class UserProfileModeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        items_padded = True

        # Data sources
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), csv_file,
            )

        self.__source_file = csv_file

        # Load triples from dataframe
        triples = pd.read_csv(self.__source_file)

        # Keep important attributes
        self.ui = triples["ui"].to_numpy(copy=True)
        self.pi = triples["pi"].to_numpy(copy=True)
        self.ni = triples["ni"].to_numpy(copy=True)
        if items_padded:
            self.pi = self.pi + 1
            self.ni = self.ni + 1

        self.profile = self.create_profiles(triples["profile"], remove_items=self.pi, padded=items_padded)

        self.users = np.unique(self.ui)
        self.items = np.unique(self.pi)

        # Common setup
        self.transform = transform

    def create_profiles(self, profile_column, remove_items=None, padded=True):
        profiles = profile_column.to_list()
        profiles = [[int(item) for item in profile.split()] for profile in profiles]
        if padded:
            profiles = [[item + 1 for item in profile] for profile in profiles] # Profile ids start at 1 due to padding

        if remove_items is not None:
            for profile, item in zip(profiles, remove_items):
                profile.remove(item)

        profiles = [torch.tensor(profile) for profile in profiles]

        return profiles

    def __len__(self):
        return len(self.ui)

    def __getitem__(self, idx):
        return (
            self.ui[idx],
            self.profile[idx],
            self.pi[idx],
            self.ni[idx],
        )
