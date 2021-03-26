"""Profile mode Dataset (PyTorch) object

This module contains Dataset object with the triples information 
represented as (profile, pi, ni), where profile is a set of items
and pi and ni are identifier.
"""
import errno
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ProfileModeDataset(Dataset):
    """Represents the Dataset as a PyTorch Dataset that yields tuples
    of 3 items: (profile, pi, ni). This mode, represents users as a
    profile, a set of items.

    Attributes:
        profile_sizes: Size of each user profile.
        unique_profiles: Actual profile data to save space.
        profile, pi, ni: Dataset triples (in different arrays).
        transform: Transforms for each sample.
    """

    def __init__(self, csv_file, transform=None, id2index=None):
        """Inits a UGallery Dataset.

        Args:
            csv_file: Path (string) to the triplets file.
            transform: Optional. Torchvision like transforms.
            id2index: Optional. Transformation to apply on items.
        """
        # Data sources
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), csv_file,
            )
        self.__source_file = csv_file
        # Load triples from dataframe
        triples = pd.read_csv(self.__source_file)
        # Process profile elements
        if id2index:
            # Note: Assumes id is str and index is int
            def map_id2index(element):
                if type(element) is list:
                    return [id2index[e] for e in element]
                else:
                    return id2index[str(element)]
            triples["profile"] = triples["profile"].map(lambda p: p.split())
            triples = triples.applymap(map_id2index)
            triples["profile"] = triples["profiles"].map(lambda p: " ".join(p))
        # Mapping to unique profiles and use it to calculate profile sizes
        unique_profiles = triples["profile"].unique()
        profile2index = {k: v for v, k in enumerate(unique_profiles)}
        triples["profile"] = triples["profile"].map(profile2index)
        profile_sizes = np.fromiter(
            map(lambda p: p.count(" "), unique_profiles),
            dtype=int, count=len(unique_profiles),
        ) + 1
        profile_sizes = triples["profile"].map(dict(enumerate(profile_sizes)))
        self.unique_profiles = unique_profiles.astype(np.string_)
        self.profile_sizes = profile_sizes.to_numpy(copy=True)
        # Using numpy arrays for faster lookup
        self.profile = triples["profile"].to_numpy(copy=True)
        self.pi = triples["pi"].to_numpy(copy=True)
        self.ni = triples["ni"].to_numpy(copy=True)
        # Common setup
        self.transform = transform

    def __len__(self):
        return len(self.pi)

    def __getitem__(self, idx):
        prof = self.profile[idx]
        if isinstance(idx, int) or isinstance(idx, np.number):
            profile = np.fromstring(
                self.unique_profiles[prof], dtype=int, sep=" ",
            )
        else:
            profile = np.fromstring(
                b" ".join(self.unique_profiles[prof]), dtype=int, sep=" ",
            ).reshape((len(idx), -1))

        return (
            profile,
            self.pi[idx],
            self.ni[idx],
        )

