import errno
import os

import numpy as np
import pandas as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset


class UserModeFeatDataset(Dataset):
    def __init__(self, csv_file, feature_path, transform=None):
        # Data sources
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), csv_file,
            )
        if not os.path.isfile(feature_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), feature_path,
            )

        self.__source_file = csv_file

        # Load triples from dataframe
        triples = pd.read_csv(self.__source_file)

        # Keep important attributes
        self.ui = triples["ui"].to_numpy(copy=True)
        self.pi = triples["pi"].to_numpy(copy=True)
        self.ni = triples["ni"].to_numpy(copy=True)
        self.profile, self.pad_token = self.create_profiles(triples["profile"], remove_items=self.pi)

        self.feature_data, self.feature_default_idx = self.load_feature_data(feature_path)

        self.users = np.unique(self.ui)
        self.items = np.unique(self.pi) + 1 # Add pad_token
        self.feature_dim = self.feature_data.shape[-1]

        # Common setup
        self.transform = transform

    def load_feature_data(self, feature_path):
        with open(feature_path, 'rb') as fp:
            feature_data = np.load(fp, allow_pickle=True)
        feature_data = feature_data[:,1].tolist()
        feature_data = np.array(feature_data) # Faster when transformed to numpy first
        feature_data = torch.tensor(feature_data)
        feature_data = feature_data.permute((0,2,3,1)) # TODO: Hack: by default d should be last dimension
        feature_data, default_idx = self.append_default_features(feature_data)
        # feature_data = np.array(feature_data)
        # feature_data = feature_data.transpose((0,2,3,1)) # TODO: Hack: by default d should be last dimension
        # feature_data, default_idx = self.append_default_features(feature_data)
        return feature_data, default_idx

    def append_default_features(self, feature_data):
        # feature_dims = feature_data.shape[1:]
        # default_features = np.zeros((1, *feature_dims))
        # feature_data = np.append(default_features, feature_data, axis=0)
        # default_idx = 0
        feature_dims = feature_data.shape[1:]
        default_features = torch.zeros((1, *feature_dims))
        feature_data = torch.cat((default_features, feature_data), dim=0)
        default_idx = 0
        return feature_data, default_idx

    def create_profiles(self, profile_column, remove_items=None):
        profiles = profile_column.to_list()
        profiles = [[int(item) for item in profile.split()] for profile in profiles]
        if remove_items is not None:
            for profile, item in zip(profiles, remove_items):
                profile.remove(item)

        profiles = [torch.tensor(profile) + 1 for profile in profiles] # Leave pad_token = 0
        pad_token = 0
        return profiles, pad_token

    def __len__(self):
        return len(self.ui)

    def __getitem__(self, idx):
        return (
            self.ui[idx],
            self.profile[idx],
            self.pi[idx],
            self.ni[idx],
        )

    def get_features(self, ids):
        if isinstance(ids, int):
                ids = torch.tensor([ids])
        if isinstance(ids, list):
                ids = torch.tensor(ids)
#         ids[ids < 0] = self.feature_default_idx
        # return self.feature_data.take(ids, axis=0)
        return self.feature_data[ids]
