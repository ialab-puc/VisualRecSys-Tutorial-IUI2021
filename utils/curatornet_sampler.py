"""Custom batch sampler"""
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import BatchSampler, RandomSampler


class SameProfileSizeBatchSampler(BatchSampler):
    """Custom batch sampler that yields batches of triples with the
    same profile size (CuratorNet).

    Retrieves items from the sampler and yields a batch of size
    batch_size with items of the same size.

    Attributes:
        sampler: PyTorch sampler object to retrieve triples.
        batch_size: Max number of triples in each batch.
        profile_items_per_batch: Max number of items in profile.
        drop_last: Decides what to do with items that do not fill a
            batch.
        n_largest_first: How many of the largest batches to return
            first.
    """

    def __init__(self, sampler, batch_size=None, profile_items_per_batch=None,
                 drop_last=False, n_largest_first=0):
        # Data sources
        self.sampler = sampler
        assert hasattr(self.sampler.data_source, "profile_sizes")
        # Minibatch limits
        assert batch_size is not None or profile_items_per_batch is not None
        self.batch_size = batch_size if batch_size else float("inf")
        self.profile_items_per_batch = profile_items_per_batch if profile_items_per_batch else float("inf")
        # More setup
        self.drop_last = drop_last
        self.n_largest_first = n_largest_first
        # Settings under the hood
        self.__shuffle = isinstance(sampler, RandomSampler)
        self.__minibatches = None
        self.__samples_per_profile_size = None
        # Prepare sampler
        self.prepare()

    def prepare(self):
        # Group samples of the same size to avoid doing it while training
        self.__samples_per_profile_size = defaultdict(list)
        for idx in self.sampler:
            p_size = self.sampler.data_source.profile_sizes[idx]
            self.__samples_per_profile_size[p_size].append(idx)
        # Transform each list into numpy array
        self.__samples_per_profile_size = {
            k: np.array(v)
            for k, v in self.__samples_per_profile_size.items()
        }
        # Generate minibatches for the first time to fill attributes
        self.generate_minibatches()

    def generate_minibatches(self):
        minibatches = list()
        for p_size, samples in self.__samples_per_profile_size.items():
            # Shuffle samples if necessary
            if self.__shuffle:
                np.random.shuffle(samples)
            batch_size = min(
                self.batch_size,
                self.profile_items_per_batch // p_size,
            )
            # Reduce samples to chunks
            for i in range(0, len(samples), batch_size):
                minibatch = samples[i:i+batch_size]
                minibatches.append((
                    len(minibatch) * p_size,  # Items in profile
                    len(minibatch),  # Items in pi/ni
                    minibatch,  # Actual minibatch
                ))
            # Drop "irregular" batches
            if self.drop_last:
                minibatches.pop(-1)
        self.__minibatches = minibatches

    def __iter__(self):
        # Generate minibatches
        self.generate_minibatches()
        # Prepare largest items first
        if self.n_largest_first:
            self.__minibatches = sorted(
                self.__minibatches,
                key=lambda mb: (mb[0], mb[1]),
                reverse=True,
            )
        largest_minibatches = self.__minibatches[:self.n_largest_first]
        minibatches = self.__minibatches[self.n_largest_first:]
        # Prepare and shuffle other minibatches if necessary
        if self.__shuffle:
            random.shuffle(minibatches)
        # Join and yield minibatches
        self.__minibatches = largest_minibatches + minibatches
        for _, _, minibatch in self.__minibatches:
            yield minibatch

    def __len__(self):
        return len(self.__minibatches)
