import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        # TODO:
        # Implement the logic required for this sampler.
        # If the length of the data source is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        # ====== YOUR CODE: ======
        if len(self.data_source) >= 1:
            left = 0
            right = len(self.data_source)-1
            step = 0
            while left <= right:
                if step % 2 == 0:
                    yield left
                    left += 1
                    step += 1
                else:
                    yield right
                    right -= 1
                    step += 1
        # ========================

    def __len__(self):
        return len(self.data_source)
    
# ====== MY CODE: =========
class SplitSampler(Sampler):
    
    def __init__(self, data_source: Sized, indices):
        super().__init__(data_source)
        self.data_source = data_source
        self.indices = indices
        
    def __iter__(self) -> Iterator[int]:
        i = 0
        while i < self.indices:
            yield self.indices[i]
            i+=1

    def __len__(self):
        return len(self.data_source)
# ========================

def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======
    N = len(dataset)
    V = int(N * validation_ratio)

    valid_ids, train_ids = torch.split(torch.Tensor(list(range(N))),[V,N-V]) #split might want tensor?

    train_sampler = SplitSampler(dataset,train_ids)
    valid_sampler = SplitSampler(dataset,valid_ids)
    
    dl_train = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=train_sampler)
    
    dl_valid = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=valid_sampler)
    
    # ========================

    return dl_train, dl_valid



    