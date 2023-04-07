import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler # <-- added Sampler
from typing import Sized, Iterator # <-- added

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        x_train, y_train = dataloader_utils.flatten(dl_train)
        n_classes = len(set(y_train))
        # ========================

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            neighbors_ids = torch.topk(dist_matrix[:,i], self.k, largest=False, sorted=False)[1]
            neighbors_lbls = self.y_train[neighbors_ids]
            y_pred[i] = int(torch.mode(neighbors_lbls)[0])
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.
    #  - Don't use torch.cdist

    dists = None
    # ====== YOUR CODE: ======
    N1, D = x1.shape
    N2 = x2.shape[0]
    dists = ((x1**2).sum(dim=1).view((N1,1)) - 2*(x1@x2.T) + (x2**2).sum(dim=1)).sqrt()
    
    # ========================

    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    return torch.where(y==y_pred, 1, 0).double().mean()
    # ========================

    return accuracy
# ==============MY_CODE===============
class k_cv_sampler(Sampler):
    
    def __init__(self, data_source: Sized, num_folds, val_fold_iteration, validation_bool):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_folds = num_folds
        self.val_fold_iteration = val_fold_iteration
        N = len(data_source)
        i = val_fold_iteration
        self.val_ids = list(range(i*N//num_folds, (i+1)*N//num_folds))
        self.train_ids = list(set(range(N)) - set(self.val_ids))
        self.validation_bool = validation_bool

    def __iter__(self) -> Iterator[int]:
        if self.validation_bool:
            for j in range(len(self.val_ids)):
                yield self.val_ids[j]
        else:
            for i in range(len(self.train_ids)):
                yield self.train_ids[i]


    def __len__(self):
        return len(self.data_source)

# ====================================

def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []
    N = len(ds_train)

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        choice_accuracies = []
        
        for cv_k_iteration in range(num_folds):
                              
            sampler_train = k_cv_sampler(ds_train, num_folds, val_fold_iteration=cv_k_iteration, validation_bool=False)
            dl_train_folds = torch.utils.data.DataLoader(ds_train, sampler = sampler_train)
            
            sampler_valid = k_cv_sampler(ds_train, num_folds, val_fold_iteration=cv_k_iteration, validation_bool=True)
            dl_valid_fold = torch.utils.data.DataLoader(ds_train, sampler = sampler_valid)
            x_test, y_test = dataloader_utils.flatten(dl_valid_fold)
            
            model.train(dl_train_folds)
            y_pred = model.predict(x_test)

            iteration_acc = accuracy(y_test, y_pred)
            choice_accuracies.append(iteration_acc)
        
        accuracies.append(choice_accuracies)
        
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
