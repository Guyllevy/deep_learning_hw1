import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.normal(mean = 0, std = weight_std, size = (n_features, n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        # S = XW
        class_scores = x @ self.weights
        # row i of S is the scores of x_i (row i of x)
        y_pred = torch.max(class_scores, dim=1).indices
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = (y == y_pred).float().mean()
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            
            valid_loss_accumulate = 0
            valid_acc_accumulate = 0
            counter = 0
            
            for (x_v, y_v) in dl_valid:
                yp_v, cs_v = self.predict(x_v)
                valid_loss_accumulate  += loss_fn(x_v, y_v, cs_v, yp_v)
                valid_acc_accumulate += self.evaluate_accuracy(y_v, yp_v).item()
                counter += 1
                
            valid_res.accuracy.append(valid_acc_accumulate / counter )
            valid_res.loss.append(valid_loss_accumulate / counter )
            
            train_loss_accumulate = 0
            train_acc_accumulate = 0
            counter = 0

            for (x_t, y_t) in dl_train:
                yp_t, cs_t = self.predict(x_t)
                train_loss_accumulate  += loss_fn(x_t, y_t, cs_t, yp_t, self.weights, weight_decay)
                train_acc_accumulate += self.evaluate_accuracy(y_t, yp_t)
                self.weights -= loss_fn.grad() * learn_rate
                counter += 1
                
            train_res.accuracy.append(train_acc_accumulate / counter )
            train_res.loss.append(train_loss_accumulate / counter )

            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        # weights shape is (features, classes)
        # each column j of weights represents a picture to dot with an example to get the example score on class j
        # process:
        # (features, classes) --1--> (classes, features) --2--> (classes,channel,d1,d2)
        W = torch.clone(self.weights)
        if has_bias:
            W = W[:-1,:]
        W = W.T
        n_classes = W.shape[0]
        w_images = W.view((n_classes, *img_shape))
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.015
    hp['learn_rate'] = 0.009
    hp['weight_decay'] = 0.0015
    # ========================

    return hp
