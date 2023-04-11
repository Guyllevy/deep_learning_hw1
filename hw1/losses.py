import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted, weights = None, weight_decay = 0):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        M = (x_scores.T - x_scores[torch.arange(x_scores.shape[0]), y].T).T
        D = torch.zeros_like(M) + self.delta
        D[torch.arange(x_scores.shape[0]), y] = 0
        M += D
        M[M < 0] = 0
        M_sums = M.sum(dim =1)
        loss = M_sums.mean()
        if weights != None:
            loss += 0.5*weight_decay*torch.norm(weights)
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx = (M,x,y,weights,weight_decay)
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # M (B,C)
        # X (B,D)
        # (D,B) @ (B,C) = (D,C)
        # first assume B = 1
        M,x,y,weights,weight_decay = self.grad_ctx
        G = torch.zeros_like(M)
        # M in the i, y_i spot is 0 so leave G at 0 and add elements in those spots later
        G = (M > 0).float()
        G_new = G.clone()
        G_new[torch.arange(G_new.shape[0]), y] = -G.sum(dim=1)
        G_new *= 1/x.shape[0]
        grad = x.T @ G_new
        if weights != None:
            grad += weight_decay*weights
        
        # ========================

        return grad
