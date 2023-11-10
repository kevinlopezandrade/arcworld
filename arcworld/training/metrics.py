import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


class ArcPercentageOfPerfectlySolvedTasks(Metric):
    """
    Computes the number of perfectly solved tasks.
    Predictions should be passed as unnormalized logits, since
    we compute the class using argmax over the probability distribution of the
    pixel, similar to how CrossEntropyLoss from pytorch also does.
    """

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "n_perfectly_solved", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor):
        """
        Args:
            preds: Shape [B, C, H, W]. Unnormalized logits.
            target: Shape [B, H, W]. Non binary target grid.
        """
        assert preds.shape[0] == target.shape[0]

        preds = torch.argmax(F.softmax(preds, dim=1), dim=1)

        for pred, grid in zip(preds, target):
            difference = pred - grid
            if difference.count_nonzero() == 0:
                self.n_perfectly_solved += 1

        self.total += preds.shape[0]

    def compute(self):
        return (self.n_perfectly_solved.float() / self.total) * 100


class ArcPixelDifference(Metric):
    """
    Computes the pixel difference of the predictions with respect
    to the output. Predictions should be passed as unnormalized logits, since
    we compute the class using argmax over the probability distribution of the
    pixel, similar to how CrossEntropyLoss from pytorch also does.
    """

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "pixel_difference", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor):
        """
        Args:
            preds: Shape [B, C, H, W]. Unnormalized logits.
            target: Shape [B, H, W]. Non binary target grid.
        """

        assert preds.shape[0] == target.shape[0]

        preds = torch.argmax(F.softmax(preds, dim=1), dim=1)

        for pred, grid in zip(preds, target):
            difference = pred - grid
            self.pixel_difference += difference.count_nonzero()

        self.total += preds.shape[0]

    def compute(self):
        return self.pixel_difference.float() / self.total
