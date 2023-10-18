from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from numpy.typing import NDArray
from torch import FloatTensor, Tensor, nn
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader
from torchmetrics import Metric

from arcworld.internal.constants import Example, Task
from arcworld.training.dataloader import decode_colors
from arcworld.utils import plot_grids, plot_task


def _task_from_sequence(seq: FloatTensor) -> Task:
    """
    Given a tensor with shape [6, C, H, W] where the input and output pairs
    are arranged contiguously, decodes the colors and constructs a Task.
    """
    seq_np: NDArray[np.float32] = seq.cpu().numpy()
    task: Task = []
    for i in range(seq_np.shape[0] // 2):
        inp = decode_colors(seq_np[2 * i])
        out = decode_colors(seq_np[2 * i + 1])
        task.append(Example(input=inp, output=out))

    return task


def evaluate(
    model: nn.Module,
    metrics: List[Metric],
    dataloader: DataLoader[Tuple[Tensor, Tensor, Tensor]],
    device: Device,
    rank: Optional[int] = None,
):
    """
    Evalutes the model over the passed metrics without computing gradients.

    Args:
        model: Pytorch module to evaluate.
        metrics: List of metrics for which to evaluate the model.
        dataloader: DataLoader from which to get the batches.
        device: Device used to store the tensors.
    """
    model.eval()
    with torch.no_grad():
        for seq, inp_test, out_test in dataloader:
            seq = seq.to(device)
            inp_test = inp_test.to(device)
            out_test = out_test.to(device)

            pred = model(seq, inp_test)

            for metric in metrics:
                metric(pred, out_test)

        for metric in metrics:
            res = metric.compute()
            if rank is None or rank == 0:
                wandb.log({metric.__class__.__name__: res}, commit=False)

        if rank is None or rank == 0:
            # Randomly plot an input, output pair.
            seq, inp_test, out_test = next(iter(dataloader))
            pred = model(seq.to(device), inp_test.to(device))[0]
            pred = torch.argmax(F.softmax(pred, dim=0), dim=0).cpu().numpy()

            task = _task_from_sequence(seq[0])

            inp_test = decode_colors(inp_test[0].cpu().numpy())
            out_test = out_test[0].cpu().numpy()
            task.append(Example(input=inp_test, output=out_test))

            wandb.log({"random_task": plot_task(task, return_fig=True)}, commit=False)
            wandb.log(
                {"prediction": plot_grids(out_test, pred, return_fig=True)},
                commit=False,
            )


def train(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    dataloader: DataLoader[Tuple[Tensor, Tensor, Tensor]],
    device: Device,
    rank: Optional[int] = None,
):
    """
    Performs one epoch, and backpropagates at each processed batch.

    Args:
        model: Pytorch module to train.
        optimizer: Optimizer used for updating the weights.
        loss_fn: Loss function to minimize.
        dataloader: DataLoader from which to get the batches.
        device: Device used to store the tensors.
        rank: If rank is different to None, then it means we are in
            a distributed setting.
    """
    epoch_loss = 0.0

    for (
        seq,
        inp_test,
        out_test,
    ) in dataloader:
        optimizer.zero_grad()

        seq = seq.to(device)
        inp_test = inp_test.to(device)
        out_test = out_test.to(device)

        output = model(seq, inp_test)
        loss = loss_fn(output, out_test)
        loss.backward()
        optimizer.step()

        epoch_loss += loss

    with torch.no_grad():
        epoch_loss = epoch_loss * (1 / len(dataloader))
        if rank is not None and rank >= 0:
            dist.reduce(tensor=epoch_loss, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            # NOTE: The loss computed is the average epoch_loss over the number
            # of workers. A more fine grained loss can be computed at the cost
            # of storing two more variables and using two more reduce
            # perations. The different is not significant though.
            epoch_loss = epoch_loss * 1 / dist.get_world_size()

        if rank is None or rank == 0:
            wandb.log({loss_fn.__class__.__name__: epoch_loss}, commit=False)
