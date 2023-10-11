import os
from typing import List

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from torch import FloatTensor, nn
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from arcworld.internal.constants import Example, Task
from arcworld.training.dataloader import TransformerOriginalDataset, decode_colors
from arcworld.training.metrics import ArcPixelDifference
from arcworld.training.pixeltransformer import PixelTransformer
from arcworld.utils import plot_grids, plot_task

wandb.login()


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
    model: nn.Module, metrics: List[Metric], dataloader: DataLoader, device: Device
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
            wandb.log({metric.__class__.__name__: metric.compute()}, commit=False)
            metric.reset()

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
            {"prediction": plot_grids(out_test, pred, return_fig=True)}, commit=False
        )


def train(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: Device,
):
    """
    Performs one epoch, and backpropagates at each processed batch.

    Args:
        model: Pytorch module to train.
        optimizer: Optimizer used for updating the weights.
        loss_fn: Loss function to minimize.
        dataloader: DataLoader from which to get the batches.
        device: Device used to store the tensors.
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
        wandb.log({loss_fn.__class__.__name__: epoch_loss}, commit=False)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    wandb.init(
        entity=cfg.user,
        project=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        notes=cfg.wandb_notes,
        name=cfg.wand_name,
    )

    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    train_dataset = TransformerOriginalDataset(
        cfg.dataset.train_path, h_bound=cfg.dataset.h_bound, w_bound=cfg.dataset.w_bound
    )
    eval_dataset = TransformerOriginalDataset(
        cfg.dataset.eval_path, h_bound=cfg.dataset.h_bound, w_bound=cfg.dataset.w_bound
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=0,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=0,
    )

    model = PixelTransformer(
        h=cfg.dataset.h_bound, w=cfg.dataset.w_bound, pos_encoding=cfg.pos_encoding
    ).to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1], device=device)
    )
    metrics = [
        Accuracy(task="multiclass", num_classes=11).to(device),
        ArcPixelDifference().to(device),
    ]
    for epoch in tqdm(range(cfg.epochs), desc="Training"):
        train(model, optimizer, loss_fn, train_dataloader, device)
        evaluate(model, metrics, eval_dataloader, device)

        if epoch % 5 == 0:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            checkpoint_path = os.path.join(
                hydra_cfg.runtime.output_dir, f"model_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

        # Commit all accumlated plots for this step.
        wandb.log({})


if __name__ == "__main__":
    main()
