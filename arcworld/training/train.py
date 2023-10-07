import os
from typing import List

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import Accuracy
from torchmetrics.text import EditDistance
from tqdm import tqdm

from arcworld.internal.constants import Example, Task
from arcworld.training.dataloader import TransformerOriginalDataset, decode_colors
from arcworld.training.pixeltransformer import PixelTransformer
from arcworld.utils import plot_grids, plot_task

wandb.login()


def evaluate(
    model: nn.Module, metrics: List[Metric], dataloader: DataLoader, device: Device
):
    """
    Evalutes the model without computing gradients.
    """
    model.eval()
    with torch.no_grad():
        for seq, inp_test, out_test in dataloader:
            seq = seq.to(device)
            inp_test = inp_test.to(device)
            out_test = out_test.to(device)

            pred = model(seq, inp_test)

            for metric in metrics:
                if isinstance(metric, EditDistance):
                    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
                    out_str: List[str] = []
                    pred_str: List[str] = []

                    for grid in out_test:
                        out_str.append(str(grid.cpu().numpy().tolist()))

                    for grid in pred:
                        pred_str.append(str(grid.cpu().numpy().tolist()))

                    metric(pred_str, out_str)

                else:
                    metric(pred, out_test)

        for metric in metrics:
            wandb.log({metric.__class__.__name__: metric.compute()}, commit=False)
            metric.reset()

        # Randomly plot an input, output pair.
        seq, inp_test, out_test = next(iter(dataloader))
        pred = model(seq.to(device), inp_test.to(device))[0]
        pred = torch.argmax(F.softmax(pred, dim=0), dim=0).cpu().numpy()
        seq = seq.cpu().numpy()[0]

        task: Task = []
        for i in range(seq.shape[0] // 2):
            inp = decode_colors(seq[2 * i])
            out = decode_colors(seq[2 * i + 1])
            task.append(Example(input=inp, output=out))

        inp_test = decode_colors(inp_test[0].cpu().numpy())
        out_test = out_test[0].cpu().numpy()
        task.append(Example(input=inp_test, output=out_test))

        wandb.log({"random_task": plot_task(task, return_fig=True)})
        wandb.log({"prediction": plot_grids(out_test, pred, return_fig=True)})


def train(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: Device,
):
    """
    Performs one epoch, and backpropagates at each processed batch.
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
    )

    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    train_dataset = TransformerOriginalDataset(cfg.dataset.train_path)
    eval_dataset = TransformerOriginalDataset(cfg.dataset.eval_path)

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

    model = PixelTransformer().to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1], device=device)
    )
    metrics = [Accuracy(task="multiclass", num_classes=11).to(device), EditDistance()]
    for epoch in tqdm(range(cfg.epochs), desc="Training"):
        train(model, optimizer, loss_fn, train_dataloader, device)
        evaluate(model, metrics, eval_dataloader, device)

        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(
                    hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                    f"model_epoch_{epoch}.pt",
                ),
            )

        wandb.log({})


if __name__ == "__main__":
    main()
