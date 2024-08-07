import os
from typing import List

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from arcworld.training.dataloader import ARC_TENSOR, ARCDataset
from arcworld.training.metrics import (
    ArcPercentageOfPerfectlySolvedTasks,
    ArcPixelDifference,
)
from arcworld.training.sampler import ARCBatchSampler
from arcworld.training.trainer import evaluate, train


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Save output dir in wandb.
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    wandb.init(
        entity=cfg.user,
        project=cfg.project,
        name=cfg.get("wandb_run_name", None),
        notes=cfg.get("wandb_notes", None),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    train_dataset = ARCDataset(
        cfg.dataset.train_path,
        h_bound=cfg.dataset.h_bound,
        w_bound=cfg.dataset.w_bound,
        max_input_output_pairs=cfg.dataset.max_input_output_pairs,
    )

    batch_sampler = ARCBatchSampler(
        train_dataset, max_batch_size=cfg.bs, shuffle=True, drop_last=False
    )
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)

    # NOTE: Datasets are still small in scale, so we construct all of them
    # and store them in memory. For bigger loads we need to rethink this.
    eval_dataloaders: List[DataLoader[ARC_TENSOR]] = []
    for path in cfg.dataset.eval_paths:
        eval_dataset = ARCDataset(
            path,
            h_bound=cfg.dataset.h_bound,
            w_bound=cfg.dataset.w_bound,
            max_input_output_pairs=cfg.dataset.max_input_output_pairs,
        )
        batch_sampler = ARCBatchSampler(
            eval_dataset, max_batch_size=cfg.bs, shuffle=True, drop_last=False
        )
        eval_dataloader = DataLoader(eval_dataset, batch_sampler=batch_sampler)
        eval_dataloaders.append(eval_dataloader)

    partial_model = instantiate(cfg.model)
    model = partial_model(
        h=cfg.dataset.h_bound,
        w=cfg.dataset.w_bound,
        max_input_output_pairs=cfg.dataset.max_input_output_pairs,
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
        ArcPercentageOfPerfectlySolvedTasks().to(device),
    ]
    for epoch in tqdm(range(cfg.epochs), desc="Training"):
        train(model, optimizer, loss_fn, train_dataloader, device)
        for eval_dataloader in eval_dataloaders:
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
