import os
from typing import List

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from arcworld.training.dataloader import ARC_TENSOR, TransformerOriginalDataset
from arcworld.training.metrics import (
    ArcPercentageOfPerfectlySolvedTasks,
    ArcPixelDifference,
)
from arcworld.training.trainer import evaluate, train
from arcworld.training.utils import main_torch_distributed


@main_torch_distributed(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    RANK = cfg.rank  # noqa
    WORLD_SIZE = cfg.world_size  # noqa

    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:8000",
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    # HACK: There is a bug in torch.distributed
    # where processes with RANK > 0 allocate
    # memory in cuda:0. Check issue #98763.
    # Recommended workaround is the following.
    torch.cuda.set_device(RANK)
    device = torch.device(f"cuda:{RANK}")

    # Save output dir in wandb.
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if RANK == 0:
        wandb.init(
            entity=cfg.user,
            project=cfg.project,
            name=cfg.get("wandb_run_name", None),
            notes=cfg.get("wandb_notes", None),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    print(f"PID: {os.getpid()}")
    print(OmegaConf.to_yaml(cfg))

    # Create the datasets.
    train_dataset = TransformerOriginalDataset(
        cfg.dataset.train_path,
        h_bound=cfg.dataset.h_bound,
        w_bound=cfg.dataset.w_bound,
        max_input_otput_pairs=cfg.dataset.max_input_otput_pairs,
    )

    # Create the Distributed Samplers
    train_sampler = DistributedSampler(
        dataset=train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True
    )

    # Create the DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.bs,
    )

    # NOTE: Datasets are still small in scale, so we construct all of them
    # and store them in memory. For bigger loads we need to rethink this.
    eval_dataloaders: List[DataLoader[ARC_TENSOR]] = []
    for path in cfg.dataset.eval_paths:
        eval_dataset = TransformerOriginalDataset(
            path,
            h_bound=cfg.dataset.h_bound,
            w_bound=cfg.dataset.w_bound,
            max_input_otput_pairs=cfg.dataset.max_input_otput_pairs,
        )
        eval_sampler = DistributedSampler(
            dataset=eval_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=cfg.bs,
        )

        eval_dataloaders.append(eval_dataloader)

    partial_model = instantiate(cfg.model)
    model = partial_model(
        h=cfg.dataset.h_bound,
        w=cfg.dataset.w_bound,
        max_input_otput_pairs=cfg.dataset.max_input_otput_pairs,
    ).to(device)
    model_ddp = DistributedDataParallel(model, device_ids=[RANK], output_device=RANK)
    model_ddp.train()

    params = [p for p in model_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1], device=device)
    )
    metrics = [
        Accuracy(task="multiclass", num_classes=11).to(device),
        ArcPixelDifference().to(device),
        ArcPercentageOfPerfectlySolvedTasks().to(device),
    ]

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        train_sampler.set_epoch(epoch)
        train(model_ddp, optimizer, loss_fn, train_dataloader, device, RANK)

        for eval_dataloader in eval_dataloaders:
            eval_dataloader.sampler.set_epoch(epoch)
            evaluate(model_ddp, metrics, eval_dataloader, device, RANK)

        if epoch % 5 == 0:
            if RANK == 0:
                hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                checkpoint_path = os.path.join(
                    hydra_cfg.runtime.output_dir, f"model_epoch_{epoch}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_ddp.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

        # Commit all accumlated plots for this step.
        if RANK == 0:
            wandb.log({})

        dist.barrier()


if __name__ == "__main__":
    main()
