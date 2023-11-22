import functools
import os
import shutil
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import torch.distributed as dist
from hydra import version
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf

_UNSPECIFIED_: Any = object()


def main_torch_distributed(
    config_path: Optional[str] = _UNSPECIFIED_,
    config_name: Optional[str] = None,
) -> Callable[[TaskFunction], Any]:
    """
    This is a decorator based on the @hydra.main decorator, but it syncs the output
    directory first between the processes. It is only meant to use inside this project.

    Args:
        config_path: The config path, a directory where Hydra will search for
            config files. This path is added to Hydra's searchpath. Relative paths
            are interpreted relative to the declaring python file. Alternatively,
            you can use the prefix `pkg://` to specify a python package to add to
            the searchpath. If config_path is None no directory is added to the
            Config search path.
        config_name: The name of the config (usually the file name without
            the .yaml extension)
    """
    version.setbase(None)

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            args_parser = get_args_parser()
            args = args_parser.parse_args()

            cli_args = OmegaConf.from_dotlist(args.overrides)
            rank = OmegaConf.select(cli_args, "rank")
            world_size = OmegaConf.select(cli_args, "world_size")

            if rank is None or world_size is None:
                raise ValueError(
                    "In distributed mode rank and world_size" " are required"
                )

            checkpoint = OmegaConf.select(cli_args, "checkpoint")

            # Rewrite for hydra not complaining.
            for i, arg in enumerate(args.overrides):
                if arg.startswith("rank="):
                    args.overrides[i] = f"+{arg}"
                if arg.startswith("world_size="):
                    args.overrides[i] = f"+{arg}"
                if arg.startswith("checkpoint="):
                    args.overrides[i] = f"+{arg}"
                if arg.startswith("dataset="):
                    args.overrides[i] = f"+{arg}"

            # Start the TCPStore server and distribute the output directory.
            if rank == 0:
                store = dist.TCPStore(
                    host_name="127.0.0.1",
                    port=8888,
                    world_size=world_size,
                    is_master=True,
                    wait_for_workers=True,
                    timeout=timedelta(seconds=30),
                )
            else:
                store = dist.TCPStore(
                    host_name="127.0.0.1",
                    port=8888,
                    world_size=world_size,
                    is_master=False,
                    timeout=timedelta(seconds=30),
                )

            if rank == 0:
                now = datetime.now()
                output_dir = (
                    "./outputs_distributed/"
                    + f"{now.strftime('%Y-%m-%d')}/"
                    + f"{now.strftime('%H-%M-%S')}"
                )
                store.set("output_dir", output_dir)

                if checkpoint:
                    prev_config_path = os.path.join(checkpoint, "rank_0", ".hydra")
                    prev_config_name = "config.yaml"
                    prev_dict = OmegaConf.load(
                        os.path.join(prev_config_path, prev_config_name)
                    )

                    # Clone the conf directory into the sub_conf inside the
                    # checkpointed directory.
                    sub_conf_path = os.path.join(prev_config_path, "sub_conf")
                    os.makedirs(sub_conf_path, exist_ok=True)

                    assert config_path is not None
                    shutil.copytree(config_path, sub_conf_path, dirs_exist_ok=True)

                    sub_conf_name = "config.yaml"
                    with open(os.path.join(sub_conf_path, sub_conf_name), "w") as f:
                        OmegaConf.save(config=prev_dict, f=f)

                    store.set("sub_conf_path", sub_conf_path)
                    store.set("sub_conf_name", sub_conf_name)
            else:
                store.wait(["output_dir"])
                output_dir = store.get("output_dir").decode()

                if checkpoint:
                    store.wait(["sub_conf_path"])
                    sub_conf_path = store.get("sub_conf_path").decode()
                    store.wait(["sub_conf_name"])
                    sub_conf_name = store.get("sub_conf_name").decode()

            args.overrides = args.overrides + [
                f"hydra.run.dir={output_dir}/rank_{rank}"
            ]

            if checkpoint:
                print("Running from the previous checkpoint")
                # To avoid hydra complaining
                for i, arg in enumerate(args.overrides):
                    if arg.startswith("+rank="):
                        args.overrides[i] = f"+{arg}"
                    if arg.startswith("+world_size="):
                        args.overrides[i] = f"+{arg}"

                # In case I'm starting from another checkpoint
                prev_dict = OmegaConf.load(os.path.join(sub_conf_path, sub_conf_name))
                if OmegaConf.select(prev_dict, "checkpoint"):
                    for i, arg in enumerate(args.overrides):
                        if arg.startswith("+checkpoint="):
                            args.overrides[i] = f"+{arg}"

                _run_hydra(
                    args=args,
                    args_parser=args_parser,
                    task_function=task_function,
                    config_path=sub_conf_path,
                    config_name=sub_conf_name,
                )

            else:
                _run_hydra(
                    args=args,
                    args_parser=args_parser,
                    task_function=task_function,
                    config_path=config_path,
                    config_name=config_name,
                )

        return decorated_main

    return main_decorator


def find_last_model(checkpoint_path: str) -> str:
    """
    In the directory where the weights have been saved per epoch returns the
    path to the model which was saved at the maximum epoch.
    """
    master_path = os.path.join(checkpoint_path, "rank_0")
    models = [file for file in os.listdir(master_path) if file.startswith("model")]
    models = [model.split(".")[0].split("_")[-1] for model in models]

    return os.path.join(checkpoint_path, "rank_0", f"model_epoch_{max(models)}.pt")
