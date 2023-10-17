import functools
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
    mandatory_cli_args = OmegaConf.create(
        {
            "rank": "???",
            "world_size": "???",
            "output_dir": "./outputs_distributed/${%Y-%m-%d}/${%H-%M-%S}",
        }
    )

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            args_parser = get_args_parser()
            args = args_parser.parse_args()

            cli_args = OmegaConf.from_dotlist(args.overrides)
            cli_args = OmegaConf.merge(mandatory_cli_args, cli_args)

            rank = OmegaConf.select(cli_args, "rank", throw_on_missing=True)
            world_size = OmegaConf.select(cli_args, "world_size", throw_on_missing=True)

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
            else:
                store.wait(["output_dir"])
                output_dir = store.get("output_dir").decode()

            args.overrides = args.overrides + [
                f"hydra.run.dir={output_dir}/rank_{rank}"
            ]

            _run_hydra(
                args=args,
                args_parser=args_parser,
                task_function=task_function,
                config_path=config_path,
                config_name=config_name,
            )

        return decorated_main

    return main_decorator
