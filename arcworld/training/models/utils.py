import os
from pathlib import Path
from typing import Any, Dict, Tuple, Type, TypeVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from arcworld.internal.constants import Task
from arcworld.storage.fingerprint import normalize_task
from arcworld.training.dataloader import encode_colors, encode_task
from arcworld.training.models.pixeltransformer import PixelTransformer
from arcworld.training.models.pixeltransformer_modified import PixelTransformerModified

ALLOWED_MODELS = {
    PixelTransformer.__name__: PixelTransformer,
    PixelTransformerModified.__name__: PixelTransformerModified,
}

_T = TypeVar("_T")


def _get_boundaries_from_model(
    model: PixelTransformer | PixelTransformerModified,
) -> Tuple[int, int]:
    """
    The following is a hack, a workaround since we did not
    store the height and width boundaries into the attributes
    of the model and therefore we have to get them from the
    shape.
    """
    inp_out_channel = model.inp_out_channel
    h = inp_out_channel.shape[3]
    w = inp_out_channel.shape[4]

    return h, w


def load_model(weights_path: str, cls: Type[_T], **params: str) -> _T:
    """
    Given the weights of a model and its paremeters, it creates an instance of
    that model using those weights.

    Args:
        weights_path: Path where the weights are located.
        cls: Class of the model to instantiate.
        params: The keword arguments needed to instantiate
            the model.

    Returns:
        The instantiated model that lives in the cpu.
    """
    model = cls(**params)

    state = torch.load(weights_path, map_location="cpu")
    model_state = state["model_state_dict"]
    model.load_state_dict(model_state)

    return model


def load_model_from_dir(weights_path: str) -> nn.Module:
    """
    A wrapper around load_model, that assumes that
    the weights_path are contained within an output
    directory generated with hydra.

    Args:
        weights_path: Path that shoud has as parent path
            and output directory generated by hydra.
    Returns:
        A instantiated model from a checkpoint, that lives
        in the cpu.
    """
    path = Path(weights_path)
    output_dir_path = path.parent.absolute().as_posix()
    config_path = OmegaConf.load(os.path.join(output_dir_path, ".hydra", "config.yaml"))
    cls_name = config_path.model._target_
    cls_name = cls_name.split(".")[-1]
    cls_ = ALLOWED_MODELS[cls_name]
    model_params = get_model_params_hydra(output_dir_path)

    return load_model(weights_path, cls_, **model_params)


def get_model_params_yaml(config_path: str):
    cfg = OmegaConf.load(config_path)

    model_params = OmegaConf.to_container(cfg.model)
    model_params = cast(Dict[str, Any], model_params)

    # Remove the params related to Hydra
    model_params.pop("_target_")
    model_params.pop("_partial_")

    # Required params from the dataset
    model_params["h"] = cfg.dataset.h_bound
    model_params["w"] = cfg.dataset.w_bound
    model_params["max_input_output_pairs"] = cfg.dataset.max_input_output_pairs

    return model_params


def get_model_params_hydra(output_path: str) -> Dict[str, Any]:
    """
    Given an output directory generated by Hydra it returns the model
    parameters of the model that produced that directory.

    Args:
        output_path: The path of the output directory generated by
            the training of a model.

    Returns:
        The model paremeters needed to instantiate a model of
        the same type.
    """
    cfg = os.path.join(output_path, ".hydra", "config.yaml")
    return get_model_params_yaml(cfg)


def get_model_params(weights_path: str):
    """
    Given the weights_path of a model which was produced and lives
    within an hydra output directory, it returns the paremeters
    that were used to instantiate that model.
    """
    path = Path(weights_path)
    output_dir_path = path.parent.absolute().as_posix()

    return get_model_params_hydra(output_dir_path)


def predict(
    model: nn.Module,
    task: Task,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a PixelTransformer model and task it outputs the prediction
    using the argmax over the probabilites of the pixels.

    Args:
        model: The model.
        task: The task for which to predict the output.

    Returns:
        A tuple where the first element is the expected output
        and the second element is the predicted output.
    """

    if not isinstance(model, tuple(ALLOWED_MODELS.values())):
        raise ValueError(
            f"{model.__class__.__name__} is not an allowed class to use ",
            "in this predict function",
        )

    # The first examples not considering the last one are the inputs.
    if len(task) > model.max_input_output_pairs:
        raise ValueError("The task contains more input examples than the model allows")

    if len(task) < model.max_input_output_pairs:
        raise ValueError("The task contains less input examples than the model expects")

    h_bound, w_bound = _get_boundaries_from_model(model)

    task_array = normalize_task(
        task,
        h=h_bound,
        w=w_bound,
        max_input_output_pairs=model.max_input_output_pairs,
    )

    # TODO: This is a HACK to get the device but it makes
    # a bit more clean for the end user. It assumes
    # that all the model paremeters live in the same
    # device which might not be true in casese
    # where model parallelism is used.
    device = next(model.parameters()).device

    # Prepare data with batch size 1.
    X = torch.Tensor(encode_task(task_array[:-1])).unsqueeze(0).to(device)  # noqa
    inp_test = (
        torch.Tensor(encode_colors(task_array[-1, 0, :, :])).unsqueeze(0).to(device)
    )
    out_test = torch.LongTensor(task_array[-1, 1, :, :]).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out_pred = model(X, inp_test)[0]

    out_pred = torch.argmax(F.softmax(out_pred, dim=0), dim=0)

    return out_test[0], out_pred