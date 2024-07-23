"""Logging of hyperparameters"""

from typing import Any, Dict

from lightning.pytorch.utilities import rank_zero_only
from loguru import logger as log
from torch.nn.parameter import UninitializedParameter


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters

    :param object_dict: Dictionary of objects to log
    :type object_dict: Dict[str, Any]
    """

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams = {
        "task": cfg["task"],
        "encoder": cfg["encoder"],
        "decoder": cfg["decoder"],
        "model/params/total": sum(
            [
                x.numel() if not isinstance(x, UninitializedParameter) else 0
                for x in model.parameters()
            ]
        ),
        # sum(
        #     (
        #         0 if isinstance(p, UninitializedParameter) else p.numel()
        #         for p in model.parameters()
        #     )
        # ),
    }
    hparams["model/params/trainable"] = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad and not isinstance(p, UninitializedParameter)
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel()
        for p in model.parameters()
        if not p.requires_grad and not isinstance(p, UninitializedParameter)
    )

    hparams["dataset"] = cfg["dataset"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
