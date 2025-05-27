# simner/train.py
import logging
from dora import hydra_main
from .config import TrainConfig
from .trainer import train
from .eval import evaluate
import torch
import os

logger = logging.getLogger(__name__)

@hydra_main(config_name="config", config_path="../conf", version_base="1.1")
def main(cfg: TrainConfig):
    device = cfg.config.device
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available!"

    model = train(
        model_name=cfg.config.model_name,
        epochs=cfg.config.epochs,
        batch_size=cfg.config.batch_size,
        lr=cfg.config.lr,
        device=device,
        split=cfg.config.split,
        attention=cfg.config.attention,
    )

    evaluate(model, index_size=cfg.config.index_size, config_args=cfg)
