# simner/train.py
import logging
from dora import hydra_main, get_xp
from .config import TrainConfig
from .trainer import train
from .eval import evaluate
import torch
import os
import json

@hydra_main(config_name="config", config_path="../conf", version_base="1.1")
def main(cfg: TrainConfig):

    xp = get_xp()
    logger = logging.getLogger(__name__)
    logger.info(xp.sig)

    if os.path.isfile("history.json"):
        with open("history.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        for run in data:
            print(run)
    else:
        device = cfg.config.device
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            assert torch.cuda.is_available(), "❌ CUDA requested but not available!"

        model = train(
            model_name=cfg.config.model_name,
            epochs=cfg.config.epochs,
            batch_size=cfg.config.batch_size,
            lr=cfg.config.lr,
            device=device,
            split=cfg.config.split,
            model_type=cfg.config.model_type
        )

        report = evaluate(model, index_size=cfg.config.index_size, config_args=cfg)
        xp.link.push_metrics(report)
        with open("history.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        for run in data:
            print(run)
