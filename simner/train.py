# simner/train.py
import logging
from dora import hydra_main, get_xp
from .config import TrainConfig
from .trainer import train
from .eval import evaluate
import torch
import os
import json
from .models import SimNER, SimNERA, SimNERT

@hydra_main(config_name="config", config_path="../conf", version_base="1.1")
def main(cfg: TrainConfig):

    xp = get_xp()
    logger = logging.getLogger(__name__)
    logger.info(xp.sig)

    if os.path.isfile("history.json") and not cfg.meta.retrain and not cfg.meta.evaluate:
        with open("history.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        for run in data:
            print(data)
    elif cfg.meta.evaluate:
        model_name = cfg.model.model_name

        if cfg.model.model_type=="attention": model = SimNERA(model_name=model_name)  
        elif cfg.model.model_type=="transformer": model = SimNERT(model_name=model_name)  
        elif cfg.model.model_type=="base": model = SimNER(model_name=model_name)
        else: raise RuntimeError(f"❓ Unknown model type: {cfg.config.model_type}")
        
        model.load_state_dict(torch.load("./simner.pt", weights_only=True))

        metrics, report = evaluate(
            model, 
            index_size=cfg.evaluate.index_size, 
            config_args=cfg,
            dataset_name=cfg.evaluate.dataset_name
        )

        logger.info(report)
        xp.link.push_metrics(metrics)

    else:
        device = cfg.train.device
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            assert torch.cuda.is_available(), "❌ CUDA requested but not available!"

        model = train(
            model_name=cfg.model.model_name,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            lr=cfg.train.lr,
            device=device,
            split=cfg.dataset.split,
            model_type=cfg.model.model_type
        )

        torch.save(model.state_dict(), xp.folder / "simner.pt")
        logger.info(f"✅ model saved to {xp.folder} / simner.pt")

        metrics, report = evaluate(
            model, 
            index_size=cfg.evaluate.index_size, 
            config_args=cfg,
            dataset_name=cfg.evaluate.dataset_name
        )

        logger.info(report)
        xp.link.push_metrics(metrics)

        with open("history.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        for run in data:
            print(run)
