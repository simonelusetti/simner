# simner/train.py
from dora import hydra_main
from .config import TrainConfig
from .trainer import train
from .eval import evaluate
import torch
import os



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

    os.makedirs("models", exist_ok=True)
    output_path = f"models/{cfg.output_name}.pt"
    torch.save(model.state_dict(), output_path)

    evaluate(model, index_size=cfg.index_size, config_args=cfg)
