# simner/config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "bert-base-cased"
    epochs: int = 3
    batch_size: int = 8
    lr: float = 3e-5
    device: str = "cuda"
    split: str = "train[:10%]"
    attention: bool = False
    output_name: str = "simner_model"
    index_size: int = 1000
