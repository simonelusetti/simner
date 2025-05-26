import trainer
import os
import torch
from eval import evaluate

if __name__ == "__main__":
    output_dir = "./models"
    output_name = "simnera_15_perc"

    model = trainer.train(split="train[:15%]")

    output_path = f"{output_dir}/{output_name}.pt"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Model saved to: {output_path}")

    evaluate(model)