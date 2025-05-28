import subprocess
import sys
from itertools import product

# Parse --clear if present
clear_flag = "--clear" in sys.argv

# Define the parameter grid
grid = {
    "config": ["training_set/medium"],
    "config.model_type": ["base", "attention", "transformer"],
    "config.epochs": [10]
}

# Generate all combinations of parameters
keys, values = zip(*grid.items())
experiments = [dict(zip(keys, v)) for v in product(*values)]

# Run each experiment
for i, params in enumerate(experiments):
    args = [f"{k}={v}" for k, v in params.items()]
    cmd = ["dora", "run"] + (["--clear"] if clear_flag else []) + args
    print(f"\n▶️ Running experiment {i + 1}/{len(experiments)}: {' '.join(cmd)}")
    subprocess.run(cmd)
