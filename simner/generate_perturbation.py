from tqdm import tqdm
from datasets import load_dataset
from perturbation import perturb_sentence

for split in ["train", "validation", "test"]:
    sentences = [line for string in load_dataset("cnn_dailymail", "3.0.0", split=split)["highlights"] for line in string.split('\n')]

    with open(f"./data/{split}/perturbated_sentences.jsonl", "w") as f:
        for s in tqdm(sentences, desc="Generating perturbations"):
            f.write(perturb_sentence(s) + "\n")