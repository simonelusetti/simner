from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

class PerturbatedDataset(Dataset):
    def __init__(self, split="train"):
        # Step 1: Parse the split
        if "[" in split and ":" in split and "%" in split:
            base_split = split.split("[")[0]  # e.g., "train"
            slice_expr = split.split("[")[1].rstrip("]")  # e.g., ":10%"
        elif "[" in split and ":" in split:
            base_split = split.split("[")[0]  # e.g., "train"
            slice_expr = split.split("[")[1].rstrip("]")  # e.g., ":1000"
        else:
            base_split = split
            slice_expr = None

        # Step 2: Load sentence highlights and apply slicing
        sentence_dataset = load_dataset("cnn_dailymail", "3.0.0", split=base_split)
        single_sentences = [line for string in sentence_dataset["highlights"] for line in string.split("\n") if line.strip()]

        # Step 3: Load perturbated data from disk and slice accordingly
        full_perturbated = load_from_disk("perturbated")[base_split]
        if slice_expr:
            # Handle slice expression like ":10%"
            if slice_expr.startswith(":") and slice_expr.endswith("%"):
                percent = float(slice_expr[1:-1]) / 100
                count = int(len(full_perturbated) * percent)
                full_perturbated = full_perturbated.select(range(count))
                single_sentences = single_sentences[:count]
            elif slice_expr.startswith(":"):
                count = int(slice_expr[1:-1])
                full_perturbated = full_perturbated.select(range(count))
                single_sentences = single_sentences[:count]
            else:
                raise ValueError(f"Unsupported slice expression: {slice_expr}")
        
        self.perturbated_sentences = full_perturbated["text"]
        self.sentences = single_sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.perturbated_sentences[idx]
