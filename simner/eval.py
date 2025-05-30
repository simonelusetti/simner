import torch
import logging
from transformers import AutoTokenizer
from datasets import load_dataset
from .models import SimNER
from .classifier import KNNTokenClassifier
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from tqdm import tqdm
from datetime import datetime
import os
from dora import get_xp

logger = logging.getLogger(__name__)

def evaluate(
        model,
        dataset = "conll2003",
        k=3, 
        device="cuda" if torch.cuda.is_available() else "cpu", 
        index_size=1000, 
        config_args = None):
    # === Load tokenizer and model ===
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # === Load dataset ===
    if dataset == "conll2003":
        dataset = load_dataset("conll2003")
    if dataset == "wikiann":
        dataset = load_dataset("wikiann","en")
    if dataset == "wnut_17":
        dataset = load_dataset("wnut_17")

    shuffled_train = dataset["train"].shuffle()
    shuffled_test = dataset["test"].shuffle()

    label_names = shuffled_train.features["ner_tags"].feature.names

    # === Build classifier on part of the train set ===
    knn = KNNTokenClassifier(model, tokenizer, label_names, device=device, k=k)
    knn.build_index(shuffled_train.select(range(index_size)))

    # === Evaluate on test set ===
    y_true = []
    y_pred = []

    for example in tqdm(shuffled_test.select(range(index_size)), desc="evaluating"):
        tokens = example["tokens"]
        true_labels = ["O" if label_names[i] == "O" else "B-ENT" for i in example["ner_tags"]]
        pred_labels = knn.predict(tokens)
        pred_labels = ["O" if p == "O" else "B-ENT" for p in pred_labels]

        filtered_pairs = [(t, p) for t, p in zip(true_labels, pred_labels) if p != "PAD"]
        if not filtered_pairs:
            continue

        true_seq, pred_seq = zip(*filtered_pairs)
        y_true.append(list(true_seq))
        y_pred.append(list(pred_seq))

    # === Generate report content ===
    report_text = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_text.append(f"Run: {timestamp}")
    report_text.append(f"Config arguments: {vars(config_args)}\n")
    report_text.append(f"Evaluation on {dataset} (binary: ENT vs O)\n")
    report_text.append(classification_report(y_true, y_pred))
    report_text.append(f"\nPrecision: {precision_score(y_true, y_pred):.4f}")
    report_text.append(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    report_text.append(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

    report = "\n".join(report_text)

    return report