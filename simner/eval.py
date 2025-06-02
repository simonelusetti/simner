import torch
import logging
from transformers import AutoTokenizer
from datasets import load_dataset
from .classifier import KNNTokenClassifier
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

def evaluate(
        model,
        dataset_name="conll2003",
        k=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        index_size=None,
        config_args=None):
    model.eval()

    # === Load tokenizer and model ===
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # === Load dataset ===
    if dataset_name == "conll2003":
        dataset = load_dataset("conll2003")
    elif dataset_name == "wikiann":
        dataset = load_dataset("wikiann", "en")
    elif dataset_name == "wnut_17":
        dataset = load_dataset("wnut_17")

    shuffled_train = dataset["train"].shuffle()
    shuffled_test = dataset["test"].shuffle()

    if index_size != None: 
        shuffled_train = shuffled_train.select(range(index_size))
        shuffled_test = shuffled_test.select(range(index_size))

    # === Build classifier on part of the train set ===
    knn = KNNTokenClassifier(model, tokenizer, ["O","B-ENT"], device=device, k=k)
    knn.build_index(shuffled_train)

    # === Evaluate on test set ===
    y_true = []
    y_pred = []

    for example in tqdm(shuffled_test, desc="evaluating"):
        words = example["tokens"]
        binary_labels = ["O" if label == 0 else "B-ENT" for label in example["ner_tags"]]

        # Tokenize and track word-token alignment
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=True
        )
        word_ids = encoding.word_ids()

        # Predict at token level
        token_preds = knn.predict(words)

        # Align: keep only the first token prediction for each word
        aligned_preds = []
        aligned_labels = []

        # Map from word_idx → list of token indices
        word_to_token_indices = defaultdict(list)
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                word_to_token_indices[word_idx].append(idx)

        for word_idx, token_indices in sorted(word_to_token_indices.items()):
            label = binary_labels[word_idx]
            token_predictions = [token_preds[i] for i in token_indices]

            # If any token is ENT, classify the whole word as ENT
            pred = "B-ENT" if any(p == "B-ENT" for p in token_predictions) else "O"

            aligned_preds.append(pred)
            aligned_labels.append(label)

        y_true.append(aligned_labels)
        y_pred.append(aligned_preds)

    # === Generate report content ===
    report_text = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_text.append(f"Run: {timestamp}\n")
    report_text.append(f"Config arguments: {vars(config_args)}\n")
    report_text.append(f"Evaluation on {dataset_name} (multi-class NER)\n")
    report_text.append(classification_report(y_true, y_pred))

    metrics_text = []
    metrics_text.append(f"Run: {timestamp}\n")
    metrics_text.append(f"\nPrecision: {precision_score(y_true, y_pred):.4f}")
    metrics_text.append(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    metrics_text.append(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

    report = "\n".join(report_text)
    metrics = "\n".join(metrics_text)

    return metrics, report
