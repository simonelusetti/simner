import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from models import SimNER
from classifier import KNNTokenClassifier
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from tqdm import tqdm
from datetime import datetime
import os
import argparse

def evaluate(model, k=3, device="cuda" if torch.cuda.is_available() else "cpu", index_size=1000, config_args = None):
    # === Load tokenizer and model ===
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # === Load WikiANN English ===
    dataset = load_dataset("conll2003")

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
    report_text.append(f"Config arguments: {vars(config_args)}\n")
    report_text.append("Evaluation on WikiANN (binary: ENT vs O)\n")
    report_text.append(classification_report(y_true, y_pred))
    report_text.append(f"\nPrecision: {precision_score(y_true, y_pred):.4f}")
    report_text.append(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    report_text.append(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

    report = "\n".join(report_text)

    # === Print to console ===
    print("\n" + report)

    # === Save to file ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/eval_report_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {filename}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a string and a number.")
    parser.add_argument("--model_name", type=str, help="the model to load")
    parser.add_argument("--index_size", type=int, nargs="?", default=1000, help="size of the index for evaluation")

    args = parser.parse_args()

    index_size = args.index_size
    model_name = args.model_name

    model = SimNER(model_name="bert-base-cased", projection_dim=256)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(f"./models/{model_name}.pt", map_location=device))
    evaluate(model,index_size=index_size, config_args = args)
