from sklearn.neighbors import KNeighborsClassifier
from torch.nn.functional import normalize
import torch
from tqdm import tqdm

class KNNTokenClassifier:
    def __init__(self, simner_model, tokenizer, label_names, device="cpu", k=3):
        self.model = simner_model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.k = k
        self.label_names = label_names
        self.knn = None  # sklearn classifier

    def build_index(self, dataset):
        """
        dataset: Hugging Face Dataset with 'tokens' and 'ner_tags'
        """
        X = []
        y = []

        for example in tqdm(dataset, desc="building index"):
            tokens = example["tokens"]
            labels = example["ner_tags"]

            enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                emb = self.model(input_ids, attention_mask).squeeze(0)  # (T, D)
                emb = normalize(emb, dim=1)

            for i, (token_emb, attn, label_idx) in enumerate(zip(emb, attention_mask[0], labels)):
                if attn.item() == 1:
                    X.append(token_emb.cpu().numpy())
                    y.append(self.label_names[label_idx])

        self.knn = KNeighborsClassifier(n_neighbors=self.k, metric="cosine")
        self.knn.fit(X, y)

    def predict(self, tokens):
        enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            emb = self.model(input_ids, attention_mask).squeeze(0)  # (T, D)
            emb = normalize(emb, dim=1)

        preds = []
        for i, attn in enumerate(attention_mask[0]):
            if attn.item() == 1:
                vector = emb[i].cpu().numpy().reshape(1, -1)
                pred = self.knn.predict(vector)[0]
                preds.append(pred)
            else:
                preds.append("PAD")

        return preds
