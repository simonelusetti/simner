from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from .loss import nt_xent_loss
from .dataset import PerturbatedDataset
from .models import SimNER, SimNERA, SimNERT
import os
import logging

def train(
        model_name = "bert-base-cased", 
        model_type = "base",
        epochs=3, 
        batch_size=8, 
        lr=3e-5, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        split="train[:100%]"
    ):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_type=="attention": model = SimNERA(model_name=model_name)  
    elif model_type=="transformer": model = SimNERT(model_name=model_name)  
    elif model_type=="base": model = SimNER(model_name=model_name)
    else: raise RuntimeError(f"❓ Unknown model type: {model_type}")

    dataset = PerturbatedDataset(split=split)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            originals, perturbed = batch

            enc_original = tokenizer(originals, return_tensors="pt", padding=True, truncation=True)
            enc_perturbed = tokenizer(perturbed, return_tensors="pt", padding=True, truncation=True)

            input_ids_orig = enc_original["input_ids"].to(device)
            attention_mask_orig = enc_original["attention_mask"].to(device)

            input_ids_pert = enc_perturbed["input_ids"].to(device)
            attention_mask_pert = enc_perturbed["attention_mask"].to(device)

            # Get token-level embeddings
            emb_orig = model(input_ids_orig, attention_mask_orig)      # (B, T, D)
            emb_pert = model(input_ids_pert, attention_mask_pert)      # (B, T, D)

            # Compute loss over aligned tokens (define this function yourself)
            loss = nt_xent_loss(emb_orig, emb_pert, attention_mask_orig, attention_mask_pert)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} | Avg Token-wise Loss: {avg_loss:.4f}")

    return model