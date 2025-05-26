import torch
import torch.nn as nn
from transformers import AutoModel

class SimNER(nn.Module):
    def __init__(self, model_name="bert-base-cased", projection_dim=256):
        super(SimNER, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        # Get token-level embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, T, H)

        # Project each token embedding
        projected_tokens = self.projection(token_embeddings)  # (B, T, D)
        return projected_tokens

class SimNERA(nn.Module):
    def __init__(self, model_name="bert-base-cased", projection_dim=256, num_heads = 4):
        super(SimNER, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # === Attention Layer ===
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size // num_heads, num_heads=num_heads, batch_first=True)

        # === Projection Head ===
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        # Step 1: encode with transformer
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, T, H)

        # Step 2: apply self-attention
        attn_output, _ = self.attn(token_embeddings, token_embeddings, token_embeddings, key_padding_mask=~attention_mask.bool())
        # attn_output: (B, T, H)

        # Step 3: project each token
        projected = self.projection(attn_output)  # (B, T, D)
        return projected
