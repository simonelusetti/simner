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
    def __init__(self, model_name="bert-base-cased", projection_dim=256, num_heads=4, num_layers=2):
        super(SimNERA, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # === Transformer Encoder Stack ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,  # default expansion
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Projection Head ===
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        # Step 1: extract contextual embeddings from base model
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, T, H)

        # Step 2: pass through Transformer
        key_padding_mask = ~attention_mask.bool()  # flip: 1 is valid, 0 is pad
        transformed = self.transformer(token_embeddings, src_key_padding_mask=key_padding_mask)  # (B, T, H)

        # Step 3: project to final embedding
        projected = self.projection(transformed)  # (B, T, D)
        return projected
