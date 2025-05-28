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

        # === Stack of Only Multi-Head Attention Layers ===
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # === Projection Head ===
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        # Step 1: get embeddings from base model
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, T, H)

        # Step 2: build attention mask (0 = keep, -inf = mask)
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        attn_mask = (1.0 - attn_mask) * -1e9  # (B, 1, 1, T) for additive attention bias

        # Step 3: pass through multi-head attention layers
        x = token_embeddings
        for attn, norm in zip(self.attention_layers, self.attn_norms):
            residual = x
            # Note: attention_mask must be bool with shape (B, T)
            attn_out, _ = attn(x, x, x, key_padding_mask=~attention_mask.bool())
            x = norm(residual + attn_out)  # Add & Norm

        # Step 4: projection
        projected = self.projection(x)
        return projected


class SimNERT(nn.Module):
    def __init__(self, model_name="bert-base-cased", projection_dim=256, num_heads=4, num_layers=2):
        super(SimNERT, self).__init__()
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
