import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import random

# === Load frozen BERT model once ===
frozen_model_name = "bert-base-cased"
frozen_tokenizer = AutoTokenizer.from_pretrained(frozen_model_name, use_fast=True)
frozen_model = AutoModel.from_pretrained(
    frozen_model_name,
    output_hidden_states=False,
    output_attentions=True,  # Only return attentions
    attn_implementation="eager"
)
frozen_model.eval()
for param in frozen_model.parameters():
    param.requires_grad = False

# === Prepare clean vocabulary (exclude special tokens) ===
vocab_list = [
    tok for tok in frozen_tokenizer.get_vocab().keys()
    if tok not in frozen_tokenizer.all_special_tokens
]
vocab_tensor = torch.tensor(
    [frozen_tokenizer.convert_tokens_to_ids(tok) for tok in vocab_list]
)

def score(input_ids, attention_mask, epsilon=1e-6):
    """
    For each token, computes a score indicating how likely it is to be an entity:
    score = ||post - embed|| * (incoming / outgoing)
    """
    input_ids = input_ids.clone().detach()
    attention_mask = attention_mask.clone().detach()

    with torch.no_grad():
        # Get initial embeddings
        inputs_embeds = frozen_model.embeddings(input_ids)  # [batch, seq, hidden]

        # Get final hidden states and attentions
        outputs = frozen_model(input_ids=input_ids, attention_mask=attention_mask)
        final_hidden = outputs.last_hidden_state            # [batch, seq, hidden]
        attentions = torch.stack(outputs.attentions)        # [layers, batch, heads, seq, seq]

    # Token distortion (embedding shift)
    delta = (final_hidden - inputs_embeds).norm(dim=-1)     # [batch, seq]

    # Mean attention across layers and heads
    attn = attentions.mean(dim=0).mean(dim=1)               # [batch, seq, seq]
    incoming = attn.sum(dim=1)                              # [batch, seq]
    outgoing = attn.sum(dim=2)                              # [batch, seq]

    # Avoid division by zero
    ratio = (incoming + epsilon) / (outgoing + epsilon)     # [batch, seq]

    # Final score
    score = delta * ratio                                   # [batch, seq]
    score = score * attention_mask                          # mask padding

    return score  # Higher = more likely entity

def perturb_sentence(text, temperature=1.0):
    # === Tokenize input ===
    encoding = frozen_tokenizer(text, return_tensors="pt", return_attention_mask=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    seq_len = input_ids.size(1)

    # === Compute token scores ===
    with torch.no_grad():
        token_scores = score(input_ids, attention_mask)[0]  # shape: [seq_len]

    # === Softmax over *inverted* scores to get replacement probabilities ===
    replace_probs = F.softmax(-token_scores / temperature, dim=0)

    # === Build a set of tokens to replace ===
    replace_mask = torch.bernoulli(replace_probs).bool()

    # === Replace selected tokens with random valid tokens (not special) ===
    perturbed_ids = input_ids.clone()
    for idx in range(seq_len):
        if replace_mask[idx]:
            rand_id = vocab_tensor[random.randint(0, len(vocab_tensor) - 1)]
            perturbed_ids[0, idx] = rand_id

    # === Decode perturbed sentence ===
    return frozen_tokenizer.decode(perturbed_ids[0], skip_special_tokens=True)

