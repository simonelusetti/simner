import torch
import torch.nn.functional as F

def nt_xent_loss(orig, pert, mask_orig, mask_pert, temperature=0.07):
    """
    orig: (B, T₁, D)
    pert: (B, T₂, D)
    mask_orig: (B, T₁)
    mask_pert: (B, T₂)
    """
    B = orig.shape[0]
    all_losses = []

    for i in range(B):
        # Get valid tokens per sentence
        valid_orig = mask_orig[i].bool()   # (T₁,)
        valid_pert = mask_pert[i].bool()   # (T₂,)

        # Get number of tokens both have (truncate longer one)
        len_common = min(valid_orig.sum(), valid_pert.sum())

        if len_common < 2:
            continue  # skip sentence if not enough aligned tokens

        orig_tokens = orig[i][valid_orig][:len_common]  # (N, D)
        pert_tokens = pert[i][valid_pert][:len_common]  # (N, D)

        # Normalize
        orig_tokens = F.normalize(orig_tokens, dim=1)
        pert_tokens = F.normalize(pert_tokens, dim=1)

        # NT-Xent loss between aligned tokens
        logits = torch.matmul(orig_tokens, pert_tokens.T) / temperature
        labels = torch.arange(len_common, device=orig.device)
        loss = F.cross_entropy(logits, labels)
        all_losses.append(loss)

    if not all_losses:
        return torch.tensor(0.0, requires_grad=True, device=orig.device)

    return torch.stack(all_losses).mean()
