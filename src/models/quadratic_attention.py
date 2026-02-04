import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class TransformerModel(nn.Module):
    """
    Quadratic attention transformer model using GPT2 backbone.
    
    This is the exact Garg et al. transformer implementation with quadratic attention.
    """
    def __init__(self, n_dims, n_positions, n_embd=256, n_layer=6, n_head=4):
        super().__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleave x and y:
        [x0, y0, x1, y1, ..., x_{P-1}, y_{P-1}]
        with y embedded in first dim and zeros elsewhere.
        """
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            dim=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)  # (B, P, 2, D)
        zs = zs.view(bsize, 2 * points, dim)        # (B, 2P, D)
        return zs

    def forward(self, xs, ys, inds=None):
        """
        Forward pass for in-context learning.
        
        Args:
            xs: (B, P, D) - input features
            ys: (B, P) - target values
            inds: indices of positions to return predictions for (over points 0..P-1)
            
        Returns:
            (B, len(inds)) predictions
        """
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.tensor(inds, device=ys.device)
            if inds.max() >= ys.shape[1] or inds.min() < 0:
                raise ValueError("inds out of range")

        zs = self._combine(xs, ys)                 # (B, 2P, D)
        embeds = self._read_in(zs)                 # (B, 2P, n_embd)
        h = self._backbone(inputs_embeds=embeds).last_hidden_state
        logits = self._read_out(h)                 # (B, 2P, 1)

        # Read predictions at x positions: indices 0, 2, 4, ...
        preds_all = logits[:, ::2, 0]              # (B, P)
        return preds_all[:, inds]                  # (B, len(inds))
