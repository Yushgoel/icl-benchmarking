import torch
import torch.nn as nn
import torch.nn.functional as F


def relu_squared_feature_map(x):
    """Feature map φ(x) = ReLU(x)^2 + epsilon to prevent zero-division."""
    return F.relu(x) ** 2 + 1e-6


class CausalLinearAttention(nn.Module):
    """Causal linear attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, feature_map=relu_squared_feature_map, eps=1e-6):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feature_map = feature_map
        self.eps = eps

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, E = x.shape
        H = self.num_heads
        Dh = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t):
            return t.view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scale = Dh ** 0.5
        q = q / scale
        k = k / scale

        q_prime = self.feature_map(q)
        k_prime = self.feature_map(k)

        # Causal Linear Attention (Katharopoulos et al.)
        # Numerator: Sum(q * k^T * v)
        kv = k_prime.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, T, Dh, Dh)
        S = kv.cumsum(dim=2)                          # Prefix sum (Causal)
        numerators = torch.einsum("bhtd,bhtde->bhte", q_prime, S)

        # Denominator: Sum(q * k^T)
        z = k_prime.cumsum(dim=2)
        denom = torch.einsum("bhtd,bhtd->bht", q_prime, z).unsqueeze(-1)
        denom = denom.clamp(min=self.eps)

        out = numerators / denom
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.out_proj(out)


class LinearTransformerBlock(nn.Module):
    """Transformer block with linear attention."""
    
    def __init__(self, n_embd, n_head, mlp_ratio=4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalLinearAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)

        hidden_dim = int(mlp_ratio * n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LinearAttentionICLModel(nn.Module):
    """In-context learning model with linear attention."""
    
    def __init__(self, n_dims=5, n_positions=10, n_embd=256, n_layer=8, n_head=4):
        super().__init__()
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self.pos_emb = nn.Embedding(2 * n_positions, n_embd)
        self.layers = nn.ModuleList([LinearTransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self._read_out = nn.Linear(n_embd, 1)

    def _combine(self, xs, ys):
        """Interleave: x0, y0, x1, y1..."""
        B, P, D = xs.shape
        ys_wide = torch.cat((ys.view(B, P, 1), torch.zeros(B, P, D-1, device=ys.device)), dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(B, 2*P, D)
        return zs

    def forward(self, xs, ys):
        zs = self._combine(xs, ys)
        h = self._read_in(zs)

        B, T, _ = h.shape
        pos_ids = torch.arange(T, device=h.device).unsqueeze(0)
        h = h + self.pos_emb(pos_ids)

        for layer in self.layers:
            h = layer(h)

        # Output predictions
        pred_tokens = self._read_out(h)
        # We want predictions at x indices (0, 2, 4...)
        return pred_tokens[:, ::2, 0]
