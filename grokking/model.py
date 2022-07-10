from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, Int, jaxtyped
import torch
from torch import nn, Tensor

_typechecker = beartype if __debug__ else None


class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_model: int, num_heads: int) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, num_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model),
        )
        self.ffn_norm = nn.LayerNorm(dim_model)

    @jaxtyped(typechecker=_typechecker)
    def forward(self, x: Float[Tensor, "seq batch dim"]) -> Float[Tensor, "seq batch dim"]:
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        attn_out = self.self_attn_norm(x + attn_out)
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(attn_out + ffn_out)

        return ffn_out


class Transformer(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        num_tokens: int,
        seq_len: int,
    ) -> None:
        super().__init__()

        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.layers = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, num_tokens),
        )

    @jaxtyped(typechecker=_typechecker)
    def forward(self, inputs: Int[Tensor, "batch seq"]) -> Float[Tensor, "seq batch tokens"]:
        batch_size, seq_len = inputs.shape

        token_embedding = self.token_embeddings(inputs)

        positions = repeat(
            torch.arange(seq_len, device=inputs.device), "p -> b p", b=batch_size
        )
        position_embedding = self.position_embeddings(positions)

        embedding = rearrange(token_embedding + position_embedding, "b s d -> s b d")

        return self.layers(embedding)
