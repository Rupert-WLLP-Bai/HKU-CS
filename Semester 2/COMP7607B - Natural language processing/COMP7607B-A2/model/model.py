import math
from .config import LMConfig
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input tensor x.

        Ref: [RMSNorm](https://arxiv.org/abs/1910.07467)

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as x
        """
        # Write your code here
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x / math.sqrt(x.size(-1))
        return x / (rms_x + self.eps) * self.weight


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Implement the forward pass of the attention layer.

        Ref: [Attention is all you need](https://arxiv.org/abs/1706.03762)
        Ref: [GQA](https://arxiv.org/abs/2305.13245)

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)
            pos_cis: Positional Coding tensor of shape (sequence_length, hidden_dim)
            past_key_value: Optional tuple of tensors containing past keys and values
            use_cache: Whether to use cached past keys and values

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: Output tensor and past key values
        """

        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # Implement kv_cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # Implement attention
        # Write your code here
        attn_weights = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + self.mask[:, :, :seq_len, :seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, xv).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * (
                (hidden_dim + config.multiple_of - 1) // config.multiple_of
            )
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList(
            [MiniMindBlock(layer, params) for layer in range(self.n_layers)]
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(
                dim=params.dim // params.n_heads, theta=params.rope_theta
            ),
            persistent=False,
        )
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **args,
    ):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get("start_pos", 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.size(1)]
        past_kvs = []
        for layer, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis, past_key_value=past_key_values[layer], use_cache=use_cache
            )
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("past_key_values", past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        eos_token_id=2,
        max_new_tokens=1024,
        temperature=0.75,
        top_p=0.90,
        stream=False,
        rp=1.0,
        use_cache=True,
        pad_token_id=0,
        **args,
    ):
        # Stream generation
        if stream:
            return self._stream(
                input_ids,
                eos_token_id,
                max_new_tokens,
                temperature,
                top_p,
                rp,
                use_cache,
                **args,
            )

        # Non-stream generation
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(
                non_pad,
                eos_token_id,
                max_new_tokens,
                temperature,
                top_p,
                rp,
                use_cache,
                **args,
            )
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [
                    seq,
                    torch.full(
                        (1, max_length - seq.size(1)),
                        pad_token_id,
                        dtype=seq.dtype,
                        device=seq.device,
                    ),
                ],
                dim=-1,
            )
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(
        self,
        input_ids,
        eos_token_id,
        max_new_tokens,
        temperature,
        top_p,
        rp,
        use_cache,
        **args,
    ):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = (
                    self(
                        input_ids, past_key_values=past_kvs, use_cache=use_cache, **args
                    ),
                    False,
                )
            else:
                out = self(
                    input_ids[:, -1:],
                    past_key_values=past_kvs,
                    use_cache=use_cache,
                    start_pos=input_ids.shape[1] - 1,
                    **args,
                )
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= temperature + 1e-9
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                    :, :-1
                ].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
