"""JuprisMini-NCM core.

Focused PyTorch implementation of the NCM multiscale residual-VQ mechanism:
- residual is quantized from coarse to fine;
- each scale is quantized at its own temporal resolution;
- quantized scale vectors are upsampled and accumulated;
- codebooks use EMA updates (as in the upstream NCM quantizer family).

This is a compact port of the relevant method, not a wholesale copy of the
upstream repository. Tool routing remains a separate supervised runtime head.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = (1, 2, 4, 8, 16, 32)  # coarse -> fine
MAX_LEN = LEVELS[-1]
NUM_CONCEPTS = 512
TOOL_THRESHOLD = 480
TOOL_CONCEPT_IDS = frozenset({480, 481, 482, 483, 484})


class MultiscaleResidualVectorQuantizer(nn.Module):
    """Multiscale residual VQ with EMA-maintained codebooks.

    Input/output tensors use (batch, sequence, hidden). At every level, the
    current residual is area-downsampled, quantized, linearly upsampled, and
    subtracted from the residual for the next level.
    """
    def __init__(
        self,
        num_embeddings: int = NUM_CONCEPTS,
        embedding_dim: int = 128,
        levels: Sequence[int] = LEVELS,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.levels = tuple(int(x) for x in levels)
        if not self.levels or any(x <= 0 for x in self.levels):
            raise ValueError("levels must contain positive sequence lengths")
        if tuple(sorted(self.levels)) != self.levels:
            raise ValueError("levels must be ordered coarse-to-fine")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)

        self.embeddings = nn.ModuleList([
            nn.Embedding(self.num_embeddings, self.embedding_dim) for _ in self.levels
        ])
        for i, emb in enumerate(self.embeddings):
            nn.init.normal_(emb.weight, mean=0.0, std=1.0 / (self.embedding_dim ** 0.5))
            emb.weight.requires_grad_(False)  # EMA, not optimizer gradients
            self.register_buffer(f"ema_cluster_size_{i}", torch.ones(self.num_embeddings))
            self.register_buffer(f"ema_weight_{i}", emb.weight.detach().clone())

    @staticmethod
    def _resize(x: torch.Tensor, length: int, mode: str) -> torch.Tensor:
        if x.shape[1] == length:
            return x
        return F.interpolate(x.transpose(1, 2), size=length, mode=mode).transpose(1, 2).contiguous()

    @staticmethod
    def _distances(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=1, keepdim=True)
        e2 = (codebook * codebook).sum(dim=1).unsqueeze(0)
        return x2 + e2 - 2.0 * (x @ codebook.t())

    @torch.no_grad()
    def _ema_update(self, level_idx: int, flat_x: torch.Tensor, indices: torch.Tensor) -> None:
        if flat_x.numel() == 0:
            return
        counts = torch.bincount(indices, minlength=self.num_embeddings).to(flat_x.dtype)
        sums = torch.zeros_like(self.embeddings[level_idx].weight)
        sums.index_add_(0, indices, flat_x)
        cluster = getattr(self, f"ema_cluster_size_{level_idx}")
        ema_weight = getattr(self, f"ema_weight_{level_idx}")
        cluster.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        ema_weight.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        total = cluster.sum()
        smoothed = (cluster + self.epsilon) / (total + self.num_embeddings * self.epsilon) * total.clamp_min(self.epsilon)
        new_weight = ema_weight / smoothed.unsqueeze(1).clamp_min(self.epsilon)
        self.embeddings[level_idx].weight.copy_(new_weight)

    def embed(self, level_idx: int, indices: torch.Tensor) -> torch.Tensor:
        return self.embeddings[level_idx](indices)

    def forward(self, z: torch.Tensor):
        if z.ndim != 3 or z.shape[-1] != self.embedding_dim:
            raise ValueError(f"expected (B,T,{self.embedding_dim}) input; got {tuple(z.shape)}")
        z = z.float()
        residual = z.detach().clone()
        z_hat = torch.zeros_like(residual)
        indices_by_level: List[torch.Tensor] = []
        losses = []
        full_len = z.shape[1]

        for i, level_len in enumerate(self.levels):
            # Upstream method downsamples the current residual at each scale.
            level_input = self._resize(residual, level_len, "area") if i < len(self.levels) - 1 else residual
            flat = level_input.reshape(-1, self.embedding_dim)
            codebook = self.embeddings[i].weight
            indices = self._distances(flat, codebook).argmin(dim=-1).view(z.shape[0], level_len)
            quantized = self.embeddings[i](indices)
            flat_indices = indices.reshape(-1)
            if self.training:
                self._ema_update(i, flat.detach(), flat_indices.detach())
                # Read refreshed EMA weights for the current assignment.
                quantized = self.embeddings[i](indices)

            commitment = F.mse_loss(level_input, quantized.detach())
            losses.append(self.commitment_cost * commitment)
            indices_by_level.append(indices)

            upsampled = self._resize(quantized, full_len, "linear")
            z_hat = z_hat + upsampled
            residual = z.detach() - z_hat.detach()

        # STE carries encoder gradients while codebook vectors are EMA-updated.
        z_q_ste = z + (z_hat.to(z.dtype) - z).detach()
        loss = torch.stack(losses).sum() if losses else z.new_zeros(())
        return z_q_ste, indices_by_level, loss


class MultiscaleResidualVQTokenizer(nn.Module):
    """Compact token encoder feeding the multiscale residual quantizer."""
    def __init__(self, vocab_size: int, hidden_dim: int = 128, levels: Sequence[int] = LEVELS):
        super().__init__()
        self.levels = tuple(levels)
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, self.levels[-1], hidden_dim) * 0.02)
        self.rvq = MultiscaleResidualVectorQuantizer(
            num_embeddings=NUM_CONCEPTS, embedding_dim=hidden_dim, levels=self.levels
        )
    @property
    def quantizers(self):
        """Compatibility accessor; avoids registering duplicate submodules."""
        return self.rvq.embeddings

    def encode(self, token_ids: torch.Tensor):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        max_len = self.levels[-1]
        if token_ids.shape[1] < max_len:
            token_ids = F.pad(token_ids, (0, max_len - token_ids.shape[1]), value=0)
        else:
            token_ids = token_ids[:, :max_len]
        mask = (token_ids != 0).unsqueeze(-1)
        e = self.token_embed(token_ids) + self.pos_embed[:, :max_len, :]
        e = e * mask.to(e.dtype)
        z_q, indices, loss = self.rvq(e)
        # The first-level summary is used by the separate supervised router.
        h0 = F.adaptive_avg_pool1d(e.transpose(1, 2), 1).squeeze(-1)
        # The last level is the original token sequence, used by the token head.
        return indices + [token_ids], loss, h0


class NextConceptGenerator(nn.Module):
    """Coarse-to-fine generator; positions within a level are predicted in parallel."""
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        levels: Sequence[int] = LEVELS,
        shared_quantizers=None,
        shared_token_embed: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.levels = tuple(levels)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        if shared_quantizers is None:
            self.quantizers = nn.ModuleList([
                nn.Embedding(NUM_CONCEPTS, hidden_dim) for _ in self.levels
            ])
        else:
            self.quantizers = shared_quantizers
        self.token_embed = shared_token_embed or nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.stage_lengths = self.levels + (self.levels[-1],)  # six concept scales + token emission
        self.level_embed = nn.Parameter(torch.randn(len(self.stage_lengths), hidden_dim) * 0.02)
        self.intra_pos = nn.ParameterList([
            nn.Parameter(torch.randn(1, n, hidden_dim) * 0.02) for n in self.stage_lengths
        ])
        self.mask_embed = nn.Parameter(torch.randn(len(self.stage_lengths), hidden_dim) * 0.02)
        self.start_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head_coarse = nn.ModuleList([
            nn.Linear(hidden_dim, NUM_CONCEPTS) for _ in self.levels
        ])
        self.head_token = nn.Linear(hidden_dim, vocab_size)

    def _embed_known(self, idx_list, upto: int, prefix_ctx: Optional[torch.Tensor] = None):
        parts = [prefix_ctx] if prefix_ctx is not None else []
        for j in range(upto):
            if j < len(self.levels):
                emb = self.quantizers[j](idx_list[j])
                emb = emb + self.level_embed[j] + self.intra_pos[j][:, :emb.shape[1], :]
            else:
                emb = self.token_embed(idx_list[j])
            parts.append(emb)
        return torch.cat(parts, dim=1) if parts else None

    def _level_logits(self, idx_list, i: int, prefix_ctx: Optional[torch.Tensor]):
        b = (prefix_ctx.shape[0] if prefix_ctx is not None else idx_list[0].shape[0] if idx_list else 1)
        n = self.levels[i] if i < len(self.levels) else self.levels[-1]
        ctx = self._embed_known(idx_list, i, prefix_ctx)
        query = self.mask_embed[i].view(1, 1, -1).expand(b, n, -1) + self.intra_pos[i]
        if ctx is None:
            seq = torch.cat([self.start_embed.expand(b, -1, -1), query], dim=1)
        else:
            seq = torch.cat([ctx, query], dim=1)
        h = self.transformer(seq)
        out = h[:, -n:, :]
        return self.head_coarse[i](out) if i < len(self.levels) else self.head_token(out)

    def forward(self, idx_list, prefix_ctx: Optional[torch.Tensor] = None):
        return [self._level_logits(idx_list, i, prefix_ctx) for i in range(len(self.levels) + 1)]

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k > 0 and top_k < logits.shape[-1]:
            values, _ = torch.topk(logits, top_k, dim=-1)
            cutoff = values[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        picks = torch.multinomial(flat, num_samples=1).reshape(probs.shape[:-1])
        return picks

    @torch.no_grad()
    def generate(self, prefix_ctx: Optional[torch.Tensor] = None, temperature: float = 0.2, top_k: int = 20):
        b = 1 if prefix_ctx is None else prefix_ctx.shape[0]
        idx_list = []
        for i in range(len(self.levels) + 1):
            logits = self._level_logits(idx_list, i, prefix_ctx)
            if i < len(self.levels):
                level_idx = self._sample(logits, temperature, top_k)
            else:
                # Avoid UNK output. PAD is allowed only after the first three positions.
                logits = logits.clone()
                if logits.shape[-1] > 1:
                    logits[:, :, 1] = float("-inf")
                if logits.shape[1] > 0:
                    logits[:, :min(3, logits.shape[1]), 0] = float("-inf")
                level_idx = self._sample(logits, temperature, top_k)
            idx_list.append(level_idx)
        return idx_list


class ToolRouter(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_concepts: int = NUM_CONCEPTS):
        super().__init__()
        self.head = nn.Linear(hidden_dim, num_concepts)

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        return self.head(h0)


class JuprisNCM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        self.tokenizer = MultiscaleResidualVQTokenizer(vocab_size, hidden_dim)
        self.generator = NextConceptGenerator(
            vocab_size, hidden_dim, num_layers, num_heads,
            shared_quantizers=self.tokenizer.quantizers,
            shared_token_embed=self.tokenizer.token_embed,
        )
        self.router = ToolRouter(hidden_dim)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = MAX_LEN

    def _build_prompt_ctx(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        short_ids = prompt_ids[:, :10]
        tok_ctx = self.generator.token_embed(short_ids) + self.tokenizer.pos_embed[:, :short_ids.shape[1], :]
        _, _, h0_prompt = self.tokenizer.encode(prompt_ids)
        return torch.cat([h0_prompt.unsqueeze(1), tok_ctx], dim=1)

    def forward(self, prompt_ids: torch.Tensor, reply_ids: Optional[torch.Tensor] = None):
        if reply_ids is None:
            reply_ids = prompt_ids
        _, _, h0_prompt = self.tokenizer.encode(prompt_ids)
        tool_logits = self.router(h0_prompt)
        prompt_ctx = self._build_prompt_ctx(prompt_ids)
        idx_list, loss_commit, _ = self.tokenizer.encode(reply_ids)
        logits_per_level = self.generator(idx_list, prefix_ctx=prompt_ctx)
        return logits_per_level, idx_list, loss_commit, tool_logits

    @torch.no_grad()
    def route(self, token_ids: torch.Tensor, tool_threshold: int = TOOL_THRESHOLD):
        _, _, h0 = self.tokenizer.encode(token_ids)
        concept_id = int(self.router(h0).argmax(dim=-1)[0].item())
        is_tool = concept_id in TOOL_CONCEPT_IDS and concept_id >= tool_threshold
        if is_tool:
            return {"type": "TOOL_CALL", "tool_id": concept_id - tool_threshold, "concept_id": concept_id}
        return {"type": "TEXT_STREAM", "concept_id": concept_id}

    @torch.no_grad()
    def chat(self, token_ids: torch.Tensor, temperature: float = 0.2, top_k: int = 20):
        if token_ids.shape[1] < self.max_len:
            token_ids = F.pad(token_ids, (0, self.max_len - token_ids.shape[1]), value=0)
        else:
            token_ids = token_ids[:, :self.max_len]
        prompt_ctx = self._build_prompt_ctx(token_ids)
        idx_list = self.generator.generate(prefix_ctx=prompt_ctx, temperature=temperature, top_k=top_k)
        return idx_list[-1], idx_list


def compute_ncm_loss(
    model: JuprisNCM,
    prompt_ids: torch.Tensor,
    concept_id_batch: torch.Tensor,
    pad_id: int = 0,
    router_class_weights: Optional[torch.Tensor] = None,
    reply_ids: Optional[torch.Tensor] = None,
):
    logits_per_level, idx_list, loss_commit, tool_logits = model(prompt_ids, reply_ids)
    level_losses = []
    for i, logits in enumerate(logits_per_level):
        target = idx_list[i]
        _, _, classes = logits.shape
        if i == len(logits_per_level) - 1:
            weights = torch.ones(classes, device=logits.device, dtype=logits.dtype)
            weights[pad_id] = 0.08
            loss_i = 3.0 * F.cross_entropy(logits.reshape(-1, classes), target.reshape(-1), weight=weights)
        else:
            loss_i = 0.5 * F.cross_entropy(logits.reshape(-1, classes), target.reshape(-1))
        level_losses.append(loss_i)
    loss_levels = torch.stack(level_losses).sum()
    loss_router = F.cross_entropy(
        tool_logits, concept_id_batch.clamp(0, NUM_CONCEPTS - 1), weight=router_class_weights
    )
    total = loss_levels + loss_commit.mean() + 0.5 * loss_router
    return total, {
        "loss_total": round(float(total.detach().item()), 4),
        "loss_levels": round(float(loss_levels.detach().item()), 4),
        "loss_commit": round(float(loss_commit.detach().mean().item()), 4),
        "loss_router": round(float(loss_router.detach().item()), 4),
    }
