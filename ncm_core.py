"""JuprisMini-NCM core (Base v0 Production Lock).

- 100% state_dict compatible (strict=True) with jupris_ncm_v9_stage3_frozen_v2.pt
  and jupris_ncm_v8_vqfix_v2.pt.
- Includes position-wise anti-repeat token decoding on the parallel token head
  so adjacent positions do not emit stuttering duplicates ('bosku bosku bosku').
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = (1, 2, 4, 8, 16, 32)  # coarse -> fine
MAX_LEN = LEVELS[-1]
NUM_CONCEPTS = 512
TOOL_THRESHOLD = 480

LANGUAGE_CONCEPT_IDS: Tuple[int, ...] = (120, 125, 150, 155, 210, 220)
TOOL_CONCEPT_IDS: frozenset[int] = frozenset({480, 481, 482, 483, 484})

EXPLICIT_TOOL_MAP: Dict[int, str] = {
    480: "search_web",
    481: "read_gpio",
    482: "run_python",
    483: "system_status",
    484: "repo_analyzer",
}

LANG_INDEX_MAP: Dict[int, int] = {cid: idx for idx, cid in enumerate(LANGUAGE_CONCEPT_IDS)}
TOOL_CLASS_CONCEPT_IDS: Tuple[int, ...] = (0, 480, 481, 482, 483, 484)
TOOL_INDEX_MAP: Dict[int, int] = {cid: idx for idx, cid in enumerate(TOOL_CLASS_CONCEPT_IDS)}


def _sample_tokens_no_repeat(
    token_logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    min_tokens: int = 3,
    repeat_penalty: float = 6.0,
    window: int = 4,
) -> torch.Tensor:
    """Decode parallel token logits left-to-right while suppressing immediate repeats."""
    b, seq_len, vocab_size = token_logits.shape
    logits = token_logits.clone()
    if vocab_size > 1:
        logits[:, :, 1] = float("-inf")  # Never emit UNK=1
    if seq_len > 0 and min_tokens > 0:
        logits[:, :min(min_tokens, seq_len), 0] = float("-inf")  # Delay PAD=0

    out = torch.zeros(b, seq_len, dtype=torch.long, device=logits.device)
    ended = torch.zeros(b, dtype=torch.bool, device=logits.device)

    for t in range(seq_len):
        step_logits = logits[:, t, :].clone()
        for r in range(b):
            if ended[r]:
                step_logits[r, :] = float("-inf")
                step_logits[r, 0] = 0.0
                continue
            start_t = max(0, t - window)
            for prev_t in range(start_t, t):
                prev_tok = int(out[r, prev_t].item())
                if prev_tok > 1:
                    if prev_t == t - 1:
                        step_logits[r, prev_tok] = float("-inf")
                    else:
                        step_logits[r, prev_tok] -= repeat_penalty
            if t >= 2:
                history = out[r, :t].tolist()
                for tok_id in set(history):
                    if tok_id > 1 and history.count(tok_id) >= 2:
                        step_logits[r, tok_id] = float("-inf")

        if temperature <= 0:
            picked = step_logits.argmax(dim=-1)
        else:
            scaled = step_logits / temperature
            if 0 < top_k < vocab_size:
                vals, _ = torch.topk(scaled, top_k, dim=-1)
                cutoff = vals[:, -1, None]
                scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
            probs = F.softmax(scaled, dim=-1)
            picked = torch.multinomial(probs, num_samples=1).squeeze(-1)

        out[:, t] = picked
        ended = ended | (picked == 0)

    return out


class MultiscaleResidualVectorQuantizer(nn.Module):
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
            emb.weight.requires_grad_(False)
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
        smoothed = (
            (cluster + self.epsilon)
            / (total + self.num_embeddings * self.epsilon)
            * total.clamp_min(self.epsilon)
        )
        new_weight = ema_weight / smoothed.unsqueeze(1).clamp_min(self.epsilon)
        self.embeddings[level_idx].weight.copy_(new_weight)

    def embed(self, level_idx: int, indices: torch.Tensor) -> torch.Tensor:
        return self.embeddings[level_idx](indices)

    @torch.no_grad()
    def decode_indices(
        self,
        indices_by_level: Sequence[torch.Tensor],
        output_length: Optional[int] = None,
    ) -> torch.Tensor:
        if len(indices_by_level) != len(self.levels):
            raise ValueError(f"expected {len(self.levels)} VQ levels, got {len(indices_by_level)}")
        batch_size = indices_by_level[0].shape[0]
        target_len = int(output_length or self.levels[-1])
        device = self.embeddings[0].weight.device
        dtype = self.embeddings[0].weight.dtype
        z_hat = torch.zeros(batch_size, target_len, self.embedding_dim, device=device, dtype=dtype)
        for level_idx, indices in enumerate(indices_by_level):
            indices = indices.to(device=device, dtype=torch.long)
            quantized = self.embeddings[level_idx](indices)
            upsampled = self._resize(quantized, target_len, "linear")
            z_hat = z_hat + upsampled
        return z_hat

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None, update_ema: bool = True):
        if z.ndim != 3 or z.shape[-1] != self.embedding_dim:
            raise ValueError(f"expected (B,T,{self.embedding_dim}) input; got {tuple(z.shape)}")
        orig_dtype = z.dtype
        z_f = z.float()
        z_hat = torch.zeros_like(z_f)
        indices_by_level: List[torch.Tensor] = []
        losses = []
        full_len = z_f.shape[1]

        for i, level_len in enumerate(self.levels):
            residual = z_f - z_hat.detach()
            level_input = self._resize(residual, level_len, "area") if i < len(self.levels) - 1 else residual
            flat = level_input.reshape(-1, self.embedding_dim)
            codebook = self.embeddings[i].weight
            with torch.no_grad():
                indices = self._distances(flat.detach(), codebook).argmin(dim=-1).view(z_f.shape[0], level_len)
            quantized = self.embeddings[i](indices)
            flat_indices = indices.reshape(-1)

            if self.training and update_ema:
                if mask is not None:
                    level_mask = (
                        self._resize(mask.float(), level_len, "area") > 0.5
                        if i < len(self.levels) - 1
                        else (mask > 0)
                    )
                    valid_flat = level_mask.reshape(-1)
                    if valid_flat.any():
                        self._ema_update(i, flat.detach()[valid_flat], flat_indices.detach()[valid_flat])
                else:
                    self._ema_update(i, flat.detach(), flat_indices.detach())
                quantized = self.embeddings[i](indices)

            commitment = F.mse_loss(level_input, quantized.detach())
            losses.append(self.commitment_cost * commitment)
            indices_by_level.append(indices)

            upsampled = self._resize(quantized, full_len, "linear")
            z_hat = z_hat + upsampled

        z_q_ste = z + (z_hat.to(orig_dtype) - z).detach()
        loss = torch.stack(losses).sum() if losses else z.new_zeros(())
        return z_q_ste, indices_by_level, loss


class MultiscaleResidualVQTokenizer(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128, levels: Sequence[int] = LEVELS):
        super().__init__()
        self.levels = tuple(levels)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, self.levels[-1], self.hidden_dim) * 0.02)
        self.encoder_norm = nn.LayerNorm(self.hidden_dim)
        self.rvq = MultiscaleResidualVectorQuantizer(
            num_embeddings=NUM_CONCEPTS, embedding_dim=self.hidden_dim, levels=self.levels
        )
        self.decoder_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

    @property
    def quantizers(self):
        return self.rvq.embeddings

    def decode_quantized(self, z_q: torch.Tensor) -> torch.Tensor:
        h_dec = self.decoder_proj(z_q)
        return F.linear(h_dec, self.token_embed.weight)

    @torch.no_grad()
    def decode_code_indices(
        self,
        code_indices: Sequence[torch.Tensor],
        output_length: Optional[int] = None,
    ) -> torch.Tensor:
        z_q = self.rvq.decode_indices(
            code_indices,
            output_length=output_length or self.levels[-1],
        )
        return self.decode_quantized(z_q)

    def encode(self, token_ids: torch.Tensor, update_ema: bool = True):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        max_len = self.levels[-1]
        if token_ids.shape[1] < max_len:
            token_ids = F.pad(token_ids, (0, max_len - token_ids.shape[1]), value=0)
        else:
            token_ids = token_ids[:, :self.levels[-1]]

        non_pad = (token_ids != 0)
        mask = non_pad.unsqueeze(-1)
        e = self.token_embed(token_ids) + self.pos_embed[:, :max_len, :]
        e = self.encoder_norm(e) * mask.to(e.dtype)

        z_q, indices, commit_loss = self.rvq(e, mask=mask, update_ema=update_ema)

        recon_logits = self.decode_quantized(z_q)
        if non_pad.any():
            recon_loss = F.cross_entropy(
                recon_logits.reshape(-1, self.vocab_size),
                token_ids.reshape(-1),
                ignore_index=0,
            )
        else:
            recon_loss = commit_loss * 0.0

        total_vq_loss = commit_loss + recon_loss

        mask_f = mask.to(e.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        h0 = (e * mask_f).sum(dim=1) / denom

        return indices + [token_ids], total_vq_loss, h0


class NextConceptGenerator(nn.Module):
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
        self.stage_lengths = self.levels + (self.levels[-1],)
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
    def generate_code_hierarchy(
        self,
        prefix_ctx: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> List[torch.Tensor]:
        if prefix_ctx is None:
            raise ValueError("prefix_ctx is required for conditional generation")
        indices_by_level: List[torch.Tensor] = []
        for level_idx in range(len(self.levels)):
            logits = self._level_logits(indices_by_level, level_idx, prefix_ctx)
            indices = self._sample(logits, temperature, top_k)
            indices_by_level.append(indices)
        return indices_by_level

    @torch.no_grad()
    def generate(
        self,
        prefix_ctx: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        top_k: int = 0,
    ):
        if prefix_ctx is None:
            raise ValueError("prefix_ctx is required for conditional generation")
        idx_list = self.generate_code_hierarchy(
            prefix_ctx=prefix_ctx,
            temperature=temperature,
            top_k=top_k,
        )
        token_logits = self._level_logits(idx_list, len(self.levels), prefix_ctx)
        tokens = _sample_tokens_no_repeat(
            token_logits,
            temperature=temperature,
            top_k=top_k,
            min_tokens=3,
        )
        idx_list.append(tokens)
        return idx_list


class ToolRouter(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_concepts: int = NUM_CONCEPTS):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.head = nn.Linear(hidden_dim, self.num_concepts)
        self.lang_head = nn.Linear(hidden_dim, len(LANGUAGE_CONCEPT_IDS))
        self.tool_head = nn.Linear(hidden_dim, len(TOOL_CLASS_CONCEPT_IDS))

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        logits = self.head(h0)
        lang_logits = self.lang_head(h0)
        tool_logits = self.tool_head(h0)
        for idx, cid in enumerate(LANGUAGE_CONCEPT_IDS):
            logits[:, cid] = logits[:, cid] + lang_logits[:, idx]
        for idx, cid in enumerate(TOOL_CLASS_CONCEPT_IDS[1:], start=1):
            logits[:, cid] = logits[:, cid] + (tool_logits[:, idx] - tool_logits[:, 0])
        return logits

    def route_decoupled(
        self,
        h0: torch.Tensor,
        tool_conf_threshold: float = 0.65,
        margin_threshold: float = 0.15,
    ) -> Dict[str, object]:
        lang_logits = self.lang_head(h0)
        tool_logits = self.tool_head(h0)

        lang_probs = F.softmax(lang_logits, dim=-1)
        lang_conf, lang_idx = lang_probs.max(dim=-1)
        lang_cid = LANGUAGE_CONCEPT_IDS[int(lang_idx[0].item())]

        tool_probs = F.softmax(tool_logits, dim=-1)
        top2_probs, top2_idx = torch.topk(tool_probs, k=2, dim=-1)
        best_tool_prob = float(top2_probs[0, 0].item())
        second_tool_prob = float(top2_probs[0, 1].item())
        best_tool_idx = int(top2_idx[0, 0].item())
        margin = best_tool_prob - second_tool_prob

        if best_tool_idx != 0 and best_tool_prob >= tool_conf_threshold and margin >= margin_threshold:
            tool_cid = TOOL_CLASS_CONCEPT_IDS[best_tool_idx]
            tool_name = EXPLICIT_TOOL_MAP.get(tool_cid)
            if tool_name is not None:
                return {
                    "type": "TOOL_CALL",
                    "tool_name": tool_name,
                    "concept_id": tool_cid,
                    "lang_concept_id": lang_cid,
                    "confidence": round(best_tool_prob, 4),
                    "abstained": False,
                }

        return {
            "type": "TEXT_STREAM",
            "tool_name": None,
            "concept_id": lang_cid,
            "lang_concept_id": lang_cid,
            "confidence": round(float(lang_conf[0].item()), 4),
            "abstained": best_tool_idx != 0,
        }


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

    def _build_prompt_ctx(self, prompt_ids: torch.Tensor, h0_prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        short_ids = prompt_ids[:, :10]
        tok_ctx = self.generator.token_embed(short_ids) + self.tokenizer.pos_embed[:, :short_ids.shape[1], :]
        if h0_prompt is None:
            _, _, h0_prompt = self.tokenizer.encode(prompt_ids, update_ema=False)
        return torch.cat([h0_prompt.unsqueeze(1), tok_ctx], dim=1)

    def forward(self, prompt_ids: torch.Tensor, reply_ids: Optional[torch.Tensor] = None):
        if reply_ids is None:
            reply_ids = prompt_ids
        _, _, h0_prompt = self.tokenizer.encode(prompt_ids, update_ema=False)
        tool_logits = self.router(h0_prompt)
        prompt_ctx = self._build_prompt_ctx(prompt_ids, h0_prompt=h0_prompt)
        idx_list, loss_commit, _ = self.tokenizer.encode(reply_ids, update_ema=True)
        logits_per_level = self.generator(idx_list, prefix_ctx=prompt_ctx)
        return logits_per_level, idx_list, loss_commit, tool_logits

    @torch.no_grad()
    def route(
        self,
        token_ids: torch.Tensor,
        tool_threshold: int = TOOL_THRESHOLD,
        tool_conf_threshold: float = 0.65,
    ):
        _, _, h0 = self.tokenizer.encode(token_ids, update_ema=False)
        decision = self.router.route_decoupled(h0, tool_conf_threshold=tool_conf_threshold)
        return decision

    @torch.no_grad()
    def decode_code_hierarchy(
        self,
        code_indices: Sequence[torch.Tensor],
        temperature: float = 0.0,
        top_k: int = 0,
        min_tokens: int = 3,
    ) -> torch.Tensor:
        logits = self.tokenizer.decode_code_indices(
            code_indices,
            output_length=self.max_len,
        )
        return _sample_tokens_no_repeat(
            logits,
            temperature=temperature,
            top_k=top_k,
            min_tokens=min_tokens,
        )

    @torch.no_grad()
    def chat(self, token_ids: torch.Tensor, temperature: float = 0.0, top_k: int = 0):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.shape[1] < self.max_len:
            token_ids = F.pad(token_ids, (0, self.max_len - token_ids.shape[1]), value=0)
        else:
            token_ids = token_ids[:, :self.max_len]

        _, _, h0_prompt = self.tokenizer.encode(token_ids, update_ema=False)
        prompt_ctx = self._build_prompt_ctx(token_ids, h0_prompt=h0_prompt)

        generated = self.generator.generate(
            prefix_ctx=prompt_ctx,
            temperature=temperature,
            top_k=top_k,
        )
        code_indices = generated[:len(self.generator.levels)]
        generated_token_ids = generated[-1]
        return generated_token_ids, code_indices


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

    clamped_concepts = concept_id_batch.clamp(0, NUM_CONCEPTS - 1)
    loss_router_base = F.cross_entropy(
        tool_logits, clamped_concepts, weight=router_class_weights
    )

    _, _, h0_prompt = model.tokenizer.encode(prompt_ids, update_ema=False)
    lang_logits = model.router.lang_head(h0_prompt)
    tool_head_logits = model.router.tool_head(h0_prompt)

    device = prompt_ids.device
    lang_targets = torch.full_like(clamped_concepts, fill_value=-100)
    tool_targets = torch.zeros_like(clamped_concepts)

    for cid, lang_idx in LANG_INDEX_MAP.items():
        lang_targets = torch.where(clamped_concepts == cid, torch.tensor(lang_idx, device=device), lang_targets)
    for cid, tool_idx in TOOL_INDEX_MAP.items():
        if cid != 0:
            tool_targets = torch.where(clamped_concepts == cid, torch.tensor(tool_idx, device=device), tool_targets)

    loss_lang = (
        F.cross_entropy(lang_logits, lang_targets, ignore_index=-100)
        if (lang_targets != -100).any()
        else loss_router_base * 0.0
    )
    loss_tool_bin = F.cross_entropy(tool_head_logits, tool_targets)
    loss_router = 0.5 * loss_router_base + 0.5 * (loss_lang + loss_tool_bin)

    total = loss_levels + loss_commit.mean() + 0.5 * loss_router
    return total, {
        "loss_total": round(float(total.detach().item()), 4),
        "loss_levels": round(float(loss_levels.detach().item()), 4),
        "loss_commit": round(float(loss_commit.detach().mean().item()), 4),
        "loss_router": round(float(loss_router.detach().item()), 4),
    }
