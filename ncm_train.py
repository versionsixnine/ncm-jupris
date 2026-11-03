"""Training entry point for JuprisMini-NCM (Stage 3 Frozen VQ-VAE + Scheduled Sampling).

Loads and freezes the trained Stage-2 VQ-VAE tokenizer (jupris_vqvae_stage2_v3.pt)
so codebooks do not drift or collapse during generator training, and trains
NextConceptGenerator with coarse-to-fine scheduled sampling to eliminate the
Teacher-Forced (83.4%) vs Free-Running (19.3%) exposure bias gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ncm_core import (
    JuprisNCM,
    compute_ncm_loss,
    LANG_INDEX_MAP,
    MAX_LEN,
    NUM_CONCEPTS,
    TOOL_INDEX_MAP,
)

VOCAB_SIZE = 16384
VOCAB_FILE = "jupris_vocab.json"
CORPUS_FILE = "jupris_corpus.jsonl"
PAD_ID = 0
UNK_ID = 1


def normalize_words(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def split_user_reply(raw_text: str) -> Tuple[str, str]:
    """Split a serialized User/Mini-Jup or User/Assistant pair."""
    if not isinstance(raw_text, str):
        return "", ""
    match = re.search(r"\b(?:Mini-Jup|Assistant)\s*:", raw_text, flags=re.IGNORECASE)
    if match:
        u_part = raw_text[:match.start()]
        a_part = raw_text[match.end():]
        u_part = re.sub(r"^\s*User\s*:\s*", "", u_part, flags=re.IGNORECASE)
        return normalize_words(u_part), normalize_words(a_part)
    cleaned = normalize_words(raw_text)
    return cleaned, cleaned


def _read_jsonl(jsonl_path: str):
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Corpus file not found: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skip invalid JSON at {jsonl_path}:{line_no}: {exc.msg}")
                continue
            if isinstance(obj, dict):
                yield line_no, obj
            else:
                print(f"[WARN] Skip non-object JSON at {jsonl_path}:{line_no}")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_vocab(jsonl_path: str = CORPUS_FILE) -> dict:
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict) or vocab.get("[PAD]") != PAD_ID or vocab.get("[UNK]") != UNK_ID:
            raise ValueError(f"{VOCAB_FILE} must be a token->ID object with [PAD]=0 and [UNK]=1")
        try:
            ids = [int(v) for v in vocab.values()]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{VOCAB_FILE} contains non-integer token IDs") from exc
        if any(i < 0 or i >= VOCAB_SIZE for i in ids):
            raise ValueError(f"{VOCAB_FILE} contains token IDs outside [0, {VOCAB_SIZE - 1}]")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{VOCAB_FILE} contains duplicate token IDs")
        return {str(k): int(v) for k, v in vocab.items()}

    print(f"[*] Building vocabulary from {jsonl_path} (one-time initialization)")
    vocab = {"[PAD]": PAD_ID, "[UNK]": UNK_ID}
    for _, obj in _read_jsonl(jsonl_path):
        user_text, reply_text = split_user_reply(obj.get("text", ""))
        for word in (user_text + " " + reply_text).split():
            if word not in vocab and len(vocab) < VOCAB_SIZE:
                vocab[word] = len(vocab)
    if len(vocab) <= 2:
        raise ValueError(f"No usable text found in {jsonl_path}; vocabulary was not created")
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab


def text_to_ids(text: str, vocab: dict, max_len: int = MAX_LEN) -> List[int]:
    cleaned = normalize_words(text)
    ids = [int(vocab.get(w, UNK_ID)) for w in cleaned.split()[:max_len]]
    ids = [i if 0 <= i < VOCAB_SIZE else UNK_ID for i in ids]
    return ids + [PAD_ID] * (max_len - len(ids))


class JuprisCorpus(Dataset):
    def __init__(self, records: Sequence[Tuple[torch.Tensor, torch.Tensor, int]]):
        self.data = list(records)
        if not self.data:
            raise ValueError("Dataset split received 0 records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_and_split_corpus(
    jsonl_path: str,
    vocab: dict,
    seed: int = 42,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
) -> Tuple[JuprisCorpus, JuprisCorpus, JuprisCorpus, Dict[str, int]]:
    by_concept: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = defaultdict(list)
    seen_pairs = set()
    skipped = 0
    deduped = 0

    for line_no, obj in _read_jsonl(jsonl_path):
        user_text, reply_text = split_user_reply(obj.get("text", ""))
        if not user_text or not reply_text:
            skipped += 1
            continue
        try:
            concept_id = int(obj.get("concept_id", 0))
        except (TypeError, ValueError):
            print(f"[WARN] Skip invalid concept_id at {jsonl_path}:{line_no}")
            skipped += 1
            continue
        if not 0 <= concept_id < NUM_CONCEPTS:
            print(f"[WARN] Skip out-of-range concept_id={concept_id} at line {line_no}")
            skipped += 1
            continue

        pair_key = (user_text, reply_text, concept_id)
        if pair_key in seen_pairs:
            deduped += 1
            continue
        seen_pairs.add(pair_key)

        p_ids = text_to_ids(user_text, vocab)
        r_ids = text_to_ids(reply_text, vocab)
        by_concept[concept_id].append((
            torch.tensor(p_ids, dtype=torch.long),
            torch.tensor(r_ids, dtype=torch.long),
            concept_id,
        ))

    rng = random.Random(seed)
    train_records, val_records, test_records = [], [], []

    for cid in sorted(by_concept.keys()):
        bucket = list(by_concept[cid])
        rng.shuffle(bucket)
        n = len(bucket)
        if n <= 2:
            train_records.extend(bucket)
            continue
        n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
        n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
        if n_test + n_val >= n:
            n_test = 0
            n_val = 1
        test_records.extend(bucket[:n_test])
        val_records.extend(bucket[n_test:n_test + n_val])
        train_records.extend(bucket[n_test + n_val:])

    if not val_records:
        val_records = train_records[:max(1, len(train_records) // 10)]
    if not test_records:
        test_records = val_records

    stats = {
        "total_unique": len(train_records) + len(val_records) + len(test_records),
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
        "skipped": skipped,
        "deduped": deduped,
    }
    return (
        JuprisCorpus(train_records),
        JuprisCorpus(val_records),
        JuprisCorpus(test_records),
        stats,
    )


def compute_router_class_weights(dataset: JuprisCorpus, num_concepts: int = NUM_CONCEPTS) -> torch.Tensor:
    counts = Counter(cid for _, _, cid in dataset.data if 0 <= cid < num_concepts)
    weights = torch.ones(num_concepts, dtype=torch.float32)
    total = sum(counts.values())
    classes_seen = len(counts)
    if total and classes_seen:
        for cid, count in counts.items():
            weights[cid] = total / (classes_seen * count)
    shown = {cid: round(float(weights[cid]), 3) for cid in sorted(counts)}
    print(f"[*] Router class weights (train split): {shown}")
    return weights


def lr_lambda(step: int, warmup_steps: int, total_steps: int):
    if warmup_steps > 0 and step < warmup_steps:
        return max(1, step) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def compute_stage3_scheduled_loss(
    model: JuprisNCM,
    prompt_ids: torch.Tensor,
    reply_ids: torch.Tensor,
    concept_id_batch: torch.Tensor,
    pad_id: int = 0,
    router_class_weights: Optional[torch.Tensor] = None,
    ss_prob: float = 0.5,
    freeze_vq: bool = True,
):
    """Compute NCM loss with Frozen VQ-VAE, Coarse-to-Fine Scheduled Sampling, and Unigram Anti-Repeat."""
    if freeze_vq:
        model.tokenizer.eval()
        with torch.no_grad():
            _, _, h0_prompt = model.tokenizer.encode(prompt_ids, update_ema=False)
            idx_list, loss_commit, _ = model.tokenizer.encode(reply_ids, update_ema=False)
        loss_commit = loss_commit.detach() * 0.0
    else:
        _, _, h0_prompt = model.tokenizer.encode(prompt_ids, update_ema=False)
        idx_list, loss_commit, _ = model.tokenizer.encode(reply_ids, update_ema=True)

    tool_logits = model.router(h0_prompt)
    prompt_ctx = model._build_prompt_ctx(prompt_ids, h0_prompt=h0_prompt)

    n_vq = len(model.generator.levels)
    conditioning_indices: List[torch.Tensor] = []
    logits_per_level: List[torch.Tensor] = []

    for i in range(n_vq + 1):
        logits_i = model.generator._level_logits(conditioning_indices, i, prompt_ctx)
        logits_per_level.append(logits_i)
        if i < n_vq:
            if model.training and ss_prob > 0.0 and random.random() < ss_prob:
                with torch.no_grad():
                    pred_i = logits_i.detach().argmax(dim=-1)
                    if random.random() < 0.5:
                        # Mix per-position predicted vs gold codes to regularize downstream levels
                        mask = torch.rand_like(pred_i.float()) < ss_prob
                        next_cond = torch.where(mask, pred_i, idx_list[i])
                    else:
                        next_cond = pred_i
                conditioning_indices.append(next_cond)
            else:
                conditioning_indices.append(idx_list[i])

    level_losses = []
    for i, logits in enumerate(logits_per_level):
        target = idx_list[i]
        _, _, classes = logits.shape
        if i == len(logits_per_level) - 1:
            weights = torch.ones(classes, device=logits.device, dtype=logits.dtype)
            weights[pad_id] = 0.15
            ce_tok = F.cross_entropy(logits.reshape(-1, classes), target.reshape(-1), weight=weights)

            # Adjacent repetition penalty on non-PAD token probabilities to stop "siap siap siap siap"
            probs = F.softmax(logits, dim=-1)
            nonpad_probs = probs[:, :, 1:]  # exclude PAD=0 from repetition penalty
            adj_sim = (nonpad_probs[:, 1:, :] * nonpad_probs[:, :-1, :]).sum(dim=-1)
            active_adj = (target[:, 1:] != pad_id).float()
            rep_pen = (adj_sim * active_adj).sum() / active_adj.sum().clamp_min(1.0)

            loss_i = 3.0 * ce_tok + 1.5 * rep_pen
        else:
            loss_i = 0.5 * F.cross_entropy(logits.reshape(-1, classes), target.reshape(-1))
        level_losses.append(loss_i)
    loss_levels = torch.stack(level_losses).sum()

    clamped_concepts = concept_id_batch.clamp(0, NUM_CONCEPTS - 1)
    loss_router_base = F.cross_entropy(
        tool_logits, clamped_concepts, weight=router_class_weights
    )

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


def run_one_batch(model, batch, optimizer, device, router_weights, ss_prob: float = 0.5, freeze_vq: bool = True):
    prompt_ids, reply_ids, concept_ids = (x.to(device) for x in batch)
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = compute_stage3_scheduled_loss(
        model,
        prompt_ids,
        reply_ids,
        concept_ids,
        pad_id=PAD_ID,
        router_class_weights=router_weights,
        ss_prob=ss_prob,
        freeze_vq=freeze_vq,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu())}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("Non-finite gradient norm")
    optimizer.step()
    return metrics


@torch.no_grad()
def evaluate_split(model: JuprisNCM, loader: DataLoader, device: torch.device, router_weights: torch.Tensor) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_levels = 0.0
    total_commit = 0.0
    total_router = 0.0
    batches = 0

    lang_correct = 0
    lang_total = 0
    tool_correct = 0
    tool_total = 0
    free_tok_ok = 0
    free_tok_tot = 0
    free_exact = 0
    free_rows = 0

    for batch in loader:
        prompt_ids, reply_ids, concept_ids = (x.to(device) for x in batch)
        _, metrics = compute_ncm_loss(
            model,
            prompt_ids,
            concept_ids,
            pad_id=PAD_ID,
            router_class_weights=router_weights,
            reply_ids=reply_ids,
        )
        total_loss += metrics["loss_total"]
        total_levels += metrics["loss_levels"]
        total_commit += metrics["loss_commit"]
        total_router += metrics["loss_router"]
        batches += 1

        _, _, h0 = model.tokenizer.encode(prompt_ids, update_ema=False)
        lang_preds = model.router.lang_head(h0).argmax(dim=-1)
        tool_preds = model.router.tool_head(h0).argmax(dim=-1)

        for idx_in_batch, cid_val in enumerate(concept_ids.tolist()):
            if cid_val in LANG_INDEX_MAP:
                lang_total += 1
                if int(lang_preds[idx_in_batch].item()) == LANG_INDEX_MAP[cid_val]:
                    lang_correct += 1
                tool_total += 1
                if int(tool_preds[idx_in_batch].item()) == 0:
                    tool_correct += 1
            elif cid_val in TOOL_INDEX_MAP:
                tool_total += 1
                if int(tool_preds[idx_in_batch].item()) == TOOL_INDEX_MAP[cid_val]:
                    tool_correct += 1

        # Evaluate FREE-RUNNING generation quality directly on validation/test split
        fr_pred, _ = model.chat(prompt_ids, temperature=0.0, top_k=0)
        for r in range(reply_ids.shape[0]):
            free_rows += 1
            gold_row = [int(x) for x in reply_ids[r].tolist() if int(x) != PAD_ID]
            pred_row = []
            for x in fr_pred[r].tolist():
                if int(x) == PAD_ID:
                    break
                pred_row.append(int(x))
            if pred_row == gold_row:
                free_exact += 1
            for pos, tid in enumerate(gold_row):
                free_tok_tot += 1
                if pos < fr_pred.shape[1] and int(fr_pred[r, pos].item()) == tid:
                    free_tok_ok += 1

    denom = max(1, batches)
    return {
        "val_loss": round(total_loss / denom, 4),
        "val_levels": round(total_levels / denom, 4),
        "val_commit": round(total_commit / denom, 4),
        "val_router": round(total_router / denom, 4),
        "lang_acc": round(lang_correct / max(1, lang_total), 4),
        "tool_acc": round(tool_correct / max(1, tool_total), 4),
        "free_tok_acc": round(free_tok_ok / max(1, free_tok_tot), 4),
        "free_exact": round(free_exact / max(1, free_rows), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train JuprisMini-NCM")
    parser.add_argument("--smoke-test", action="store_true", help="run one forward/backward batch; do not save checkpoints")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--corpus", default=CORPUS_FILE, help="Training corpus JSONL")
    parser.add_argument("--freeze-vq", default="jupris_vqvae_stage2_v3.pt", help="Stage-2 VQ-VAE checkpoint to load and freeze")
    parser.add_argument("--checkpoint-prefix", default="jupris_ncm_v9_stage3_frozen", help="Checkpoint filename prefix")
    parser.add_argument("--resume", default="", help="Resume NCM weights from a compatible checkpoint")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be >= 1")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = get_vocab(args.corpus)
    train_set, val_set, test_set, split_stats = load_and_split_corpus(args.corpus, vocab, seed=args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=min(args.batch_size, len(train_set)) if args.smoke_test else args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(
        f"[*] Splits -> train: {split_stats['train']} | val: {split_stats['val']} | "
        f"test: {split_stats['test']} | deduped: {split_stats['deduped']} | "
        f"skipped: {split_stats['skipped']} | vocab: {len(vocab)} | device={device}"
    )

    model = JuprisNCM(vocab_size=VOCAB_SIZE, hidden_dim=128, num_layers=4, num_heads=4).to(device)
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint, strict=True)
        print(f"[*] Resumed compatible NCM weights from {args.resume}")

    freeze_vq_active = False
    if args.freeze_vq and os.path.isfile(args.freeze_vq):
        try:
            vq_payload = torch.load(args.freeze_vq, map_location=device, weights_only=True)
        except TypeError:
            vq_payload = torch.load(args.freeze_vq, map_location=device)
        if isinstance(vq_payload, dict) and "tokenizer_state_dict" in vq_payload:
            model.tokenizer.load_state_dict(vq_payload["tokenizer_state_dict"], strict=True)
            for p in model.tokenizer.parameters():
                p.requires_grad_(False)
            model.tokenizer.eval()
            freeze_vq_active = True
            print(f"[*] Loaded and FROZE Stage-2 VQ-VAE tokenizer from {args.freeze_vq}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    router_weights = compute_router_class_weights(train_set).to(device)

    if args.smoke_test:
        model.train()
        metrics = run_one_batch(model, next(iter(train_loader)), optimizer, device, router_weights, ss_prob=0.5, freeze_vq=freeze_vq_active)
        print(f"[PASS] One-batch forward/backward/optimizer step: {metrics}")
        print("[INFO] Smoke test does not save or overwrite checkpoints.")
        return

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = min(max(1, int(0.04 * total_steps)), max(1, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, warmup_steps, total_steps),
    )

    best_free_score = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if freeze_vq_active:
            model.tokenizer.eval()
        epoch_loss = 0.0
        # Ramp up scheduled sampling from 0.25 (E1) to 0.75 (E6)
        ss_prob = min(0.75, 0.15 + 0.12 * epoch)
        for batch_idx, batch in enumerate(train_loader):
            metrics = run_one_batch(model, batch, optimizer, device, router_weights, ss_prob=ss_prob, freeze_vq=freeze_vq_active)
            scheduler.step()
            epoch_loss += metrics["loss_total"]
            if batch_idx % 200 == 0:
                print(f"  [E{epoch} B{batch_idx} ss={ss_prob:.2f} lr={scheduler.get_last_lr()[0]:.2e}] {metrics}")

        train_mean = round(epoch_loss / max(1, len(train_loader)), 4)
        val_metrics = evaluate_split(model, val_loader, device, router_weights)
        print(f"[E{epoch}] train_loss={train_mean:.4f} | {val_metrics}")
        history.append({"epoch": epoch, "train_loss": train_mean, **val_metrics})

        # Select best checkpoint based on actual FREE-RUNNING accuracy + router accuracy
        free_score = val_metrics["free_tok_acc"] + val_metrics["free_exact"] + 0.2 * val_metrics["tool_acc"]
        if free_score > best_free_score:
            best_free_score = free_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_split(model, test_loader, device, router_weights)
    print(f"[TEST] Held-out test metrics: {test_metrics}")

    checkpoint_v2 = f"{args.checkpoint_prefix}_v2.pt"
    checkpoint_legacy = f"{args.checkpoint_prefix}.pt"
    torch.save(model.state_dict(), checkpoint_v2)
    torch.save(model.state_dict(), checkpoint_legacy)

    meta_path = f"{args.checkpoint_prefix}_v2.meta.json"
    metadata = {
        "architecture": "JuprisNCM_v9_stage3_frozen_vq_scheduled_sampling",
        "vocab_size": VOCAB_SIZE,
        "active_vocab_entries": len(vocab),
        "vocab_sha256": file_sha256(VOCAB_FILE) if os.path.isfile(VOCAB_FILE) else None,
        "corpus_file": args.corpus,
        "corpus_sha256": file_sha256(args.corpus),
        "frozen_vq_checkpoint": args.freeze_vq if freeze_vq_active else None,
        "checkpoint_sha256": file_sha256(checkpoint_v2),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "split_stats": split_stats,
        "best_free_score": round(best_free_score, 4),
        "test_metrics": test_metrics,
        "epoch_history": history,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Saved {checkpoint_v2}, {checkpoint_legacy}, and {meta_path}")


if __name__ == "__main__":
    main()