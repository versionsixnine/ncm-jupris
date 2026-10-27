"""Training entry point for JuprisMini-NCM.

Keeps the existing corpus/vocabulary filenames and the ncm_core public API.
Use --smoke-test to validate one batch without saving/overwriting checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ncm_core import JuprisNCM, compute_ncm_loss, MAX_LEN, NUM_CONCEPTS

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
    """Split a serialized User/Mini-Jup pair; otherwise use text as both sides."""
    if not isinstance(raw_text, str):
        return "", ""
    match = re.search(r"\bMini-Jup\s*:", raw_text, flags=re.IGNORECASE)
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


def get_vocab(jsonl_path: str = CORPUS_FILE) -> dict:
    # Keep an existing vocabulary stable so checkpoints and token IDs stay aligned.
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
    def __init__(self, jsonl_path: str, vocab: dict):
        self.data = []
        self.skipped = 0
        for line_no, obj in _read_jsonl(jsonl_path):
            user_text, reply_text = split_user_reply(obj.get("text", ""))
            if not user_text or not reply_text:
                self.skipped += 1
                continue
            try:
                concept_id = int(obj.get("concept_id", 0))
            except (TypeError, ValueError):
                print(f"[WARN] Skip invalid concept_id at {jsonl_path}:{line_no}")
                self.skipped += 1
                continue
            if not 0 <= concept_id < NUM_CONCEPTS:
                print(f"[WARN] Skip out-of-range concept_id={concept_id} at line {line_no}; expected 0..{NUM_CONCEPTS - 1}")
                self.skipped += 1
                continue
            p_ids = text_to_ids(user_text, vocab)
            r_ids = text_to_ids(reply_text, vocab)
            self.data.append((
                torch.tensor(p_ids, dtype=torch.long),
                torch.tensor(r_ids, dtype=torch.long),
                concept_id,
            ))
        if not self.data:
            raise ValueError(f"No valid training examples found in {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def compute_router_class_weights(dataset: JuprisCorpus, num_concepts: int = NUM_CONCEPTS) -> torch.Tensor:
    counts = Counter(cid for _, _, cid in dataset.data if 0 <= cid < num_concepts)
    weights = torch.ones(num_concepts, dtype=torch.float32)
    total = sum(counts.values())
    classes_seen = len(counts)
    if total and classes_seen:
        for cid, count in counts.items():
            weights[cid] = total / (classes_seen * count)
    shown = {cid: round(float(weights[cid]), 3) for cid in sorted(counts)}
    print(f"[*] Router class weights (observed classes): {shown}")
    return weights


def lr_lambda(step: int, warmup_steps: int, total_steps: int):
    if warmup_steps > 0 and step < warmup_steps:
        return max(1, step) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def run_one_batch(model, batch, optimizer, device, router_weights):
    prompt_ids, reply_ids, concept_ids = (x.to(device) for x in batch)
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = compute_ncm_loss(
        model,
        prompt_ids,
        concept_ids,
        pad_id=PAD_ID,
        router_class_weights=router_weights,
        reply_ids=reply_ids,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu())}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("Non-finite gradient norm")
    optimizer.step()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train JuprisMini-NCM")
    parser.add_argument("--smoke-test", action="store_true", help="run one forward/backward batch; do not save checkpoints")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--corpus", default=CORPUS_FILE, help="Training corpus JSONL (e.g. teacher-curated corpus)")
    parser.add_argument("--checkpoint-prefix", default="jupris_ncm_core", help="Checkpoint filename prefix; avoid overwriting a known-good model")
    parser.add_argument("--resume", default="", help="Resume NCM weights from a compatible checkpoint")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = get_vocab(args.corpus)
    dataset = JuprisCorpus(args.corpus, vocab)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)) if args.smoke_test else args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    print(f"[*] Dataset: {len(dataset)} examples | skipped: {dataset.skipped} | vocab: {len(vocab)} | device={device}")

    model = JuprisNCM(vocab_size=VOCAB_SIZE, hidden_dim=128, num_layers=4, num_heads=4).to(device)
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        except TypeError:  # compatibility with older PyTorch
            checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint does not contain a state_dict")
        model.load_state_dict(checkpoint, strict=True)
        print(f"[*] Resumed compatible NCM weights from {args.resume}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
    router_weights = compute_router_class_weights(dataset).to(device)

    if args.smoke_test:
        model.train()
        metrics = run_one_batch(model, next(iter(loader)), optimizer, device, router_weights)
        print(f"[PASS] One-batch forward/backward/optimizer step: {metrics}")
        print("[INFO] Smoke test does not save or overwrite checkpoints.")
        return

    total_steps = max(1, args.epochs * len(loader))
    warmup_steps = min(max(1, int(0.03 * total_steps)), max(1, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, warmup_steps, total_steps),
    )

    last_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            metrics = run_one_batch(model, batch, optimizer, device, router_weights)
            scheduler.step()
            last_metrics = metrics
            epoch_loss += metrics["loss_total"]
            if batch_idx % 100 == 0:
                print(f"  [E{epoch} B{batch_idx} lr={scheduler.get_last_lr()[0]:.2e}] {metrics}")
        print(f"[E{epoch}] mean_loss={epoch_loss / max(1, len(loader)):.4f}")

    checkpoint_v2 = f"{args.checkpoint_prefix}_v2.pt"
    checkpoint_legacy = f"{args.checkpoint_prefix}.pt"
    torch.save(model.state_dict(), checkpoint_v2)
    torch.save(model.state_dict(), checkpoint_legacy)
    print(f"[+] Saved {checkpoint_v2} and {checkpoint_legacy}")
    if last_metrics is not None:
        print(f"Score: {last_metrics['loss_total'] * 100:.2f}")


if __name__ == "__main__":
    main()
