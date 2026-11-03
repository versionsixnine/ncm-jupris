# JuprisMini-NCM — Trilingual Multiscale Concept Model & Tool Router
Question  : NEXT CONCEPT MODEELING -CHAT ? ..

**Status:** Production Ready (`v6-wsp` / CPU Inference ~33–45ms)
**Primary Method Reference:** [HazyResearch/ncm](https://github.com/HazyResearch/ncm)
**Scope:** Lightweight edge-oriented trilingual chat model (Indonesian, Javanese, English) with decoupled concept/tool routing and zero-gradient weight-space surgical editing.
---
> ⚠️ **WARNING — EXPERIMENTAL STATUS**
> This is **NOT** a production-ready daily-chat model. JuprisMini-NCM is an **experimental architecture** currently usable only as a **chat/tool-calling proof-of-concept**. Output quality, routing accuracy, and general conversational robustness are still immature and require further training, calibration, and testing before any reliable daily-use deployment.
---

## 1. Project Goal

**JuprisMini-NCM** is a compact neural architecture (~25MB checkpoint) built to:
1. Chat coherently in **Indonesian** (casual/formal), **Javanese** (ngoko/krama), and **Technical English**.
2. Handle mixed-language and trilingual code-switching naturally.
3. Route and invoke registered system/web/hardware tools deterministically via an explicit concept router.
4. Run entirely on CPU/edge hardware with ~33–45ms latency without external web-framework bloat.

---

## 2. Core Architecture & Engineering Rules

### 2.1 Upstream NCM Baseline
This project preserves the core **Next-Concept Model (NCM)** paradigm from `HazyResearch/ncm`:
- **Multiscale Residual VQ Tokenizer (`VQ-VAE`):** Encodes a token sequence into a hierarchy of discrete latent codes across multiple scales (`levels`).
- **Next-Concept Generator (`NextConceptGenerator`):** Predicts the discrete code hierarchy from coarse to fine (`head_coarse[0..L-1]` followed by `head_token`). Each level predicts its positions in parallel rather than token-by-token autoregressive decoding.

### 2.2 Explicit Separation: Concept Labels vs. VQ Codebooks
Dataset-level `concept_id` values are **supervision metadata for routing and filtering**, NOT indices inside the VQ-VAE learned codebooks:

| Concept ID | Route Type | Tool Name | Intended Semantic / Domain |
|---:|---|---|---|
| `120` | `TEXT_STREAM` | `null` | Indonesian casual / slang |
| `125` | `TEXT_STREAM` | `null` | Indonesian formal / technical |
| `150` | `TEXT_STREAM` | `null` | Javanese ngoko |
| `155` | `TEXT_STREAM` | `null` | Javanese krama / polite |
| `210` | `TEXT_STREAM` | `null` | Technical English |
| `220` | `TEXT_STREAM` | `null` | Trilingual / code-switching / translation |
| `480` | `TOOL_CALL` | `search_web` | Web search, browser, maps, location & routing |
| `481` | `TOOL_CALL` | `read_gpio` | Hardware / GPIO pin activation & telemetry |
| `482` | `TOOL_CALL` | `exec_python` | Sandboxed Python execution |
| `483` | `TOOL_CALL` | `check_server` | Edge / VPS server health & status telemetry |
| `484` | `TOOL_CALL` | `inspect_repo` | Repository & code inspection |

> **Strict Guardrail:** Never infer a tool ID by subtracting `480` from a VQ codebook index. Tool routing is handled by the dedicated router head (`model.route()`) with explicit schema mapping and confidence thresholding (`tool_conf_threshold = 0.65`).

---

## 3. Repository Structure (Production Layout)

```text
├── ncm_core.py                  # Core JuprisNCM, VQ-VAE Tokenizer, Concept Router & NextConceptGenerator
├── jupris_vocab.json            # Production word-level vocabulary (10,331 active tokens, 16,384 max capacity)
├── jupris_ncm_prod_ready.pt     # Production CPU checkpoint (25MB, Orthogonal + WSP calibrated)
├── ncm_api_server.py            # Zero-dependency Python ThreadingHTTPServer daemon (Port 8893 + Hot-Reload)
├── ncm_wsp_surgeon.py           # MIT Closed-Form Weight-Space Surgical Editor & Live Vocab Injector
├── jupris_cli.py                # Standalone CLI benchmark & interactive chat runner
├── ncm_train.py                 # Full multiscale VQ-VAE + Next-Concept training pipeline
├── systemd/
│   └── jupris-ncm-api.service   # Systemd unit for persistent background daemon on Port 8893
└── web/
    ├── index.php                # Tailwind CSS responsive chat dashboard
    └── api_proxy.php            # PHP cURL bridge to 127.0.0.1:8893 (/health & /v1/chat)
```

---

## 4. Quickstart & Production Operations

### 4.1 Run via CLI
```bash
python jupris_cli.py "kopi rokok wis" "piye kabare bosku" "food in indonesian" "nyalakan mesin"
```

### 4.2 Run the Native HTTP API Server (Port 8893)
`ncm_api_server.py` uses Python's built-in `ThreadingHTTPServer` (zero FastAPI/Uvicorn dependency overhead) and performs an automatic warm-up pass on boot:

```bash
sudo cp systemd/jupris-ncm-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jupris-ncm-api
```

**Endpoints:**
- `GET http://127.0.0.1:8893/health` — Returns server status, active checkpoint path, and vocabulary count.
- `POST http://127.0.0.1:8893/v1/chat` — Executes routing and coarse-to-fine generation:

```bash
curl -s -X POST http://127.0.0.1:8893/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"buka browser", "tool_conf_threshold": 0.65}'
```

- `POST http://127.0.0.1:8893/v1/reload` — Thread-safe hot-reload of `jupris_vocab.json` and `jupris_ncm_prod_ready.pt` with zero downtime.

### 4.3 Live Surgical Editing & Vocabulary Injection (`ncm_wsp_surgeon.py`)
To inject new vocabulary words (up to slot 16,384) or align specific prompt-response pairs in < 2 seconds without backpropagation (using closed-form Ridge projection `W_new = W + ΔY · Hᵀ(HHᵀ + λI)⁻¹`):

```bash
python ncm_wsp_surgeon.py "prompt baru" "target jawaban yang diinginkan"
```

This script automatically registers unknown tokens into `jupris_vocab.json`, updates `head_coarse` and `head_token` weights in-place, saves `jupris_ncm_prod_ready.pt`, and triggers `POST /v1/reload` on port 8893.

---

## 5. Next-Phase Roadmap: Teacher-Guided NCM Training

**Core Principle:** NCM remains the sole inference architecture of Jupris. External teacher models (e.g., Qwen) are used strictly offline to audit, clean, and score the training corpus—never to replace NCM with a standard autoregressive Transformer.

### 5.1 Teacher-Guided Data Curation Pipeline

```text
[Raw jupris_corpus.jsonl]
   │  (Contains valid pairs, noisy translations, duplicates, or mislabeled concept_ids)
   ▼
[Qwen Teacher + Quality Scorer (Offline Audit)]
   │  1. Verifies prompt-response coherence & natural register (ID / JV / EN / MIX)
   │  2. Audits & corrects `lang` and `concept_id` (120..220 vs 480..484 tool intents)
   │  3. Rewrites broken responses & quarantines ambiguous samples
   ▼
[Curated NCM Dataset (jupris_corpus_verified.jsonl)]
   │  (Deduplicated, schema-validated, zero train/val/test leakage)
   ▼
[NCM Multiscale VQ-VAE Tokenizer]
   │  Encodes verified target responses into discrete multiscale code hierarchies
   ▼
[NCM Next-Concept Generator]
      Learns coarse-to-fine level-wise code prediction from high-purity targets
```

### 5.2 Trilingual Data Contract (`jupris_corpus.jsonl`)
Every record in the training corpus must follow this canonical JSONL schema:

```json
{"text": "User: wis ngopi durung\nAssistant: wis ngopi bosku kopi ireng anget mantap", "lang": "jv", "concept_id": 150}
```

**Allowed `lang` values:** `id` (Indonesian), `jv` (Javanese), `en` (English), `mix` (Code-switching).

**Dataset Compiler Rules:**
- Reject malformed JSON lines with counted warnings (never silently pass errors).
- Deduplicate exact and near-duplicate pairs before splitting into train / val / test.
- Preserve technical tokens, URLs, and Javanese register distinctions.

---

## 6. Acceptance Checklist for Future Contributors

Before merging changes to `ncm_core.py` or replacing `jupris_ncm_prod_ready.pt`, verify:

- [ ] **No Hardcoded Inference Rules** — Generation and routing must come strictly from model weights (`head_coarse`, `head_token`, and router head), with zero if-else string patches inside `ncm_core.py`.
- [ ] **Sub-50ms CPU Latency** — Warm single-query latency on worker-v1 CPU must remain under 50ms.
- [ ] **Unseen Generalization Audit** — Any weight-space edit (`ncm_wsp_surgeon.py`) or fine-tuning run must pass both the 22 anchor calibration prompts and the 12 unseen generalization test prompts without semantic bleeding.
- [ ] **Closed-Loop Tool Safety** — Tool intents (480..484) must map explicitly to registered runtime functions and report truthful execution status.
