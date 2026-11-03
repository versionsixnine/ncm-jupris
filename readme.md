# JuprisMini-NCM — Trilingual Multiscale Concept Model & Tool Router

**Status:** Production Ready (`v6-wsp` / CPU Inference ~33–45ms)  
**Primary Method Reference:** [HazyResearch/ncm](https://github.com/HazyResearch/ncm)  
**Scope:** Lightweight edge-oriented trilingual chat model (Indonesian, Javanese, English) with decoupled concept/tool routing and zero-gradient weight-space surgical editing.

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
| `120` | `TEXT_STREAM` | `null` | Indonesian casual / slang[cite: 2] |
| `125` | `TEXT_STREAM` | `null` | Indonesian formal / technical[cite: 2] |
| `150` | `TEXT_STREAM` | `null` | Javanese ngoko[cite: 2] |
| `155` | `TEXT_STREAM` | `null` | Javanese krama / polite[cite: 2] |
| `210` | `TEXT_STREAM` | `null` | Technical English[cite: 2] |
| `220` | `TEXT_STREAM` | `null` | Trilingual / code-switching / translation[cite: 2] |
| `480` | `TOOL_CALL` | `search_web` | Web search, browser, maps, location & routing[cite: 2] |
| `481` | `TOOL_CALL` | `read_gpio` | Hardware / GPIO pin activation & telemetry[cite: 2] |
| `482` | `TOOL_CALL` | `exec_python` | Sandboxed Python execution[cite: 2] |
| `483` | `TOOL_CALL` | `check_server` | Edge / VPS server health & status telemetry[cite: 2] |
| `484` | `TOOL_CALL` | `inspect_repo` | Repository & code inspection[cite: 2] |

> **Strict Guardrail:** Never infer a tool ID by subtracting `480` from a VQ codebook index[cite: 2]. Tool routing is handled by the dedicated router head (`model.route()`) with explicit schema mapping and confidence thresholding (`tool_conf_threshold = 0.65`)[cite: 2].

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
