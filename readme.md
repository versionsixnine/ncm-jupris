# JuprisMini-NCM — Corrected Master Blueprint
**Status:** Architecture baseline for implementation  
**Primary method reference:** https://github.com/HazyResearch/ncm  
**Scope:** Small edge-oriented trilingual chat model with explicit tool integration.

## 1. Goal — unchanged

Build **JuprisMini-NCM**, a compact model intended to:
1. Chat simply and coherently in Indonesian (casual/formal), Javanese (ngoko/krama where supported by data), and technical English.
2. Handle mixed-language/code-switching naturally.
3. Select and invoke registered tools when a task requires them.
4. Run within constrained local hardware where practical.

The model is not expected to become a general-purpose frontier LLM merely by adding an NCM architecture. Quality depends on tokenizer/model design, training data, compute, evaluation, and tool implementations.

## 2. Architecture rule: upstream NCM is the method baseline

Use the method in the HazyResearch NCM repository as the architectural source of truth. Reuse only the relevant implementation and concepts; do not copy or download the whole repository wholesale.

The upstream method is a two-part system:
- **Multiscale residual VQ tokenizer (VQ-VAE):** encodes a token sequence into a hierarchy of discrete codes at multiple scales.
- **Next-concept generator:** predicts the code hierarchy from coarse to fine. Each level predicts its positions in parallel; generation proceeds level-by-level, not token-by-token in a conventional autoregressive loop.

Reference files to study before implementation:
- `models/vqvae.py`
- `models/next_concept.py`
- `models/quant.py`
- `models/common.py`
- `tokenization/vqvae_tokenizer.py`
- relevant `config/` files

Reference: https://github.com/HazyResearch/ncm

### Required distinction

Jupris is an adaptation for a smaller trilingual chat/tool use case. It must preserve the **NCM method** (multiscale residual quantization + coarse-to-fine level-wise generation), while any smaller dimensions, fewer levels, or altered context length must be documented as deliberate engineering adaptations and tested.

Do not describe a custom two-codebook RVQ with `head_coarse`, `head_fine`, and `head_token` as the upstream NCM architecture. That design is not the method implemented by the reference repository.

## 3. Remove unsupported / hallucinated design claims

The following rules in the previous blueprint are **not valid as mandatory NCM architecture** and must be removed unless separately designed, implemented, and tested:

- A fixed, exactly two-stage `codebook_coarse` / `codebook_fine` RVQ as the definition of NCM.
- The claim that upstream NCM requires `head_coarse`, `head_fine`, and `head_token`.
- The claim that `concept_id >= 480` inherently means a tool call or can safely short-circuit generation in one pass.
- The claim that a fine code can be trained by the stated cross-entropy objective as a canonical NCM loss.
- The hard-coded assertion that a tool call can be inferred by subtracting `480` from a predicted concept ID.
- The claim that one training-loss formula or a fixed level list is universally canonical for every NCM configuration.
- Any mock tool output presented as successful execution.
- Any statement such as “zero latency” or guaranteed “one forward pass” without measured evidence.

Do not replace these with another invented architecture. Verify implementation details against the upstream source and record deviations explicitly.

## 4. Jupris-specific semantics: concepts are labels, not the NCM codebook

Preserve these dataset-level semantic labels as metadata for supervision, filtering, and evaluation. They are **not** NCM VQ code indices and must not be confused with the tokenizer's learned codebooks.

| Concept ID | Intended label |
|---:|---|
| 120 | Indonesian casual/slang |
| 125 | Indonesian formal/technical |
| 150 | Javanese ngoko |
| 155 | Javanese krama/polite |
| 210 | Technical English |
| 220 | Trilingual/code-switching |
| 480 | Tool intent: web search |
| 481 | Tool intent: hardware/GPIO telemetry |
| 482 | Tool intent: Python execution |
| 483 | Tool intent: system status |
| 484 | Tool intent: repository inspection |

These IDs are project-defined labels, not special behavior supplied by NCM. The training code must validate labels against the task's label schema. Tool IDs must not be treated as a continuous range where every ID at or above 480 is automatically executable.

## 5. Tooling architecture

Tool execution is a separate, explicit control layer around the NCM text model.

1. **Input:** accept the user message and preserve its original language.
2. **Intent decision:** use a trained tool-intent classifier/router or a clearly defined policy layer. Do not use a VQ code index as a tool ID.
3. **Validation:** map the selected intent to a registered tool by exact name/schema. Validate arguments and permissions before execution.
4. **Execution:** invoke the actual configured tool. If the tool is unavailable, fails, or is only mocked, report that accurately.
5. **Result conditioning:** append the actual tool result to the conversation context.
6. **Response synthesis:** use the NCM generator to produce the user-facing response based on the original request and tool result.
7. **Trace:** log selected intent, tool name, validated arguments, execution status, and errors without leaking secrets.

Initial tool intent mapping may use the project labels above, but mapping must be explicit, e.g. `480 -> search_web`; never calculate a tool ID by subtracting a threshold. Tool calls must not be executed solely because an unvalidated model output happens to fall in a numeric range.

Tools are capabilities supplied by Jupris' runtime, not capabilities automatically created by NCM.

## 6. Trilingual data contract

Keep `jupris_corpus.jsonl` as the corpus filename. Each line is one JSON object with this minimum schema:

```json
{"text":"User: ...\nAssistant: ...","lang":"id","concept_id":120}
```

Allowed `lang` values should be normalized and documented, for example:
- `id` — Indonesian
- `jv` — Javanese
- `en` — English
- `mix` — mixed-language/code-switching

`concept_id` is an optional project task label when a sample has a reliable label. Do not fabricate a label when it is unknown; either omit it or use an explicitly documented unknown label supported by the implementation.

Dataset compiler requirements:
- Accept the existing `text` format and, if required, a normalized `{ "user": ..., "assistant": ..., "lang": ..., "concept_id": ... }` source format.
- Convert both source formats to one canonical internal record.
- Preserve language and Javanese register metadata where available.
- Reject malformed JSON with a counted warning; do not silently swallow all exceptions.
- Deduplicate and split train/validation/test without leakage.
- Avoid lowercasing or stripping punctuation blindly if it damages names, code, URLs, technical terms, or language distinctions.
- Keep the existing `jupris_vocab.json` only if the selected tokenizer is actually this project vocabulary. If adopting the upstream tokenizer/BPE path, define and version the tokenizer artifact explicitly; do not claim that a word-level JSON vocabulary is interchangeable with it.

## 7. Implementation layout and naming

Preserve these project-facing filenames where possible:
- `blueprint_juprismini_ncm.md`
- `jupris_corpus.jsonl`
- `jupris_vocab.json` (only for the matching tokenizer implementation)
- `ncm_core.py`
- `ncm_train.py`
- `ncm_cli.py`
- `infer_ncm.py`
- `jupris_ncm_core_v2.pt`
- `jupris_ncm_core.pt` (legacy compatibility only, if architecture and metadata are compatible)

Required engineering rules:
- One authoritative model implementation; no duplicated class definitions that can drift.
- Checkpoint must include architecture/config metadata and tokenizer identity/version.
- Loading a checkpoint must validate compatibility and fail clearly on mismatch.
- Keep `fast_reflect_action` only if its behavior is implemented and tested against the final architecture. Its name alone does not establish a one-pass capability.
- Remove or rename APIs that are no longer implemented; do not leave dead methods or heads.
- No dummy/no-op quantizer, fake generated response, fake successful tool result, or silently swallowed training/data error.

## 8. Training plan

Do not copy the previous blueprint's fixed loss formula as “the NCM loss.” Derive training objectives from the selected upstream NCM components and configuration.

Minimum stages:
1. Verify the tokenizer's encode/decode and multiscale code shapes.
2. Verify the VQ-VAE/tokenizer reconstruction objective and codebook/commitment terms against the upstream implementation.
3. Verify the generator predicts the correct target codes at every configured level, using the upstream level-wise objective/masking strategy.
4. Add Jupris trilingual/chat fine-tuning with a documented data mixture and validation split.
5. Train and evaluate tool-intent routing separately or as an explicitly designed auxiliary objective; it must not corrupt the NCM VQ code semantics.
6. Save model, tokenizer, configuration, label schema, and training metadata together.

Do not assume a short “one verification epoch” proves model quality. It can verify that the training path runs, losses are finite, gradients exist, and a checkpoint reloads. It is not evidence of useful chat quality.

## 9. Inference flow

1. Load model checkpoint, tokenizer, model config, and label schema.
2. Validate all artifacts are mutually compatible.
3. Receive user input and preserve language/register cues.
4. Decide whether a registered tool is required using the explicit router/policy.
5. If no tool is required, generate with the NCM coarse-to-fine generator and decode using the matching tokenizer.
6. If a tool is required, validate and execute it, then condition response generation on the actual tool result.
7. Display the final response and a truthful tool execution status.
8. Log latency and errors for evaluation; do not promise zero latency.

## 10. Acceptance tests — required before calling it done

- **Architecture:** code path uses multiscale residual VQ and coarse-to-fine level-wise generation from the chosen NCM implementation or a documented, tested adaptation.
- **No fake components:** no no-op quantizer, dead heads, canned tool-success outputs, or silently ignored exceptions.
- **Shapes:** tokenizer code shapes and generator logits match each configured level.
- **Round trip:** tokenizer encode/decode smoke test passes.
- **Training:** one batch has finite loss, nonzero gradients on intended trainable parameters, and optimizer update changes parameters.
- **Checkpoint:** save/reload produces compatible outputs with matching config/tokenizer metadata.
- **Trilingual:** held-out Indonesian, Javanese, English, and mixed-language tests are reported separately.
- **Tooling:** each tool intent maps explicitly to a registered tool; invalid arguments, missing tools, and tool failures are handled safely.
- **Closed loop:** actual tool output is included in response context; no canned result is substituted.
- **Evaluation:** report validation loss plus task-specific chat, language, and tool-routing metrics. Do not report training loss alone as proof of success.

## 11. Source-of-truth hierarchy

When instructions conflict, use this order:

1. **Upstream NCM repository** for the NCM method and reference implementation.
2. **This corrected blueprint** for Jupris' goal, constraints, trilingual behavior, and tooling requirements.
3. **Current Jupris code** only where it passes the checks above.
4. Old blueprint snippets and prior generated code are not authoritative when they conflict with the upstream method.

No full-repository fork/download is required. Inspect the relevant source files, port only the necessary components, document every adaptation, and test the resulting implementation.

--
step for now : 
Kita harus menjaga satu prinsip: NCM tetap menjadi metode inti Jupris. Qwen hanya memasok supervisi untuk memperbaiki data dan target pembelajaran—bukan menggantikan NCM dengan Transformer autoregresif biasa.
--
1. Gunakan teacher-guided NCM training, bukan sekadar menambah data
Dataset Jupris yang sekarang

Contoh benar, salah, duplikat, dan label yang mungkin keliru

Qwen Teacher + Quality Scorer

Periksa pasangan, label bahasa, kesesuaian jawaban, dan maksud tool

Curated NCM training set

Positif terverifikasi + koreksi + contoh ambigu yang dikarantina

NCM tokenizer / VQ-VAE

Encode target teks menjadi hierarki discrete codes

NCM Next-Concept Generator

Belajar memprediksi kode coarse-to-fine dari contoh yang sudah diperbaiki
