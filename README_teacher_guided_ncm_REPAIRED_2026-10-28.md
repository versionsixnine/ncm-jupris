# JuprisMini-NCM — Teacher-Guided Iterative Training

**Documentation snapshot:** 2026-10-28  
**Current phase:** Full-corpus teacher curation is running on the VPS.  
**Student model:** JuprisMini-NCM.  
**Teacher role:** Offline dataset reviewer/corrector through an Ollama-compatible API; not the runtime chat generator.

## 1. Project goal

Build a compact edge-oriented trilingual conversational model for Indonesian, Javanese, technical English, natural code-switching, and explicit tool-intent routing. The NCM model remains the student and the core generative method. Gemma/Qwen supplies offline supervision and quality review; it does not replace NCM.

The longer-term process is **Teacher-Guided Iterative Training**:

1. Review and curate the source corpus with an external teacher.
2. Train or fine-tune the compatible NCM student using the approved corpus.
3. Evaluate on held-out language, answer-quality, and tool-routing tasks.
4. Inspect failures; add corrected examples to a new curation cycle without rewriting the original corpus.
5. Retrain from a compatible checkpoint and compare results against the previous baseline.

The first full curation run is not complete yet. Do not describe the model as trained or evaluated until those steps have actually passed.

## 2. Method and architecture guardrails

The architecture baseline is the [HazyResearch NCM repository](https://github.com/HazyResearch/ncm). Preserve the method's key idea:

- Multiscale residual vector quantization / VQ-VAE tokenizer.
- Next-concept generation from coarse to fine, predicting positions level-wise.
- Document deliberate reductions/adaptations (dimensions, levels, sequence length) and test them.

Do not present a generic autoregressive Transformer, a two-codebook design, or a hand-written router as equivalent to upstream NCM without source verification and tests. Project `concept_id` values are semantic dataset labels; they are not NCM VQ codebook indices.

### Project concept labels

| `concept_id` | Project meaning |
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
| 483 | Tool intent: system/resource status |
| 484 | Tool intent: repository inspection |

These meanings must be enforced by explicit validation, not numeric-range guesses. In the current safety trial, firmware-flash and restart/watchdog examples were quarantined because their intent did not match the current label definitions. Reconcile label definitions deliberately before retraining; do not silently relabel them.

## 3. Components and file roles

| File | Role | State at snapshot |
|---|---|---|
| `jupris_corpus.jsonl` | Original source corpus; input to curator | Present on VPS; 26,931 unique valid records, 0 invalid in the dry run |
| `qwen_teacher_curator_v7.py` | V7 teacher curator (filename retained for compatibility; currently configured model is Gemma) | Active runtime script on VPS; malformed JSON retry/quarantine patch applied |
| `jupris_teacher_reviews_v7.jsonl` | Append-only teacher review ledger/audit trail | In progress; do not delete |
| `jupris_corpus_curated_v7.jsonl` | Approved-only output corpus built by the curator | Output target; final full corpus is not yet confirmed |
| `tool_safety_test.jsonl` | Small tool-intent safety fixture | Present on VPS; 20-record test source |
| `tool_safety_test_reviews_v7.jsonl` | Safety-test ledger | Trial result: 15 approved, 5 quarantined |
| `tool_safety_test_curated_v7.jsonl` | Safety-test approved-only corpus | Trial output, not a training corpus |
| `ncm_core.py` | NCM student model implementation | Training-side source snapshot; audit against upstream NCM remains required |
| `ncm_train.py` | Student training entry point (`--corpus`, `--resume`, `--checkpoint-prefix`, `--smoke-test`) | Training not yet run on the completed V7 corpus |
| `ncm_cli.py` | NCM CLI helper | Training-side source snapshot |
| `blueprint_juprismini_ncm_corrected.md` | Corrected architecture and project blueprint | Design baseline |
| `jupris_vocab.json` | Token-to-ID vocabulary, if compatible with the active tokenizer | Must be present and validated in the actual training directory |
| `jupris_ncm_core_v2.pt` | Candidate resumable checkpoint | Compatibility must be checked before resume |

The current `/opt/datasetjup` VPS directory is the data-curation work area. The last directory listing did not show `ncm_core.py`, `ncm_train.py`, `ncm_cli.py`, `jupris_vocab.json`, or a checkpoint there. Training must be run from the actual NCM project directory, with a verified copy/path for the completed curated corpus.

## 4. VPS deployment snapshot

**Data-curation directory:** `/opt/datasetjup`  
**Python:** 3.12.3  
**Virtual environment:** `/opt/datasetjup/.venv`  
**Service unit:** `/etc/systemd/system/jupris-curator-v7.service`  
**Service name:** `jupris-curator-v7.service`  
**Configured model:** `huihui_ai/gemma-4-abliterated:26b`  
**Endpoint:** configured on the VPS; do not publish endpoint credentials or private network details in public archives.

The environment checks reported PASS for Python version, required standard-library imports, `pip check`, V7 syntax compilation, CLI `--help`, source dry-run, and reachability of the configured Ollama `/api/tags` endpoint with the configured model present. A real one-record smoke test succeeded. These checks verify the curator path, not training quality.

### Systemd behavior

The service was confirmed `enabled` and `active (running)` on 2026-10-28. It uses `Restart=on-failure`, a 60-second restart delay, and the V7 script without `--watch`.

- While curation has pending records, it sends sequential teacher requests and appends review rows.
- A malformed teacher JSON response is retried up to three attempts. If it remains malformed, that source record is quarantined and the run continues.
- Transport/API errors still stop the run; inspect the journal before restarting.
- On a successful full pass, the process exits. Systemd should then show `inactive (dead)` while remaining `enabled`; it does not keep making teacher requests after successful completion.
- After a later VPS reboot, an enabled one-shot run may start again, inspect the ledger, rebuild the output if there is nothing pending, then exit without teacher requests.

Commands:

```bash
# Status
systemctl status jupris-curator-v7.service --no-pager

# Follow the run
journalctl -u jupris-curator-v7.service -f

# Recent service output
journalctl -u jupris-curator-v7.service -n 80 --no-pager

# Review ledger line count (line count may exceed unique reviewed hashes if retries were recorded)
wc -l /opt/datasetjup/jupris_teacher_reviews_v7.jsonl
```

Do not start a second curator process while the service is active. Preserve the ledger and the original source corpus.

## 5. Current progress and verified outcomes

See [`PROJECT_STATE_JUPRISMINI_NCM.md`](PROJECT_STATE_JUPRISMINI_NCM.md) for timestamped progress and the next-stage checklist.

At the last captured service logs on 2026-10-28:

- `jupris_corpus.jsonl`: 26,931 unique valid records, 0 invalid.
- Service startup reported 537 previously reviewed unique records and 26,394 waiting.
- The first two records after that startup returned approved results; the third request had begun. This is a historical snapshot, not a live status query.
- Earlier, a teacher response without a JSON object stopped the old run around record 517. V7 was patched to retry malformed format responses and quarantine persistent format errors; the code passed `py_compile` and CLI checks after the patch.
- Tool safety fixture: 15/20 approved and 5 quarantined. Quarantined examples were not forced into the training set.
- A preliminary latency estimate based on only a few requests was approximately one minute per record. It is too small a sample to promise an end time; the full pass may take many days.

## 6. Curator operations

Defaults in V7:

- Source: `jupris_corpus.jsonl`
- Ledger: `jupris_teacher_reviews_v7.jsonl`
- Curated output: `jupris_corpus_curated_v7.jsonl`
- Minimum answer score: 80
- Minimum label score: 80
- No `--watch` in the systemd service

Do not use the partial smoke-test output for training. When the current run is done, confirm the final `Scan complete` line and audit the latest ledger entry per source hash (the ledger is append-only and can have multiple rows per hash over time).

The V7 patch was applied directly on the VPS, and a backup was created there. The code copy in this export is a portable snapshot: its default Ollama URL is intentionally replaced with a placeholder. The live VPS has its own configured endpoint.

## 7. Next steps — gate before training

### Step A — Finish curation (IN PROGRESS)

- Keep `jupris-curator-v7.service` running; do not launch a duplicate process.
- Monitor the journal for request progress, JSON retries, quarantines, or transport errors.
- Wait for the final `Scan complete` message.
- Confirm the unique V7 ledger coverage against the 26,931 unique source rows. Count latest review per `source_hash`, not raw ledger lines.
- Record final approved, rejected/uncertain/quarantined, and pending totals in the project state file.

### Step B — Audit curated data (NOT STARTED)

- Validate every line is parseable JSON and the expected fields (`text`, `lang`, `concept_id`) are present.
- Confirm no duplicate source hashes and that quarantined/uncertain records are excluded.
- Review label distributions and a stratified sample of corrections, especially tool intents and Javanese register.
- Preserve source corpus and ledger; do not overwrite them.

### Step C — Validate training inputs (NOT STARTED)

- Ensure `ncm_core.py`, `ncm_train.py`, `jupris_vocab.json`, and a candidate checkpoint exist in the actual training directory.
- Inspect `ncm_train.py` dataset parsing and vocabulary behavior. Confirm curated output format is supported and check unknown-token rate.
- Check `NUM_CONCEPTS` and label ranges; the training loader must support all approved `concept_id` values.
- Do not silently bypass checkpoint shape/key mismatches.

### Step D — NCM smoke training (NOT STARTED)

Run in the NCM training directory after the completed curated corpus is made available there:

```bash
python ncm_train.py \
  --corpus /path/to/jupris_corpus_curated_v7.jsonl \
  --smoke-test \
  --batch-size 4
```

A smoke test must prove that the data loader works, loss is finite, gradients are nonzero, and the code path completes. It is not evidence of useful language quality.

### Step E — Resume a compatible checkpoint and evaluate (NOT STARTED)

After the smoke test and checkpoint compatibility checks:

```bash
python ncm_train.py \
  --corpus /path/to/jupris_corpus_curated_v7.jsonl \
  --resume /path/to/jupris_ncm_core_v2.pt \
  --checkpoint-prefix jupris_ncm_teacher_v1 \
  --epochs 3
```

Use a new checkpoint prefix. Evaluate separately on held-out Indonesian, Javanese ngoko/krama, English, mixed-language dialogue, and tool-intent routing. Keep a validation/test split separate from the curated training set.

### Step F — Iterative improvement (FUTURE)

Collect student failures, have the teacher review proposed repairs, human-review high-impact or ambiguous cases, curate into a new versioned dataset, train a new checkpoint, and compare results to the prior baseline. Do not feed model-generated answers back as ground truth without review.

## 8. Safety and data integrity

- The original `jupris_corpus.jsonl` is read-only for the curator.
- The review ledger is the audit trail; do not delete it to “clean up” a run.
- Teacher scores are heuristics, not ground truth.
- Tool intent is not proof that a tool ran. The curator never executes tools and must not create fake sensor readings, search results, system status, or successful execution claims.
- `reject`, `uncertain`, and quarantined examples are excluded from approved training rows.
- `concept_id` labels are project metadata, not NCM VQ code indices.
- Do not claim full equivalence to upstream NCM until the code is audited and deviations are documented.
- Do not include the full corpus, private ledgers, logs, credentials, or private endpoint details in a public code archive by default.

## 9. Archive contents

The companion ZIP includes documentation, a portable V7 curator code snapshot, the corrected blueprint, snapshots of the NCM training source files, a sample systemd unit, and an archive manifest. It intentionally excludes the full corpus, review ledgers, generated curated corpora, runtime logs, checkpoints, and secrets.
