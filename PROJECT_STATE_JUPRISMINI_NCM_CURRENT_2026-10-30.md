# JuprisMini-NCM — Project State & Progress Log

**Snapshot date:** 2026-10-30  
**Archive purpose:** Preserve current state, decisions, tested commands, and the next task so work can resume without repeating debugging.  
**Current phase:** NCM scratch training on the audited V7 curated corpus.  
**Overall status:** Curation `COMPLETE`; curated-corpus count audit `PASS`; NCM smoke test `PASS`; 3-epoch scratch training `RUNNING`; held-out validation and acceptance `PENDING`.

## 1. Executive state

JuprisMini-NCM uses an NCM-based student architecture. An external Gemma/Qwen-compatible teacher is used offline to review/correct training examples; it is not the runtime chat model and does not replace NCM. The master architecture reference is `blueprint_juprismini_ncm_corrected.md` and upstream NCM (`HazyResearch/ncm`).

The V7 teacher-curation pass and independent count/parity audit are complete. The approved-only curated corpus is now being used for a fresh NCM training run on the Windows training workstation. This run has no held-out validation split and the current trainer does not save checkpoint metadata, so a falling training loss is progress evidence, not acceptance evidence.

## 2. Last verified environment checks

| Check | Result | Evidence |
|---|---|---|
| Python | PASS | Python 3.12.3 on VPS |
| Venv | PASS | `/opt/datasetjup/.venv` activated |
| Standard library imports | PASS | All imports used by V7 available |
| pip health | PASS | `pip check`: no broken requirements found |
| Curator syntax | PASS | `python -m py_compile qwen_teacher_curator_v7.py` |
| CLI | PASS | `python qwen_teacher_curator_v7.py --help` |
| Source dry run | PASS | 26,931 unique valid rows; 0 invalid |
| Ollama connectivity | PASS | `/api/tags` reachable; configured model found |
| Real teacher smoke test | PASS | One record returned a valid review and was written to its smoke ledger |
| V7 tool-safety subset | PASS with quarantine | 20 reviewed; 15 approved; 5 quarantined |
| Main full-corpus curation | PASS / COMPLETE | V7 final summary and independent audit: 26,931 reviewed; 21,180 approved; 5,751 excluded; 0 unreviewed |
| Curated dataset count/parity audit | PASS | 21,180 curated rows; training loader parsed all rows, skipped 0 |
| NCM smoke training | PASS | One-batch forward/backward/optimizer step; finite losses; CUDA |
| NCM scratch training | RUNNING | 3 epochs, batch size 8; latest supplied log E1 B1100 |
| Held-out validation / acceptance | PENDING | No validation split in current run |
| Checkpoint metadata | GAP | Current trainer does not persist complete run metadata |

`PASS` means the specific check above ran successfully. It does not imply that the whole model or training pipeline is complete.

## 3. VPS paths and roles

**Working directory:** `/opt/datasetjup`  
**Venv:** `/opt/datasetjup/.venv`  
**Systemd unit:** `/etc/systemd/system/jupris-curator-v7.service`

| Path | Purpose | Handling |
|---|---|---|
| `jupris_corpus.jsonl` | 26,931-record source corpus | Preserve; curator reads it without rewriting |
| `qwen_teacher_curator_v7.py` | Active curator code (filename retained) | Production snapshot on VPS; patched and syntax/CLI checked |
| `jupris_teacher_reviews_v7.jsonl` | V7 audit ledger | Preserve; append-only; do not delete |
| `jupris_corpus_curated_v7.jsonl` | Approved-only training corpus | Full run complete; 21,180 approved rows; Windows training copy used below |
| `tool_safety_test.jsonl` | 20-row tool-intent fixture | Test input only; not the full training set |
| `tool_safety_test_reviews_v7.jsonl` | Test audit results | Test artifact |
| `tool_safety_test_curated_v7.jsonl` | Test-only approved output | Never use as main training data |
| `dataset_run_v7.log` | Historical nohup output | Old run log, not the authoritative live service journal |
| `qwen_teacher_curator_v7.py.bak-*` | Code backups made during patching | Historical recovery copies; retain in the archive unless deliberately cleaned up |
| `patch_v7_json_retry.py` | Patch utility used on VPS | Historical patch utility; don't rerun without review |

Training-side files (`ncm_core.py`, `ncm_train.py`, `ncm_cli.py`, `jupris_vocab.json`, checkpoint) were present in the conversation's project snapshots but were **not** in the last shown `ls` for `/opt/datasetjup`. Do not assume that the VPS curation directory is the NCM training directory.

## 4. Current service operation

Unit file: `/etc/systemd/system/jupris-curator-v7.service`

Important settings:

- `WorkingDirectory=/opt/datasetjup`
- `ExecStart=/opt/datasetjup/.venv/bin/python -u /opt/datasetjup/qwen_teacher_curator_v7.py`
- `Restart=on-failure`
- `RestartSec=60`
- No `--watch` flag: this is a one-pass job, not a permanent polling loop.

V7 completed successfully on 2026-10-30. Final service journal summary supplied:

```text
[635/635] A=95/keep L=100/keep approved=True cid=125 | hai jupris sapaan formal
[+] Scan complete: reviewed=635 | approved unique rows=21180 | pending/invalid=5751
systemd: Deactivated successfully
```

The `635/635` counter represents the last resumed processing segment, not the full source-corpus size. The independent audit below verifies full coverage. The service exited successfully. It had previously been enabled with `systemctl enable --now`; a later recommendation to disable autostart was made, but no command output confirms that it was actually disabled. Current enablement state is therefore unverified.

For a fresh VPS status check:

```bash
systemctl status jupris-curator-v7.service --no-pager
journalctl -u jupris-curator-v7.service -n 40 --no-pager
```

### Service lifecycle expected

When all pending source hashes have a V7 review, the script should rebuild the approved-only corpus and exit. With `Restart=on-failure`, a normal successful exit does not cause a restart. The unit was previously configured as `enabled`; its current enablement state is unverified. If it remains enabled, a later boot may run it once more, find no pending rows, rebuild output and exit without teacher calls. If an API/transport failure stops the job, inspect logs before retrying. Do not launch a nohup copy while the systemd service is active.

## 5. Dataset and safety test history

### Main source corpus and final audit

- Unique valid source rows: `26,931`; invalid source rows in dry run: `0`.
- Initial full pass stopped after a malformed teacher response around the early 500s. V7 retry/quarantine hardening allowed the resumed pass to complete.
- Final independent audit (Windows): source unique valid `26,931`; unique reviews `26,931`; approved reviews `21,180`; excluded reviews `5,751`; still unreviewed `0`; curated file rows `21,180`.
- Approval rate: `78.64%`; excluded rate: `21.36%`.
- Training loader later reported `Dataset: 21180 examples | skipped: 0`, consistent with the curated row count.

Final verdict distribution from the audit:

```python
{
    'keep': 16083,
    'needs_review': 4438,
    'correct_label': 2257,
    'correct_both': 2736,
    'uncertain': 1313,
    'correct_answer': 104,
}
```

### V7 hardening applied on the VPS

- Malformed/empty teacher JSON response is retried up to three attempts for the same record.
- If format remains invalid after retries, that record is written to the ledger as a quarantine (`approved=false`, `selected=null`, `quarantined_teacher_format=true`) and the loop continues.
- Transport/API failures remain distinct; they can stop the scan so the endpoint is not hammered continuously.
- After patching, V7 passed Python compilation and CLI `--help` checks.
- A one-line quarantine progress counter typo (`index` to `done`) was corrected directly on the VPS; a backup was kept.
- This exported curator copy has the corresponding counter correction and a placeholder endpoint, not the VPS's private/default endpoint configuration.

### Tool-safety fixture (20 records)

- Result: 15 approved, 5 quarantined.
- The 5 quarantined rows included firmware-flash and restart/watchdog intents that don't map cleanly to the current documented labels. `484` means repository inspection; `483` means system/resource status; `482` means Python execution.
- Do not force these rows into the approved corpus merely to increase approval rate. If project requirements need a distinct firmware/restart tool intent, document and implement a label schema change first.

## 6. Change history / lessons recorded

| Stage | Change/learning | State |
|---|---|---|
| V4 | Initial audit ledger, score/verdict validation, tool-result safety heuristic; 20-row test exposed over-rejection of tool examples | Historical |
| V5 | Added deterministic safe acknowledgements for unsupported tool-result claims; tool subset yielded 15/20 approved | Historical |
| V6 | Label validation and safety logic refined; the live 20-row output remained 15/20 approved, 5 quarantined | Historical |
| V7 | Output paths/versioned ledger, visible per-record status, malformed JSON retry, persistent format quarantine, resumable hash-based processing | Active |
| VPS systemd | Replaced manual `nohup` management with enabled one-shot systemd service; avoids losing control of job when SSH disconnects | Active |

A repeated invalid JSON response should no longer stop the entire corpus loop; it should quarantine that specific row after retry. Verify future logs actually show the expected `QUARANTINED teacher-format error after retries` line if this happens.

## 7. Commands for routine monitoring

```bash
# Live status
systemctl status jupris-curator-v7.service --no-pager

# Follow live logs; Ctrl+C exits the viewer, not the service
journalctl -u jupris-curator-v7.service -f

# Recent logs
journalctl -u jupris-curator-v7.service -n 80 --no-pager

# Ledger line count (not identical to unique reviewed rows due to append-only history)
wc -l /opt/datasetjup/jupris_teacher_reviews_v7.jsonl

# Check service process
systemctl show jupris-curator-v7.service -p ActiveState -p SubState -p MainPID
```

Do not run `qwen_teacher_curator_v7.py` manually while the service is active. Do not use `--recheck` without a deliberate plan because it asks for reviews again.

## 8. Completion criteria for curation

The full-corpus curation/count-parity phase is complete based on the final service summary and independent audit:

- [x] V7 journal contains the final `Scan complete` summary.
- [x] All 26,931 unique source rows have corresponding unique reviews; 0 remain unreviewed.
- [x] Curated output contains 21,180 rows; training loader parsed 21,180 and skipped 0.
- [x] Approved count matches curated output; 5,751 non-approved reviews were excluded.
- [x] Final verdict distribution and counts are recorded in Section 5.
- [ ] A deeper qualitative/stratified review of corrected samples and language/tool-label distributions remains pending. Count parity does not prove semantic quality.

## 9. NCM training state and next tasks

### T1 — V7 curation and count/parity audit (COMPLETE)

Main curated file: `jupris_corpus_curated_v7.jsonl`, 21,180 approved rows. Full-corpus coverage and row-count parity are verified; qualitative sample review remains a separate task.

### T2 — NCM smoke test (PASS)

Command:

```powershell
python ncm_train.py --corpus jupris_corpus_curated_v7.jsonl --smoke-test --batch-size 4
```

Supplied output: `Dataset: 21180 examples | skipped: 0 | vocab: 10330 | device=cuda`; one-batch forward/backward/optimizer step passed with finite values (`loss_total=51.9256`, `loss_levels=48.6806`, `loss_commit=0.1012`, `loss_router=6.2876`). Smoke test does not save or overwrite checkpoints. Flash-attention warning is nonfatal.

### T3 — Old checkpoint incompatibility (DECISION: TRAIN FROM SCRATCH)

Attempting `--resume .\\jupris_ncm_core_v2.pt` failed with `strict=True` due to missing/unexpected keys and tensor-shape mismatches across the quantizer hierarchy and generator stages. This is an architecture incompatibility, not a curated-dataset failure. Do not use `strict=False` to force-load the old checkpoint. The active run starts from a fresh initialization; it is not fine-tuning that old checkpoint.

### T4 — Train on curated V7 (RUNNING)

Command used on the Windows workstation:

```powershell
python .\ncm_train.py `
  --corpus .\jupris_corpus_curated_v7.jsonl `
  --checkpoint-prefix jupris_ncm_curated_v7_scratch `
  --epochs 3 `
  --batch-size 8
```

Latest supplied log: epoch 1, batch 1,100; dataset 21,180; skipped 0; vocab size 10,330; device CUDA. At batch size 8, there are approximately 2,648 batches per epoch (the final batch may be smaller), so E1 B1100 is approximately 41.6% through epoch 1.

Loss samples from the supplied log:

| Point | Total | VQ levels | Commitment | Router |
|---|---:|---:|---:|---:|
| E1 B0 | 52.5110 | 49.3133 | 0.0880 | 6.2192 |
| E1 B100 | 25.0490 | 21.9166 | 0.0469 | 6.1710 |
| E1 B500 | 19.6477 | 17.8878 | 0.0425 | 3.4347 |
| E1 B900 | 11.2720 | 10.1130 | 0.0393 | 2.2393 |
| E1 B1100 | 13.4795 | 12.6768 | 0.0393 | 1.5269 |

These samples show a substantial downward trend with batch-to-batch fluctuation. They are training metrics only and do not establish held-out generalization. The warning `Torch was not compiled with flash attention` is nonfatal; training continued.

**Do not stop the active run to retrofit validation or metadata.** Let the requested three epochs complete. Based on the inspected `ncm_train.py`, there is no validation split and checkpoints are saved only after all epochs complete. Expected output filenames are `jupris_ncm_curated_v7_scratch_v2.pt` and `jupris_ncm_curated_v7_scratch.pt`; their actual creation has not yet been confirmed.

### T5 — Post-training evaluation and acceptance (PENDING)

This run has no held-out validation split and no complete checkpoint metadata record. The eventual checkpoint may be used for controlled CLI/inference experiments, but must not be marked accepted or `done` based only on falling training loss.

After the run completes:

1. Confirm checkpoint files exist; record file sizes and SHA-256 hashes; preserve the full training log.
2. Evaluate against data held out from training and report validation loss plus per-task metrics. Since this run had no validation split, label any post-hoc held-out evaluation accurately; do not call it in-run validation.
3. Record model/config parameters, corpus checksum, vocabulary/tokenizer version, code revision, seed, optimizer/LR schedule, device/runtime, epoch/step count, and checkpoint hashes in a metadata sidecar.
4. Evaluate Indonesian casual/formal, Javanese ngoko/krama, technical English, code-switching, and tool-intent classification. Assess routing separately from response quality.
5. Compare results to the blueprint's acceptance criteria before declaring the checkpoint accepted.

### T6 — Architecture and runtime follow-up (PENDING)

The compact implementation still needs a source-level equivalence audit against upstream `HazyResearch/ncm`. Also revisit explicit tool-ID mappings and replace mock tool execution before treating runtime tool integration as production-ready. Teacher scores are heuristic; model outputs are not ground truth without review.

## 10. What is not complete

- Qualitative/stratified semantic review of curated V7 rows and corrected labels.
- Completion of the current 3-epoch scratch training run; latest supplied progress is E1 B1100.
- Held-out validation loss and multilingual/tool-routing evaluation.
- Checkpoint metadata sidecar and reproducibility record.
- Upstream NCM architecture equivalence audit.
- Explicit tool-ID mapping and production tool execution (current mock executor is not production-ready).
- An iterative second training/data-correction cycle.

## 11. Archive handling

The ZIP produced with this state contains docs and code snapshots for reproducibility. It excludes the full source corpus, ledgers, generated curated outputs, logs, checkpoints, and private runtime settings. The V7 script's default endpoint is sanitized to a placeholder for export; the VPS keeps its own configured endpoint. Compare exported code with the VPS copy before deploying the archive elsewhere.
