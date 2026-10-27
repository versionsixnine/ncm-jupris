# JuprisMini-NCM — Project State & Progress Log

**Snapshot date:** 2026-10-28  
**Archive purpose:** Preserve current state, decisions, tested commands, and the next task so work can resume without repeating debugging.  
**Current phase:** Full-corpus teacher curation on VPS.  
**Overall status:** Curation `IN PROGRESS`; NCM training `NOT STARTED on the completed curated corpus`.

## 1. Executive state

JuprisMini-NCM uses an NCM-based student architecture. An external Gemma/Qwen-compatible teacher is used offline to review/correct training examples; it is not the runtime chat model and does not replace NCM. The master architecture reference is `blueprint_juprismini_ncm_corrected.md` and upstream NCM (`HazyResearch/ncm`).

The active work on the VPS is to curate the source JSONL corpus, maintain a durable audit ledger, then validate the curated format before starting NCM training.

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
| Main full-corpus curation | IN PROGRESS | systemd service started successfully |
| Full curated dataset QA | NOT STARTED | Await full pass |
| NCM smoke training | NOT STARTED | Await curated dataset QA |
| Checkpoint fine-tuning/evaluation | NOT STARTED | Await smoke training and compatibility validation |

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
| `jupris_corpus_curated_v7.jsonl` | Approved-only training corpus | Final full-run result not yet confirmed |
| `tool_safety_test.jsonl` | 20-row tool-intent fixture | Test input only; not the full training set |
| `tool_safety_test_reviews_v7.jsonl` | Test audit results | Test artifact |
| `tool_safety_test_curated_v7.jsonl` | Test-only approved output | Never use as main training data |
| `dataset_run_v7.log` | Historical nohup output | Old run log, not the authoritative live service journal |
| `qwen_teacher_curator_v7.py.bak-*` | Code backups made during patching | Keep until full curation is complete and audited |
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

At the last captured status check on 2026-10-28:

- `systemctl enable --now jupris-curator-v7.service` succeeded.
- Service was `active (running)`, PID `2001092` at the captured moment.
- Startup log: `Source unique valid=26931 | invalid=0 | ledger reviewed=537 | waiting=26394`.
- The next two records then completed and were approved:
  - `kula nyuwun pangapura` — language/concept correction retained as `cid=155`, approved.
  - `why is this broken` — `cid=210`, approved.
- The third request (`how do i fix null`) had begun at the last captured follow log.

This is the last supplied log snapshot, not a live check of the service at archive time. Refresh status on the VPS before acting:

```bash
systemctl status jupris-curator-v7.service --no-pager
journalctl -u jupris-curator-v7.service -n 40 --no-pager
```

### Service lifecycle expected

When all pending source hashes have a V7 review, the script should rebuild the approved-only corpus and exit. With `Restart=on-failure`, a normal successful exit does not cause a restart. The unit remains `enabled` for reboot; a later boot may run it once more, find no pending rows, rebuild output and exit without teacher calls. If an API/transport failure stops the job, inspect logs before retrying. Do not launch a nohup copy while the systemd service is active.

## 5. Dataset and safety test history

### Main source corpus

- Unique valid source rows: `26,931`.
- Invalid rows in dry run: `0`.
- Initial full pass stopped after a teacher response with no JSON object around the early 500s. This was a format failure, not a corrupted source corpus.

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

## 8. Completion criteria for the current phase

Do not mark curation complete until all of these are true:

- [ ] Service journal contains a final `Scan complete` summary (or equivalent successful final summary from V7).
- [ ] Latest V7 review per unique source hash covers all 26,931 valid source rows, including intentional quarantines.
- [ ] The final curated corpus exists and each line is valid JSON.
- [ ] Curated count equals the number of approved latest reviews; rejected/uncertain/quarantined entries are excluded.
- [ ] Label distributions and samples of corrected rows have been reviewed, with attention to tool-label mismatches.
- [ ] Final approved/pending/quarantined totals have been recorded here.

## 9. Next tasks and exit criteria

### T1 — Finish V7 curation (IN PROGRESS)

Keep the systemd job running. Record the final summary when complete. Do not patch curator code unless a concrete failure blocks progress.

### T2 — Audit curated JSONL (NEXT)

Validate JSON on every line, required keys, duplicate hashes, count parity, label range, language labels, and a stratified sample of teacher corrections. Preserve the source and ledger.

### T3 — Ensure training directory and artifacts are present

Locate the authoritative NCM training repo and confirm `ncm_core.py`, `ncm_train.py`, `jupris_vocab.json`, and the compatible checkpoint exist there. Copy the final curated corpus only after QA, and record its checksum.

### T4 — NCM smoke test

Run `ncm_train.py --corpus ... --smoke-test --batch-size 4`. Pass criteria: dataset loads, loss is finite, gradients are finite/nonzero, optimizer path completes, and no unexpected label-range rows are silently skipped. If checkpoint resume is involved, verify compatibility separately and do not bypass strict key/shape errors.

### T5 — Fine-tune and evaluate

Train from a compatible checkpoint using a new checkpoint prefix. Keep validation/test examples separate. Record config, corpus checksum, tokenizer/vocab version, checkpoint base, epoch/step count, loss metrics, and device/runtime details.

### T6 — Compare baseline and student; then iterate

Evaluate held-out Indonesian casual/formal, Javanese ngoko/krama, technical English, code-switching, and tool-intent classification. Only then create a new corrective data cycle. Teacher scores are heuristic, and model outputs are not ground truth without review.

## 10. What is not complete

- Full 26,931-record curation: not yet confirmed complete at this snapshot.
- Final curated corpus quality audit: not started.
- NCM training using the V7 full curated corpus: not started.
- Upstream NCM architecture equivalence audit: pending.
- Held-out multilingual and tool-routing evaluation: pending.
- An iterative second training/data-correction cycle: future work.

## 11. Archive handling

The ZIP produced with this state contains docs and code snapshots for reproducibility. It excludes the full source corpus, ledgers, generated curated outputs, logs, checkpoints, and private runtime settings. The V7 script's default endpoint is sanitized to a placeholder for export; the VPS keeps its own configured endpoint. Compare exported code with the VPS copy before deploying the archive elsewhere.
