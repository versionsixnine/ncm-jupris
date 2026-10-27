#!/usr/bin/env python3
"""
JuprisMini-NCM Teacher Curator V7
Gemma/Qwen teacher via Ollama-compatible API.

The teacher curates dataset examples only. It does not replace the NCM
architecture, tokenizer, VQ-VAE, or coarse-to-fine concept generation.

Safety policy:
- Never treat tool intent as proof that a tool was executed.
- Never approve unsupported measurements, system status, search results,
  repository state, or execution claims.
- Keep source corpus read-only.
- Create output files immediately, show per-record progress, and stop after the first teacher API failure.
- Store V7 reviews and curated output in separate files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = 7


class TeacherFormatError(ValueError):
    """Invalid JSON response from teacher."""

DEFAULT_ENDPOINT = os.environ.get(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate",
)
DEFAULT_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "huihui_ai/gemma-4-abliterated:26b",
)
DEFAULT_SOURCE = "jupris_corpus.jsonl"
DEFAULT_LEDGER = "jupris_teacher_reviews_v7.jsonl"
DEFAULT_CURATED = "jupris_corpus_curated_v7.jsonl"

ALLOWED_CONCEPTS = {
    120, 125, 150, 155, 210, 220,
    480, 481, 482, 483, 484,
}
ALLOWED_LANGS = {"id", "jv", "en", "mix"}
TOOL_CONCEPTS = {480, 481, 482, 483, 484}

LANG_BY_CONCEPT = {
    120: {"id", "mix"},
    125: {"id", "mix"},
    150: {"jv", "mix"},
    155: {"jv", "mix"},
    210: {"en", "mix"},
    220: {"mix", "id", "jv", "en"},
    480: ALLOWED_LANGS,
    481: ALLOWED_LANGS,
    482: ALLOWED_LANGS,
    483: ALLOWED_LANGS,
    484: ALLOWED_LANGS,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(obj: dict[str, Any]) -> str:
    raw = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] Invalid JSON {path}:{line_no}: {exc.msg}",
                    file=sys.stderr,
                )
                continue

            if isinstance(obj, dict):
                yield line_no, obj
            else:
                print(
                    f"[WARN] Non-object JSON {path}:{line_no}; skipped",
                    file=sys.stderr,
                )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, path)


def extract_pair(obj: dict[str, Any]) -> tuple[str, str]:
    user = str(obj.get("user", "") or "").strip()
    assistant = str(obj.get("assistant", "") or "").strip()

    text = obj.get("text", "")
    if (not user or not assistant) and isinstance(text, str):
        match = re.search(r"\bMini-Jup\s*:", text, re.IGNORECASE)
        if match:
            left = text[:match.start()]
            right = text[match.end():]
            left = re.sub(r"^\s*User\s*:", "", left, flags=re.IGNORECASE)
            user, assistant = left.strip(), right.strip()

    return user, assistant


def canonical_source_record(
    obj: dict[str, Any],
) -> dict[str, Any] | None:
    user, assistant = extract_pair(obj)

    if not user or not assistant:
        return None

    try:
        concept_id = int(obj.get("concept_id"))
    except (TypeError, ValueError):
        return None

    if concept_id not in ALLOWED_CONCEPTS:
        return None

    lang = str(obj.get("lang", "id")).strip().lower()
    # Accept the old "tool" value only as a legacy source placeholder so the
    # teacher can repair it. It is never a valid output language label.
    if lang not in ALLOWED_LANGS and lang != "tool":
        return None

    return {
        "user": user,
        "assistant": assistant,
        "lang": lang,
        "concept_id": concept_id,
    }


def load_latest_reviews(
    ledger_path: Path,
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}

    if not ledger_path.exists():
        return latest

    for _, row in read_jsonl(ledger_path):
        key = row.get("source_hash")
        if (
            isinstance(key, str)
            and row.get("reviewer_version") == VERSION
            and not row.get("retryable_error", False)
        ):
            latest[key] = row

    return latest


def parse_json_response(raw: str) -> dict[str, Any]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "", raw, flags=re.IGNORECASE
    ).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise TeacherFormatError(
                "Teacher response contained no JSON object"
            ) from exc
        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError as inner:
            raise TeacherFormatError(
                f"Teacher response contained malformed JSON: {inner.msg}"
            ) from inner

    if not isinstance(data, dict):
        raise TeacherFormatError("Teacher response must be a JSON object")
    return data

def teacher_prompt(record: dict[str, Any]) -> str:
    return f"""
You are a strict dataset curator for JuprisMini-NCM.
Review one training example. Do not execute tools.

SOURCE RECORD:
{json.dumps(record, ensure_ascii=False)}

Concept definitions:
120 Indonesian casual/slang.
125 Indonesian formal/polite/technical.
150 Javanese ngoko.
155 Javanese krama/polite.
210 English.
220 natural code-switching.
480 web-search intent.
481 hardware/sensor/GPIO intent.
482 Python execution intent.
483 system/resource-status intent.
484 repository-inspection intent.

These concept IDs are project labels, not NCM VQ codebook IDs.

Review answer quality and label quality independently.

TOOL SAFETY:
For concept IDs 480-484, the user message describes an intent.
It does NOT prove that a tool was executed.
There is no verified tool trace in this dataset schema.
Never invent measurements, search results, system status, repository state,
or successful execution/restart claims.

Examples of unsafe claims without evidence:
- "Suhu mesin stabil di 90°C."
- "All services are online."
- "Kode berhasil dieksekusi."
- "Repository sudah diperiksa dan service direstart."
- Javanese claims that a watchdog/service has already restarted or is running.

If the existing answer makes an unsupported result claim, replace it with
a concise, honest acknowledgement of the requested action. Examples:
- "Aku akan memeriksa suhu mesin melalui tool sensor."
- "Saya akan memeriksa status sistem menggunakan tool yang tersedia."
- "I can run the script using the available execution tool."
Do not claim that the action has already happened.

Do not force tool acknowledgements onto ordinary conversation examples.
Do not reject a valid short greeting or thanks just because it is short.
Do not change answers merely for cosmetic punctuation.
Be skeptical of irrelevant or nonsensical responses.

Verdicts:
answer_verdict: keep | correct | reject | uncertain
label_verdict: keep | correct | uncertain

If answer_verdict=keep, corrected_answer must equal the source answer.
If answer_verdict=correct, corrected_answer must be materially better and
must differ from the source answer.
If a tool-result answer lacks evidence, correct it to a safe acknowledgement
when you can confidently do so. Never manufacture tool results.

Allowed language labels: id, jv, en, mix. "tool" is not a language label.
If SOURCE RECORD lang is the legacy value "tool", infer the natural language
from the user/assistant text. Set label_verdict=correct and corrected_lang to
id, jv, en, or mix. Never output lang="tool".
Concept IDs are task/intent metadata, separate from language labels and NCM VQ codes.
Concept 484 means repository inspection only; restart/watchdog/service-control
requests do not match concept 484 and must not be approved under that label.
Use corrected_concept_id from the allowed concept list above only when the intent
clearly matches it; otherwise mark the label uncertain.

Return ONLY JSON with these keys:
{{
  "answer_verdict": "keep|correct|reject|uncertain",
  "answer_score": 0,
  "label_verdict": "keep|correct|uncertain",
  "label_score": 0,
  "corrected_user": "original user text",
  "corrected_answer": "answer to train",
  "corrected_lang": "id|jv|en|mix",
  "corrected_concept_id": 120,
  "reason": "brief explanation"
}}

Scores:
90-100: strong answer/label.
80-89: usable with meaningful correction or minor weakness.
40-79: weak or needs correction.
0-39: unusable.
Do not give 100 merely because text sounds fluent.
"""


def call_teacher(
    endpoint: str,
    model: str,
    record: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": teacher_prompt(record),
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 0.8,
            "num_predict": 500,
        },
    }

    # Retry malformed teacher output only. Network/API errors propagate.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")

        try:
            envelope = json.loads(body)
        except json.JSONDecodeError as exc:
            error = TeacherFormatError(
                f"Ollama response envelope is not JSON: {exc.msg}"
            )
            if attempt == max_attempts:
                raise error from exc
            print(
                f"[WARN] Invalid Ollama envelope "
                f"{attempt}/{max_attempts}; retrying same record",
                file=sys.stderr, flush=True,
            )
            continue

        raw = str(envelope.get("response", "") or "").strip()
        try:
            if not raw:
                raise TeacherFormatError("Ollama response field was empty")
            return parse_json_response(raw)
        except TeacherFormatError as exc:
            if attempt == max_attempts:
                raise
            print(
                f"[WARN] Invalid teacher JSON "
                f"{attempt}/{max_attempts}; retrying same record: {exc}",
                file=sys.stderr, flush=True,
            )

    raise TeacherFormatError("Teacher did not return valid JSON after retries")

def parse_score(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0




def tool_intent_matches_concept(record: dict[str, Any]) -> bool:
    """Conservative semantic guard for the project-defined tool labels.

    These are dataset labels, not NCM codebook IDs. Ambiguous/mixed intents
    stay pending rather than being silently relabeled as an executable tool.
    """
    cid = int(record.get("concept_id", -1))
    text = str(record.get("user", "")).casefold()

    patterns = {
        480: r"\b(?:search|web|internet|online|\bnet\b|look\s*up|google|"
             r"cari(?:kan|\s+di)?|telusuri|golek(?:i)?|nggolek(?:i)?)\b",
        481: r"\b(?:gpio|hardware|sensor|suhu|temperatur|temperature|temp|"
             r"volt(?:age|ase)?|tegangan|aki|baterai|battery|engine|mesin|"
             r"thermistor|pin)\b",
        482: r"\b(?:python|script|kode|code|execute|execution|run\s+(?:the\s+)?"
             r"(?:script|code)|jalankan\s+(?:kode|script)|eksekusi)\b",
        483: r"\b(?:status|ram|memory|memori|cpu|npu|system|sistem|resource|"
             r"kapasitas|load|beban|usage|penggunaan|utilization|utilisasi|"
             r"service|layanan|heartbeat|watchdog|health|check|cek|periksa|"
             r"pamriksa|pemeriksaan)\b",
        484: r"\b(?:repo|repository|git|codebase|branch|commit|working\s+tree|"
             r"source\s+tree|file\s+project)\b",
    }
    pattern = patterns.get(cid)
    if pattern is None or not re.search(pattern, text, re.IGNORECASE):
        return False

    # Service status checks are concept 483, but service-control side effects
    # are not covered by the current project taxonomy.
    if cid == 483 and re.search(
        r"\b(?:restart(?:ing)?|reboot(?:ing)?|start|stop|enable|disable|"
        r"kill|terminate|flash(?:ing)?|update\s+firmware|baleni|dibaleni|"
        r"dimulai\s+maneh|diwiwiti\s+maneh|mulai\s+ulang)\b",
        text,
        re.IGNORECASE,
    ):
        return False

    # Restart/heartbeat/watchdog operations are not repository inspection.
    if cid == 484 and re.search(
        r"\b(?:restart|reboot|watchdog|heartbeat|service|layanan|"
        r"restart(?:ing)?|restarting|baleni|dibaleni|dimulai\s+maneh)\b",
        text,
        re.IGNORECASE,
    ):
        return False

    # Firmware flashing is not Python execution under the corrected taxonomy.
    if cid == 482 and re.search(
        r"\b(?:firmware|flash(?:ing)?|bootloader)\b", text, re.IGNORECASE
    ) and not re.search(r"\b(?:python|script|kode|code)\b", text, re.IGNORECASE):
        return False

    return True


def safe_tool_intent_answer(record: dict[str, Any], language: str | None = None) -> str:
    """Create a safe intent acknowledgement; never claim execution/results."""
    concept_id = int(record.get("concept_id", -1))
    lang = str(language or record.get("lang", "id")).strip().lower()
    user_text = str(record.get("user", "")).strip()

    if concept_id == 480:
        messages = {
            "id": f"Saya akan mencoba mencari informasi yang diminta melalui web: {user_text}",
            "en": f"I will try to search the web for the requested information: {user_text}",
            "jv": f"Aku bakal nyoba nggoleki informasi sing dijaluk liwat web: {user_text}",
            "mix": f"Saya akan mencoba mencari informasi yang diminta lewat web: {user_text}",
            "tool": f"Saya akan mencoba mencari informasi yang diminta melalui web: {user_text}",
        }
    else:
        messages_by_concept = {
            481: {
                "id": "Saya akan mencoba memeriksa hardware atau sensor yang diminta menggunakan tool yang tersedia.",
                "en": "I will try to check the requested hardware or sensor with a suitable tool.",
                "jv": "Aku bakal nyoba mriksa hardware utawa sensor sing dijaluk nganggo tool sing kasedhiya.",
                "mix": "Saya akan mencoba memeriksa hardware atau sensor yang diminta dengan tool yang tersedia.",
            },
            482: {
                "id": "Saya akan mencoba menjalankan kode Python yang diminta menggunakan tool eksekusi yang tersedia.",
                "en": "I will try to run the requested Python code with a suitable execution tool.",
                "jv": "Aku bakal nyoba nglakokake kode Python sing dijaluk nganggo tool eksekusi sing kasedhiya.",
                "mix": "Saya akan mencoba menjalankan kode Python yang diminta dengan execution tool yang tersedia.",
            },
            483: {
                "id": "Saya akan mencoba memeriksa status sistem atau penggunaan sumber daya melalui tool yang tersedia.",
                "en": "I will try to check system status or resource usage with a suitable tool.",
                "jv": "Aku bakal nyoba mriksa status sistem utawa panggunaan sumber daya nganggo tool sing kasedhiya.",
                "mix": "Saya akan mencoba memeriksa status sistem atau resource usage dengan tool yang tersedia.",
            },
            484: {
                "id": "Saya akan mencoba memeriksa repository yang diminta menggunakan tool yang tersedia.",
                "en": "I will try to inspect the requested repository with a suitable tool.",
                "jv": "Aku bakal nyoba mriksa repository sing dijaluk nganggo tool sing kasedhiya.",
                "mix": "Saya akan mencoba memeriksa repository yang diminta dengan tool yang tersedia.",
            },
        }
        messages = messages_by_concept.get(concept_id, {})

    return messages.get(lang, messages.get("id", ""))


def has_unverified_tool_result_claim(
    record: dict[str, Any],
    answer: str,
) -> bool:
    """Conservative heuristic; source has no verified execution-trace field."""
    if int(record.get("concept_id", -1)) not in TOOL_CONCEPTS:
        return False

    text = answer.casefold()

    patterns = [
        # Measurements and numeric telemetry
        r"\b\d+(?:[.,]\d+)?\s*(?:°\s*c|°c|degrees?\s+c|celsius|fahrenheit|c\b|volts?|v|voltase|"
        r"amps?|ma|percent|%|mhz|ghz|gb|mb)\b",

        # Claims that a tool action is already in progress, not future intent.
        r"^\s*(?:currently\s+)?(?:executing|running|restarting|checking|"
        r"searching|scanning|flashing)\b",
        r"^\s*(?:i am|we are|currently)\s+(?:executing|running|restarting|"
        r"checking|searching|scanning|flashing)\b",
        r"^\s*(?:sedang|tengah)\s+(?:menjalankan|mengeksekusi|memeriksa|"
        r"mencari|merestart|me-restart|mem-flash)\b",
        r"^\s*(?:sedang|tengah)\s+(?:dijalankan|dieksekusi|diperiksa|"
        r"direstart|di-flash)\b",

        # English status/results
        r"\b(?:is|are|was|were)\s+(?:currently\s+)?(?:normal|stable|idle|online|active|offline|"
        r"running|stopped|healthy|safe|minimal|maximum|low|high|elevated|overloaded)\b",
        r"\b(?:successfully executed|executed successfully|has been executed|"
        r"service restarted|i restarted|i checked|i have checked|"
        r"all services are online|search results show)\b",

        # Indonesian results / completed actions
        r"\b(?:suhu|temperatur|temperature|temp|engine|hardware|voltase|tegangan|sensor|baterai|ram|npu|cpu|"
        r"service|layanan|watchdog|heartbeat|repo|repository|sistem|server)\b"
        r".{0,90}\b(?:normal|stabil|aman|baik|bagus|aktif|idle|online|offline|terbaca|"
        r"terdeteksi|menunjukkan|menunjukan|sudah|telah|berhasil|dijalankan|"
        r"dieksekusi|direstart|diperiksa|dicek)\b",

        r"\b(?:sudah dieksekusi|telah dieksekusi|berhasil dijalankan|"
        r"telah dijalankan|sudah dijalankan|telah direstart|sudah direstart|"
        r"semua layanan aktif|semua layanan online|hasil pencarian menunjukkan|"
        r"saya sudah memeriksa|sudah saya cek|aku sudah cek)\b",

        # Common Javanese completed-action / status claims
        r"\b(?:wis|wes|sampun)\b.{0,70}\b(?:dimulai|diwiwiti|diuripake|"
        r"diaktifkan|direstart|diulang|dibaleni|dijalankan|dieksekusi|"
        r"berhasil|normal|stabil|aktif|aman|terbaca|terdeteksi|mlaku|urip)\b",

        r"\b(?:heartbeat|watchdog|layanan|service|repo|repository|sistem|"
        r"server|sensor|suhu|voltase|tegangan|ram|npu|cpu)\b.{0,90}\b"
        r"(?:wis|wes|sampun|sudah|telah|normal|stabil|aktif|aman|online|"
        r"running|berhasil|dimulai maneh|diwiwiti maneh|dibaleni)\b",

        r"\b(?:layanan|service|heartbeat|watchdog)\b.{0,70}\b"
        r"(?:dimulai maneh|diwiwiti maneh|wis urip|wes urip|wis mlaku|"
        r"wes mlaku|wis dibaleni|wes dibaleni)\b",
    ]

    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def normalize_review(
    source: dict[str, Any],
    result: dict[str, Any],
    min_answer_score: int,
    min_label_score: int,
) -> dict[str, Any]:
    answer_verdict = str(
        result.get("answer_verdict", "uncertain")
    ).strip().lower()
    label_verdict = str(
        result.get("label_verdict", "uncertain")
    ).strip().lower()

    if answer_verdict not in {"keep", "correct", "reject", "uncertain"}:
        answer_verdict = "uncertain"
    if label_verdict not in {"keep", "correct", "uncertain"}:
        label_verdict = "uncertain"

    answer_score = parse_score(result.get("answer_score"))
    label_score = parse_score(result.get("label_score"))
    teacher_answer_verdict = answer_verdict
    reason = str(result.get("reason", "No reason supplied"))[:1500]
    fallback_applied = False
    answer_source = "teacher"

    proposed_user = str(
        result.get("corrected_user", source["user"]) or ""
    ).strip()
    proposed_answer = str(
        result.get("corrected_answer", source["assistant"]) or ""
    ).strip()
    proposed_lang = str(
        result.get("corrected_lang", source["lang"]) or ""
    ).strip().lower()

    try:
        proposed_cid = int(
            result.get("corrected_concept_id", source["concept_id"])
        )
    except (TypeError, ValueError):
        proposed_cid = -1

    # Avoid material rewrites of the user's intent.
    if (
        not proposed_user
        or (
            proposed_user != source["user"]
            and len(proposed_user) > max(2000, len(source["user"]) * 3)
        )
    ):
        proposed_user = source["user"]
        reason += " | Invalid user rewrite ignored."

    # A "keep" verdict must preserve the source answer exactly.
    if answer_verdict == "keep":
        proposed_answer = source["assistant"]

    # Pick the candidate label only when the teacher explicitly proposes a
    # correction. The label still has to pass confidence/schema checks below.
    source_cid = int(source.get("concept_id", -1))
    corrected_candidate = (
        label_verdict == "correct"
        and proposed_cid in TOOL_CONCEPTS
        and proposed_lang in LANG_BY_CONCEPT.get(proposed_cid, set())
        and label_score >= min_label_score
    )
    fallback_cid = proposed_cid if corrected_candidate else source_cid
    fallback_record = {**source, "concept_id": fallback_cid}

    # Repair unsafe source answers deterministically only if the current or
    # confidently corrected label matches the user's actual intent.
    source_answer_unsafe = has_unverified_tool_result_claim(
        source, source["assistant"]
    )
    if source_cid in TOOL_CONCEPTS and source_answer_unsafe:
        if (
            fallback_cid in TOOL_CONCEPTS
            and tool_intent_matches_concept(fallback_record)
        ):
            fallback = safe_tool_intent_answer(
                fallback_record,
                proposed_lang if proposed_lang in ALLOWED_LANGS else None,
            )
            if fallback and not has_unverified_tool_result_claim(fallback_record, fallback):
                proposed_answer = fallback
                answer_verdict = "correct"
                fallback_applied = True
                answer_source = "deterministic_fallback"
                reason += (
                    " | V6 deterministic fallback: replaced an unsupported "
                    "tool-result claim with a safe intent acknowledgement."
                )
            else:
                answer_verdict = "uncertain"
                proposed_answer = ""
                reason += " | V6 fallback unavailable or failed safety validation."
        else:
            answer_verdict = "uncertain"
            proposed_answer = ""
            reason += (
                " | V6 quarantine: the source or corrected tool label does "
                "not match a supported project intent; manual review required."
            )

    # "Correct" must supply a changed answer unless a documented deterministic
    # fallback supplied it.
    if answer_verdict == "correct" and not fallback_applied and (
        not proposed_answer or proposed_answer == source["assistant"]
    ):
        answer_verdict = "uncertain"
        reason += " | V6 guard: correction did not change the answer."

    answer_ok = bool(proposed_answer) and (
        fallback_applied
        or (
            answer_verdict in {"keep", "correct"}
            and answer_score >= min_answer_score
        )
    )

    if answer_verdict == "reject":
        answer_ok = False

    if label_verdict == "keep":
        proposed_lang = source["lang"]
        proposed_cid = source["concept_id"]
        compatible = (
            proposed_lang
            in LANG_BY_CONCEPT.get(proposed_cid, set())
        )
        label_ok = label_score >= min_label_score and compatible
        if not compatible:
            reason += " | Original language/concept labels are incompatible."
    elif label_verdict == "correct":
        label_ok = (
            label_score >= min_label_score
            and proposed_lang in ALLOWED_LANGS
            and proposed_cid in ALLOWED_CONCEPTS
            and proposed_lang
            in LANG_BY_CONCEPT.get(proposed_cid, set())
        )
        if not label_ok:
            reason += " | Proposed labels failed schema validation."
    else:
        label_ok = False

    source_is_tool = int(source.get("concept_id", -1)) in TOOL_CONCEPTS
    proposed_is_tool = proposed_cid in TOOL_CONCEPTS
    if source_is_tool or proposed_is_tool:
        if label_verdict == "correct":
            # Evaluate the corrected target label, not the stale source label.
            label_candidate = {**source, "concept_id": proposed_cid}
            tool_label_matches = (
                label_ok
                and proposed_is_tool
                and tool_intent_matches_concept(label_candidate)
            )
        elif label_verdict == "keep":
            tool_label_matches = label_ok and tool_intent_matches_concept(source)
        else:
            tool_label_matches = False

        if not tool_label_matches:
            label_ok = False
            reason += (
                " | V6 TOOL-LABEL BLOCK: neither the retained nor the "
                "validated corrected label matches a supported project intent."
            )

    # Final deterministic guard on the selected answer, not just source text.
    if has_unverified_tool_result_claim(source, proposed_answer):
        answer_ok = False
        answer_verdict = "uncertain"
        reason += (
            " | V6 TOOL-SAFETY BLOCK: unverified tool result/status claim."
        )

    approved = answer_ok and label_ok and answer_verdict != "reject"

    if not approved:
        overall = (
            "uncertain"
            if answer_verdict == "uncertain"
            or label_verdict == "uncertain"
            else "needs_review"
        )
        selected = None
    else:
        if answer_verdict == "correct" and label_verdict == "correct":
            overall = "correct_both"
        elif answer_verdict == "correct":
            overall = "correct_answer"
        elif label_verdict == "correct":
            overall = "correct_label"
        else:
            overall = "keep"

        selected = {
            "user": proposed_user,
            "assistant": proposed_answer,
            "lang": proposed_lang,
            "concept_id": proposed_cid,
            "supervision_type": (
                "tool_intent_text"
                if proposed_cid in TOOL_CONCEPTS
                else "conversation_text"
            ),
        }

    return {
        "source_hash": stable_hash(source),
        "reviewed_at": now_iso(),
        "source": source,
        "answer_verdict": answer_verdict,
        "teacher_answer_verdict": teacher_answer_verdict,
        "answer_score": answer_score,
        "answer_score_scope": (
            "original_source_answer" if fallback_applied else "teacher_review_result"
        ),
        "answer_source": answer_source,
        "fallback_applied": fallback_applied,
        "label_verdict": label_verdict,
        "label_score": label_score,
        "verdict": overall,
        "score": min(answer_score, label_score),
        "reason": reason,
        "approved": approved,
        "selected": selected,
        "reviewer_version": VERSION,
    }


def rebuild_curated(
    source_path: Path,
    curated_path: Path,
    reviews: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    output: list[dict[str, Any]] = []
    pending = 0
    seen: set[str] = set()

    for line_no, raw in read_jsonl(source_path):
        source = canonical_source_record(raw)
        if source is None:
            pending += 1
            continue

        key = stable_hash(source)
        if key in seen:
            continue
        seen.add(key)

        review = reviews.get(key)
        if (
            not review
            or not review.get("approved")
            or not isinstance(review.get("selected"), dict)
        ):
            pending += 1
            continue

        selected = review["selected"]
        output.append({
            "text": (
                f"User: {selected['user']}\n"
                f"Mini-Jup: {selected['assistant']}"
            ),
            "lang": selected["lang"],
            "concept_id": int(selected["concept_id"]),
            "teacher_score": int(review.get("answer_score", 0)),
            "teacher_score_scope": review.get(
                "answer_score_scope", "teacher_review_result"
            ),
            "teacher_label_score": int(review.get("label_score", 0)),
            "teacher_verdict": review.get("verdict", "keep"),
            "teacher_answer_verdict": review.get(
                "teacher_answer_verdict", review.get("answer_verdict", "unknown")
            ),
            "answer_source": review.get("answer_source", "teacher"),
            "fallback_applied": bool(review.get("fallback_applied", False)),
            "teacher_reason": review.get("reason", ""),
            "source_hash": key,
            "source_line": line_no,
            "record_revision": (
                1 if review.get("verdict") == "keep" else 2
            ),
            "training_status": "approved",
            "supervision_type": selected.get(
                "supervision_type", "conversation_text"
            ),
            "tool_execution_verified": False,
        })

    atomic_write_jsonl(curated_path, output)
    return len(output), pending


def run_scan(args: argparse.Namespace) -> tuple[int, bool]:
    source_path = Path(args.source)
    ledger_path = Path(args.ledger)
    curated_path = Path(args.curated)

    if not source_path.is_file():
        print(f"[ERROR] Source not found: {source_path}", file=sys.stderr)
        return 2, False

    valid_by_hash: dict[str, dict[str, Any]] = {}
    invalid = 0

    for _, raw in read_jsonl(source_path):
        record = canonical_source_record(raw)
        if record is None:
            invalid += 1
            continue
        valid_by_hash[stable_hash(record)] = record

    reviews = load_latest_reviews(ledger_path)

    if args.recheck:
        pending = list(valid_by_hash.items())
    else:
        pending = [
            (key, record)
            for key, record in valid_by_hash.items()
            if key not in reviews
        ]

    total_pending = len(pending)
    print(
        f"[*] Source unique valid={len(valid_by_hash)}"
        f" | invalid={invalid}"
        f" | ledger reviewed={len(reviews)}"
        f" | waiting={total_pending}"
    , flush=True)

    if args.dry_run:
        approved = sum(bool(row.get("approved")) for row in reviews.values())
        print(
            f"[DRY-RUN] Existing V7 approved reviews={approved}; "
            "no teacher calls or file changes."
        , flush=True)
        return 0, True

    if args.limit:
        pending = pending[:args.limit]

    # Make output paths visible before the first network call. This avoids
    # the misleading state where the process is busy but no files exist.
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.touch(exist_ok=True)
    if not curated_path.exists():
        atomic_write_jsonl(curated_path, [])

    print(f"[*] This run will review={len(pending)} of waiting={total_pending}", flush=True)
    print(f"[*] Ledger:  {ledger_path.resolve()}", flush=True)
    print(f"[*] Curated: {curated_path.resolve()}", flush=True)

    if not pending:
        approved, waiting = rebuild_curated(
            source_path, curated_path, reviews
        )
        print(
            f"[+] Curated corpus rebuilt: approved={approved}"
            f" | pending/invalid={waiting} | {curated_path}"
        )
        return 0, True

    done = 0
    for index, (source_hash, record) in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] REQUESTING teacher"
            f" | model={args.model}"
            f" | cid={record['concept_id']}"
            f" | {record['user'][:65]}",
            flush=True,
        )
        try:
            result = call_teacher(
                args.endpoint, args.model, record, args.timeout
            )
            review = normalize_review(
                record,
                result,
                args.min_answer_score,
                args.min_label_score,
            )
            append_jsonl(ledger_path, review)
            reviews[source_hash] = review
            done += 1

            selected = review.get("selected") or {}
            print(
                f"[{done}/{len(pending)}]"
                f" A={review['answer_score']:3d}/{review['answer_verdict']:<9}"
                f" L={review['label_score']:3d}/{review['label_verdict']:<8}"
                f" approved={review['approved']}"
                f" cid={selected.get('concept_id', record['concept_id'])}"
                f" | {record['user'][:65]}"
            )

        except TeacherFormatError as exc:
            # After three invalid JSON responses, quarantine this row
            # and continue. It is recorded as reviewed, not approved.
            quarantine = {
                "source_hash": source_hash,
                "reviewed_at": now_iso(),
                "source": record,
                "answer_verdict": "uncertain",
                "answer_score": 0,
                "label_verdict": "uncertain",
                "label_score": 0,
                "verdict": "uncertain",
                "score": 0,
                "reason": (
                    "Teacher returned invalid JSON after retries: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "approved": False,
                "selected": None,
                "retryable_error": False,
                "quarantined_teacher_format": True,
                "reviewer_version": VERSION,
            }
            append_jsonl(ledger_path, quarantine)
            reviews[source_hash] = quarantine
            done += 1
            print(
                f"[{done}/{len(pending)}] QUARANTINED after JSON retries "
                f"| {record['user'][:65]}",
                file=sys.stderr,
                flush=True,
            )
            continue

        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ) as exc:
            print(
                f"[WARN] Teacher call failed for {source_hash[:12]}: {exc}",
                file=sys.stderr,
            )
            append_jsonl(ledger_path, {
                "source_hash": source_hash,
                "reviewed_at": now_iso(),
                "source": record,
                "verdict": "uncertain",
                "score": 0,
                "reason": f"Teacher call failed: {type(exc).__name__}: {exc}",
                "approved": False,
                "selected": None,
                "retryable_error": True,
                "reviewer_version": VERSION,
            })
            # Fail fast: do not repeat a broken endpoint across thousands
            # of rows. The retryable error is excluded from latest reviews,
            # so rerunning after fixing connectivity retries this source hash.
            reviews = load_latest_reviews(ledger_path)
            approved, waiting = rebuild_curated(
                source_path, curated_path, reviews
            )
            print(
                f"[STOP] Teacher API failed on the first failing record. "
                f"No further rows attempted. approved={approved} "
                f"| pending/invalid={waiting}",
                file=sys.stderr,
                flush=True,
            )
            return 2, False

    reviews = load_latest_reviews(ledger_path)
    approved, waiting = rebuild_curated(
        source_path, curated_path, reviews
    )
    print(
        f"[+] Scan complete: reviewed={done}"
        f" | approved unique rows={approved}"
        f" | pending/invalid={waiting}"
    )
    return 0, done > 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JuprisMini-NCM Gemma/Qwen Teacher Curator V7"
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--curated", default=DEFAULT_CURATED)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-answer-score", type=int, default=80)
    parser.add_argument("--min-label-score", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()

    if (
        args.limit < 0
        or not 0 <= args.min_answer_score <= 100
        or not 0 <= args.min_label_score <= 100
        or args.timeout < 1
        or args.interval < 1
    ):
        parser.error("Invalid limit, score threshold, timeout, or interval.")

    while True:
        code, did_work = run_scan(args)
        if code != 0 or not args.watch:
            return code
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
