import os
import sys
import re
import torch

from ncm_core import JuprisNCM, TOOL_THRESHOLD
from ncm_train import get_vocab, text_to_ids, VOCAB_SIZE, PAD_ID, UNK_ID

CONCEPT_MAP = {
    120: "ID-Slang", 125: "ID-Formal", 150: "JV-Ngoko", 155: "JV-Krama",
    210: "EN-Tech", 220: "Tri-Lingual Code-Switch",
    480: "TOOL:search_web", 481: "TOOL:read_gpio",
    482: "TOOL:run_python", 483: "TOOL:system_status", 484: "TOOL:repo_analyzer",
}


def execute_mock_tool(concept_id: int, prompt: str) -> str:
    table = {
        480: "[SEARCH] query dispatched.",
        481: "[GPIO] Pin 14 state: HIGH (3.3V) | Status: OK",
        482: "[SANDBOX] script executed, exit 0.",
        483: "[STATUS] Engine Temp: 78.4°C | Voltage: 12.6V | Status: AMAN",
        484: "[REPO] analysis complete.",
    }
    return table.get(concept_id, f"[TOOL] executed for '{prompt}'")


def clean_text(s: str) -> str:
    s = re.sub(r"\\rightarrow|\$", "", s)
    s = re.sub(r"[^\w\s]", "", s.lower()).strip()
    return s


def ids_to_text(ids, id_to_word) -> str:
    words = []
    for i in ids:
        i = int(i)
        if i == PAD_ID:
            if len(words) > 0:
                break
            continue
        if i == UNK_ID:
            continue
        w = id_to_word.get(i, "")
        if w and (not words or words[-1] != w):
            words.append(w)
    return " ".join(words)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "jupris_ncm_core_v2.pt"
    if not os.path.exists(model_path):
        print(f"[!] {model_path} belum ada. Jalankan `python ncm_train.py` dulu.")
        sys.exit(1)

    vocab = get_vocab()
    id_to_word = {v: k for k, v in vocab.items()}

    model = JuprisNCM(vocab_size=VOCAB_SIZE, hidden_dim=128, num_layers=4, num_heads=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    print("=" * 60)
    print(" JuprisMini-NCM — coarse-to-fine chat + tool router")
    print(" 'exit' untuk keluar.")
    print("=" * 60)

    while True:
        try:
            user_in = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_in or user_in.lower() in ("exit", "quit"):
            break

        clean = clean_text(user_in)
        ids = torch.tensor([text_to_ids(clean, vocab)], dtype=torch.long, device=device)

        route = model.route(ids, tool_threshold=TOOL_THRESHOLD)
        label = CONCEPT_MAP.get(route.get("concept_id", -1), f"Concept-{route.get('concept_id')}")

        if route["type"] == "TOOL_CALL":
            out = execute_mock_tool(route["concept_id"], clean)
            print(f"├─ [ROUTE: {route['concept_id']} ({label}) -> TOOL_CALL]")
            print(f"└─ Mini-Jup > {out}")
            continue

        response_ids, _ = model.chat(ids, temperature=0.2, top_k=20)
        reply = ids_to_text(response_ids[0].tolist(), id_to_word)
        print(f"├─ [ROUTE: {route['concept_id']} ({label}) -> TEXT_STREAM]")
        print(f"└─ Mini-Jup > {reply if reply.strip() else '(decode kosong)'}")


if __name__ == "__main__":
    main()