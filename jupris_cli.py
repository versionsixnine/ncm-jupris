import sys, time, json, re, torch
from ncm_core import JuprisNCM

VOCAB_PATH = "/opt/datasetjup/jupris_vocab.json"
CKPT_PATH = "/opt/datasetjup/jupris_ncm_prod_ready.pt"

vocab = {str(k): int(v) for k, v in json.load(open(VOCAB_PATH, encoding="utf-8")).items()}
id2tok = {v: k for k, v in vocab.items()}

raw = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = raw.get("model_state_dict", raw)
model = JuprisNCM(vocab_size=sd["tokenizer.token_embed.weight"].shape[0]).eval()
model.load_state_dict(sd, strict=True)

def run_query(prompt: str):
    t0 = time.perf_counter()
    words = re.sub(r"[^\w\s]", " ", prompt.lower()).split()[:32]
    ids = [vocab.get(w, 1) for w in words]
    t = torch.tensor([ids + [0] * (32 - len(ids))], dtype=torch.long)
    with torch.no_grad():
        r = model.route(t, tool_conf_threshold=0.65)
        gen, _ = model.chat(t, temperature=0.0, top_k=0)
        ans = " ".join(id2tok.get(int(i), "") for i in gen[0].tolist() if int(i) not in (0, 1)).strip()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"[{dt:5.1f} ms] [{r.get('type'):11s} | cid={r.get('concept_id')} | tool={r.get('tool_name')} | conf={float(r.get('confidence',0)):.3f}]")
    print(f"JUPRIS > {ans}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(f"USER   > {arg}")
            run_query(arg)
    else:
        print("=== JUPRIS-NCM PRODUCTION CLI (Ctrl+C to exit) ===")
        while True:
            try:
                q = input("USER   > ").strip()
                if q:
                    run_query(q)
            except (EOFError, KeyboardInterrupt):
                print()
                break
