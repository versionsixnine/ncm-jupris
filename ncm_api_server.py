import time, json, re, torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ncm_core import JuprisNCM

VOCAB_PATH = "/opt/datasetjup/jupris_vocab.json"
CKPT_PATH = "/opt/datasetjup/jupris_ncm_prod_ready.pt"
PORT = 8893

vocab = {str(k): int(v) for k, v in json.load(open(VOCAB_PATH, encoding="utf-8")).items()}
id2tok = {v: k for k, v in vocab.items()}

raw = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = raw.get("model_state_dict", raw)
model = JuprisNCM(vocab_size=sd["tokenizer.token_embed.weight"].shape[0]).eval()
model.load_state_dict(sd, strict=True)

# Warm-up 1x saat boot agar request pertama langsung ~33 ms
with torch.no_grad():
    _dummy = torch.zeros((1, 32), dtype=torch.long)
    model.route(_dummy)
    model.chat(_dummy)

class NCMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "port": PORT, "checkpoint": CKPT_PATH, "vocab_size": len(vocab)})
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/v1/chat":
            self._send_json(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        prompt = str(data.get("prompt", ""))
        thresh = float(data.get("tool_conf_threshold", 0.65))

        t0 = time.perf_counter()
        words = re.sub(r"[^\w\s]", " ", prompt.lower()).split()[:32]
        ids = [vocab.get(w, 1) for w in words]
        t = torch.tensor([ids + [0] * (32 - len(ids))], dtype=torch.long)
        with torch.no_grad():
            r = model.route(t, tool_conf_threshold=thresh)
            gen, _ = model.chat(t, temperature=0.0, top_k=0)
            ans = " ".join(id2tok.get(int(i), "") for i in gen[0].tolist() if int(i) not in (0, 1)).strip()
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self._send_json(200, {
            "prompt": prompt,
            "reply": ans,
            "route_type": r.get("type"),
            "concept_id": r.get("concept_id"),
            "tool_name": r.get("tool_name"),
            "confidence": round(float(r.get("confidence", 0.0)), 4),
            "latency_ms": dt_ms,
        })

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), NCMHandler)
    print(f"[+] JuprisMini-NCM Native HTTP Server listening on 0.0.0.0:{PORT}", flush=True)
    srv.serve_forever()
