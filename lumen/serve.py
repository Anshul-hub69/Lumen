"""Zero-dependency HTTP API (stdlib). Swap for FastAPI when you need auth, validation, async.

GET  /health
GET  /categories
GET  /search?q=text           item lookup by title
POST /recommend               {"history":[item_id,...], "categories":["Sci-Fi"], "k":10}
GET  /                        evaluation dashboard
"""
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .pipeline import TwoStageRecommender

REC = None
STATIC = {"/": "dashboard/index.html", "/metrics.js": "reports/metrics.js", "/metrics.json": "reports/metrics.json"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            return self._send(200, {"status": "ok", "items": int(REC.sp.N)})
        if u.path == "/categories":
            return self._send(200, sorted(REC.sp.cat_index))
        if u.path == "/search":
            q = parse_qs(u.query).get("q", [""])[0].lower()
            hits = REC.items[REC.items.title.str.lower().str.contains(q, regex=False)].head(20)
            return self._send(200, hits[["item_id", "title", "category"]].to_dict("records"))
        if u.path in STATIC and os.path.exists(STATIC[u.path]):
            with open(STATIC[u.path], "rb") as f:
                return self._send(200, f.read(), mimetypes.guess_type(STATIC[u.path])[0] or "text/plain")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/recommend":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            hist = [int(i) for i in body.get("history", [])]
            cats = [c for c in body.get("categories", []) if c in REC.sp.cat_index]
            k = max(1, min(int(body.get("k", 10)), 50))
            if any(i < 0 or i >= REC.sp.N for i in hist):
                return self._send(400, {"error": "unknown item id in history"})
        except (ValueError, TypeError, json.JSONDecodeError):
            return self._send(400, {"error": "invalid request"})
        mode = "cold_start" if len(hist) < 3 else "personalized"
        self._send(200, {"mode": mode, "items": REC.recommend(hist, cats, k)})


def serve(port=8000, model="models/recommender.pkl"):
    global REC
    REC = TwoStageRecommender.load(model)
    print(f"Lumen API on http://localhost:{port}  (dashboard at /)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
