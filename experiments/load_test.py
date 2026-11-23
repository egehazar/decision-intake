"""Load test: whole screenings through the real HTTP stack, several clients at once.

Each client takes a training household's answers, starts a screening on /v1/step, answers whatever the
engine asks until it finishes, then fetches /v1/assessment, carrying only the state string between calls,
exactly like the public screener. Reports per-request latency (p50/p95/p99), throughput and errors.

usage: python experiments/load_test.py [CLIENTS] [SCREENINGS_PER_CLIENT] [URL]
       (no URL: starts the service in-process on a free port)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def screening(base, questions, answers, lat, errors):
    try:
        r, ms = post(base + "/v1/step", {})
        lat.append(ms)
        while r["next"]["action"] == "ask":
            qid = r["next"]["question"]["id"]
            j = next(k for k, q in enumerate(questions) if q.id == qid)
            r, ms = post(base + "/v1/step", {"state": r["state"],
                                             "facts": [{"question": qid, "choice": questions[j].options[answers[j]]}]})
            lat.append(ms)
        _, ms = post(base + "/v1/assessment", {"state": r["state"]})
        lat.append(ms)
    except Exception as e:  # noqa: BLE001 - every failure is counted
        errors.append(repr(e))


def main():
    clients = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    from decisionintake import bundle
    from decisionintake.mcp_server import DEFAULT_BUNDLE

    b = bundle.load(str(DEFAULT_BUNDLE))
    if len(sys.argv) > 3:
        base = sys.argv[3].rstrip("/")
    else:
        from socketserver import ThreadingMixIn
        from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

        from decisionintake.web import create_app

        class Server(ThreadingMixIn, WSGIServer):
            daemon_threads = True

        class Quiet(WSGIRequestHandler):
            def log_message(self, *a):
                pass

        app = create_app(static_dir=None, log=None)
        srv = make_server("127.0.0.1", 0, app, server_class=Server, handler_class=Quiet)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"
    rng = np.random.default_rng(0)
    rows = rng.choice(len(b.pop), clients * per, replace=False)
    lat, errors = [], []
    t0 = time.perf_counter()

    def client(k):
        for r in rows[k * per:(k + 1) * per]:
            screening(base, b.pop.questions, b.pop.answers[r], lat, errors)

    threads = [threading.Thread(target=client, args=(k,)) for k in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    a = np.array(lat)
    print(f"{clients} clients x {per} screenings = {clients * per} screenings, {len(a)} requests in {wall:.1f}s "
          f"({len(a) / wall:.1f} requests/s, {clients * per / wall:.2f} screenings/s); errors {len(errors)}")
    print(f"latency per request: p50 {np.percentile(a, 50):.0f} ms, p95 {np.percentile(a, 95):.0f} ms, "
          f"p99 {np.percentile(a, 99):.0f} ms, max {a.max():.0f} ms; {len(a) / (clients * per):.1f} requests a screening")
    if errors:
        print("first error:", errors[0])


if __name__ == "__main__":
    main()
