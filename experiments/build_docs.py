"""Build the developer guide (public/docs.html) with examples produced by the real service.

Every response shown in the guide comes from running a screening through the WSGI app in this process,
so the guide can't drift from the code: rerun this after any change to the API.

usage: python experiments/build_docs.py
"""
from __future__ import annotations

import html
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from decisionintake.mcp_server import TOOLS_STATELESS, VERSION  # noqa: E402
from decisionintake.web import MAX_BODY, MAX_FACTS, App  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "public", "docs.html")


def call(app, path, body):
    raw = json.dumps(body).encode()
    env = {"REQUEST_METHOD": "POST", "PATH_INFO": path, "QUERY_STRING": "", "CONTENT_TYPE": "application/json",
           "CONTENT_LENGTH": str(len(raw)), "wsgi.input": io.BytesIO(raw)}
    return json.loads(b"".join(app(env, lambda s, h: None)))


BASE = os.environ.get("DI_PUBLIC_URL", "https://decision-intake.vercel.app")  # baked into every example
MODERN_META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}


def mcp(app, body, headers=None):
    raw = json.dumps(body).encode()
    env = {"REQUEST_METHOD": "POST", "PATH_INFO": "/mcp", "QUERY_STRING": "", "CONTENT_TYPE": "application/json",
           "CONTENT_LENGTH": str(len(raw)), "wsgi.input": io.BytesIO(raw)}
    for k, v in (headers or {}).items():
        env["HTTP_" + k.upper().replace("-", "_")] = v
    return json.loads(b"".join(app(env, lambda s, h: None)))


def examples():
    app = App(static_dir=None, log=None)
    ex = {}
    ex["start"] = call(app, "/v1/step", {})
    facts = [{"question": "children", "choice": "2"}, {"question": "state", "choice": "TX"},
             {"question": "earned_income", "amount": {"value": 2400, "period": "month", "approximate": True}}]
    ex["facts"] = facts
    ex["step"] = call(app, "/v1/step", {"state": ex["start"]["state"], "facts": facts})
    # a correction, a fact that can't be read, and a plain one, in one call
    ex["mixed_facts"] = [{"question": "children", "choice": "3"}, {"question": "tenure", "choice": "a boat"},
                         {"question": "household_size", "choice": "4"}]
    ex["mixed"] = call(app, "/v1/step", {"state": ex["step"]["state"], "facts": ex["mixed_facts"]})
    answers = {"household_size": "3", "child_under5": "no", "infant_or_pregnant": "no", "senior": "no", "disabled": "no",
               "married": "no", "self_employed": "no", "employer_insurance": "no", "college": "no", "tenure": "rent",
               "social_security": "none", "other_income": "none", "housing_cost": "$1,000 to $1,500", "childcare": "none",
               "assets": "under $2,000"}
    state, nxt = ex["step"]["state"], ex["step"]["next"]
    while nxt["action"] == "ask":
        q = nxt["question"]["id"]
        if nxt["question"]["kind"] == "amount" and "amount_ask" not in ex:
            ex["amount_ask"] = nxt["question"]
        r = call(app, "/v1/step", {"state": state, "facts": [{"question": q, "choice": answers[q]}]})
        state, nxt = r["state"], r["next"]
    ex["state"], ex["finish"] = state, nxt
    if "amount_ask" not in ex:  # the walk asked none: show one from the catalogue
        from decisionintake.service import Intake
        ex["amount_ask"] = next(q for q in Intake(app.b).catalogue() if q["kind"] == "amount")
    # someone who knows little: every later answer "don't know" ends in a hand-off
    r = call(app, "/v1/step", {"facts": [{"question": "state", "choice": "NY"}, {"question": "senior", "choice": "yes"}]})
    st, nx = r["state"], r["next"]
    while nx["action"] == "ask":
        r = call(app, "/v1/step", {"state": st, "facts": [{"question": nx["question"]["id"], "dont_know": True}]})
        st, nx = r["state"], r["next"]
    ex["handoff"] = nx
    ex["bad"] = call(app, "/v1/step", {"state": state[:-1] + ("0" if state[-1] != "0" else "1")})
    # MCP, both eras, on the same endpoint
    ex["mcp_legacy"] = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "my-agent", "version": "1.0"}}}
    ex["mcp_call_body"] = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
        "name": "record_facts", "arguments": {"state": ex["start"]["state"], "facts": facts}, "_meta": MODERN_META}}
    ex["mcp_call_headers"] = {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "record_facts"}
    ex["mcp_call"] = mcp(app, ex["mcp_call_body"], ex["mcp_call_headers"])
    return ex


def code(obj, lang="json"):
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False)
    return f'<pre tabindex="0"><code class="{lang}">{html.escape(text)}</code></pre>'


def main():
    ex = examples()
    start, facts, step, state, finish, bad = ex["start"], ex["facts"], ex["step"], ex["state"], ex["finish"], ex["bad"]
    start_view = {"state": start["state"], "readings": [], "next": {
        "action": "ask", "question": start["next"]["question"], "why": start["next"]["why"], "progress": start["next"]["progress"]}}
    step_view = {"state": step["state"], "readings": step["readings"], "next": {
        k: step["next"][k] for k in ("action", "question", "why")}}
    mixed_view = {"state": ex["mixed"]["state"], "readings": ex["mixed"]["readings"], "next": {"action": ex["mixed"]["next"]["action"], "...": "..."}}
    finish_view = {"action": finish["action"], "recommend": finish["recommend"][:2], "unsure": finish.get("unsure"),
                   "progress": finish["progress"], "note": finish["note"]}
    if len(finish["recommend"]) > 2:
        finish_view["recommend_more"] = f"... {len(finish['recommend']) - 2} more"
    ho = ex["handoff"]
    handoff_view = {"action": ho["action"], "hand_off_reason": ho["hand_off_reason"], "recommend": ho["recommend"][:1],
                    "unsure": (ho.get("unsure") or [])[:2], "progress": ho["progress"]}
    call_view = {"result": {k: v for k, v in ex["mcp_call"]["result"].items() if k != "content"}}
    call_view["result"]["structuredContent"] = {"state": ex["mcp_call"]["result"]["structuredContent"]["state"],
                                                "readings": "...", "next": {"action": "ask", "...": "..."}}
    headers = "\n".join(f"{k}: {v}" for k, v in {"Content-Type": "application/json", **ex["mcp_call_headers"]}.items())
    tools = "".join(f"<li><code>{t['name']}</code> ({html.escape(t.get('title', ''))}): {html.escape(t['description'])}</li>"
                    for t in TOOLS_STATELESS)
    js = """const BASE = "{BASE}";
async function step(state, facts) {
  const r = await fetch(BASE + "/v1/step", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({state, facts})
  });
  if (!r.ok) throw new Error((await r.json()).error.message);
  return r.json();
}

// askThePerson, explain and showResults are your UI (placeholders)
let {state, next} = await step(undefined, []);           // start
while (next.action === "ask") {
  const fact = await askThePerson(next.question, next.why); // a choice, an amount as said, or don't know
  const r = await step(state, [fact]);
  const problem = r.readings.find(x => x.error);
  if (problem) explain(problem.error);                    // not recorded: the same question comes back
  ({state, next} = r);
}
showResults(next);   // recommend, unsure, note; next.action === "hand_off" adds hand_off_reason"""
    py = """import json, urllib.request

BASE = "{BASE}"

def step(state=None, facts=()):
    req = urllib.request.Request(BASE + "/v1/step", method="POST",
        data=json.dumps({"state": state, "facts": list(facts)}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)

r = step(facts=[{"question": "state", "choice": "TX"},
                {"question": "earned_income", "amount": {"value": 2400, "period": "month"}}])
print(r["next"]["action"], r["next"].get("question", {}).get("id"))"""
    curl = """curl -s {BASE}/v1/step -H 'Content-Type: application/json' \\
  -d '{"facts":[{"question":"children","choice":"2"}]}'"""
    js, py, curl = (x.replace("{BASE}", BASE) for x in (js, py, curl))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>DecisionIntake Developer Guide</title>
<meta name="description" content="Plug the DecisionIntake benefits-screening engine into a screener or an AI assistant: the JSON API, the state string, MCP, with real examples.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 28 28'%3E%3Crect x='1' y='1' width='26' height='26' rx='7' fill='%231d5c88'/%3E%3Cpath d='M8 14.5l4 4 8-9' fill='none' stroke='white' stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<style>
:root {{ --bg: #f5f4ef; --surface: #fff; --ink: #18232c; --muted: #56636e; --line: #d9d6cc; --accent: #1d5c88; --code: #eef2f5; color-scheme: light; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #10161b; --surface: #182129; --ink: #e7edf1; --muted: #9aa8b3; --line: #2a3641; --accent: #82b8e2; --code: #0c1218; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #10161b; --surface: #182129; --ink: #e7edf1; --muted: #9aa8b3; --line: #2a3641; --accent: #82b8e2; --code: #0c1218; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
.wrap {{ max-width: 52rem; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
header {{ border-bottom: 1px solid var(--line); background: var(--surface); }}
header .wrap {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; padding-block: 14px; }}
header a {{ font-weight: 700; color: var(--ink); text-decoration: none; }}
main {{ padding-block: 28px 56px; }}
h1 {{ font-size: clamp(1.6rem, 4vw, 2.1rem); line-height: 1.2; margin: 0 0 10px; text-wrap: balance; }}
h2 {{ font-size: 1.3rem; margin: 40px 0 10px; padding-top: 16px; border-top: 1px solid var(--line); }}
h3 {{ font-size: 1.05rem; margin: 22px 0 6px; }}
p, li {{ max-width: 44rem; }}
a {{ color: var(--accent); }}
.lede {{ color: var(--muted); font-size: 1.08rem; }}
code {{ font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: .88em; background: var(--code); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }}
pre:focus-visible {{ outline: 3px solid var(--accent); outline-offset: 2px; }}
pre {{ background: var(--code); border: 1px solid var(--line); border-radius: 8px; padding: 14px; overflow-x: auto; font-size: .85rem; line-height: 1.5; }}
pre code {{ background: none; padding: 0; overflow-wrap: normal; }}
table {{ border-collapse: collapse; width: 100%; font-size: .95rem; }}
.tbl {{ overflow-x: auto; }}
th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 8px 10px; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 600; }}
.note {{ background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 8px; padding: 12px 16px; }}
nav.toc {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 18px; }}
nav.toc ol {{ margin: 6px 0 0; padding-left: 20px; }}
</style>
</head>
<body>
<header><div class="wrap"><a href="./">DecisionIntake</a><span><a href="./">screener</a> · <a href="walkthrough/">walkthrough</a> · <a href="openapi.json">openapi.json</a> · <a href="https://github.com/egehazar/decision-intake" target="_blank" rel="noopener">source</a></span></div></header>
<main class="wrap">
<h1>Developer guide</h1>
<p class="lede">DecisionIntake decides what to ask next in a US benefits screening, how far to trust each answer, and when it knows enough to recommend programs or should suggest a person review the case. Your product keeps the conversation; the engine keeps the decision. Engine {VERSION} at <code>{BASE}</code>; every example below is a real response, generated from the code by <code>experiments/build_docs.py</code>.</p>

<nav class="toc" aria-label="Contents"><b>Contents</b><ol>
<li><a href="#loop">The loop</a></li><li><a href="#facts">Facts and readings</a></li><li><a href="#responses">Responses</a></li>
<li><a href="#state">The state string</a></li><li><a href="#mcp">AI assistants (MCP)</a></li><li><a href="#code">Code</a></li>
<li><a href="#rules">Integration checklist</a></li><li><a href="#limits">Errors, limits and versions</a></li></ol></nav>

<h2 id="loop">1. The loop</h2>
<p>One endpoint does the work. Send what you know, get back the next step, repeat until the step is <code>finish</code> or <code>hand_off</code>. The service keeps nothing between calls: every response carries a <code>state</code> string, and you send it back with the next call. No key, no account.</p>
<div class="tbl"><table>
<tr><th>Call</th><th>Body</th><th>Returns</th></tr>
<tr><td><code>POST /v1/step</code></td><td><code>{{"state"?, "facts"?}}</code></td><td>the new <code>state</code>, how each fact was read, and <code>next</code>: ask, finish or hand_off</td></tr>
<tr><td><code>POST /v1/assessment</code></td><td><code>{{"state"}}</code></td><td>every program's probability, tier and yearly value range; the facts so far; how well the data covers this household</td></tr>
<tr><td><code>GET /v1/questions</code></td><td>-</td><td>the ten programs and all 18 questions with their options</td></tr>
<tr><td><code>POST /mcp</code></td><td>JSON-RPC</td><td>the same engine as tools for AI assistants (<a href="#mcp">section 5</a>)</td></tr>
<tr><td><code>GET /healthz</code></td><td>-</td><td>liveness, version, and the commit and model and rules hashes deployed</td></tr>
</table></div>
<h3>Start</h3>
{code({"facts": []})}
{code(start_view)}
<p><code>why.worth_per_year</code> is the expected cost of mistakes, in dollars a year, that the next question saves; it is asked because that is more than <code>question_cost</code>. <code>mostly_settles</code> says which programs it is about (with their ids), useful for telling the person why you ask.</p>

<h2 id="facts">2. Facts and readings</h2>
<p>Record anything the person said, asked or not. Each fact is exactly one of three kinds:</p>
<div class="tbl"><table>
<tr><th>Kind</th><th>Example</th><th>Notes</th></tr>
<tr><td>choice</td><td><code>{{"question": "children", "choice": "2"}}</code></td><td>the option's text; numbers map onto ranges like "4 or more"; true/false onto yes/no; on amount questions a range's text works too</td></tr>
<tr><td>amount</td><td><code>{{"question": "earned_income", "amount": {{"value": 2400, "period": "month", "approximate": true}}}}</code></td><td>as said: a <code>value</code> or <code>low</code>/<code>high</code>, a <code>period</code> (year, month, week, none). Don't convert it yourself: the engine does the arithmetic and spreads approximate amounts over neighbouring ranges</td></tr>
<tr><td>don't know</td><td><code>{{"question": "assets", "dont_know": true}}</code></td><td>no evidence; the belief is unchanged</td></tr>
</table></div>
<h3>Three facts at once</h3>
{code({"state": start["state"], "facts": facts})}
{code(step_view)}
<p>Each fact gets a <code>reading</code>: what the engine understood, worth showing the person so they can correct it.</p>
<h3>An amount question</h3>
<p>Amount questions carry a <code>unit</code> and the <code>ranges</code> the engine reasons in. Ask for a number; offer the ranges for people who'd rather pick one.</p>
{code(ex["amount_ask"])}
<h3>Corrections, rejected facts and warnings</h3>
<p>A fact for an answered question replaces it (<code>replaces</code> shows the old reading). A fact that can't be read is not recorded, its reading has an <code>error</code> saying how to fix it, and the other facts in the call still go in. A <code>warning</code> means no household on record fits this answer together with the others: check it with the person.</p>
{code({"state": "…", "facts": ex["mixed_facts"]})}
{code(mixed_view)}

<h2 id="responses">3. Responses</h2>
<p>When no remaining question is worth asking, <code>next.action</code> is <code>finish</code>, or <code>hand_off</code> when the doubt left is still costly enough that a person should go through it.</p>
<h3>Finish</h3>
{code(finish_view)}
<div class="tbl"><table>
<tr><th>Field</th><th>Meaning</th></tr>
<tr><td><code>recommend</code></td><td>programs worth applying for; <code>tier</code> is <code>likely</code> (probability at least 0.5) or <code>worth checking</code> (applying is worth it in expectation, but less likely than not); <code>value_if_eligible_per_year</code> is <code>[low, high]</code> in whole dollars (health coverage: its value, not cash)</td></tr>
<tr><td><code>unsure</code></td><td>programs the engine won't call from these answers: few training households look like this person and the program keeps a real chance. Show them as "we couldn't tell, you can still apply", never as unlikely</td></tr>
<tr><td><code>progress.support.households_like_this</code></td><td>the effective number of training households (a weighted count, so it can be fractional) that fit these answers before the support floor; under 10 is <code>thin</code>. <code>progress.similar_households</code> is the same after the floor</td></tr>
<tr><td><code>progress.expected_mistake_cost</code></td><td>dollars a year the remaining doubt is expected to cost</td></tr>
<tr><td><code>note</code></td><td>what this is and isn't; show it with every result</td></tr>
</table></div>
<h3>Hand off</h3>
<p>Someone who answers "don't know" to most questions can't be called with confidence. <code>hand_off_reason</code> is advice to show, not a referral: nothing is sent anywhere.</p>
{code(handoff_view)}

<h2 id="state">4. The state string</h2>
<p>The facts so far, readable, with a checksum so a mangled copy is refused instead of misread:</p>
{code(state, "text")}
<p>Choices are option text (spaces as <code>_</code>), amounts are <code>~</code>-prefixed when approximate with <code>..</code> for ranges and <code>/period</code>, don't-know is <code>?</code>, and <code>#</code> ends with four hex digits of a CRC-32 (a 16-bit check). It is not encrypted or signed: it holds what the person said, so treat it like the answers themselves. Send it in POST bodies, never in URLs, and don't log it. The service doesn't: its logs hold route, status and timing only. A damaged state is refused:</p>
{code(bad)}

<h2 id="mcp">5. AI assistants (MCP)</h2>
<p>The same engine speaks the Model Context Protocol over Streamable HTTP at <code>{BASE}/mcp</code>: the stateless 2026-07-28 revision, and the handshake revisions 2025-11-25, 2025-06-18 and 2025-03-26. JSON responses, no SSE stream, no sessions. The assistant does the talking and turns the person's words into facts; the engine decides what to ask and when to stop. In a registered test on 200 simulated households, Claude Haiku made 41% fewer costly mistakes with these tools than without them (unweighted; the <a href="https://github.com/egehazar/decision-intake/blob/master/MODEL_CARD.md" target="_blank" rel="noopener">model card</a> has the caveats).</p>
<h3>Claude Code</h3>
{code("claude mcp add --transport http decisionintake " + BASE + "/mcp", "text")}
<h3>Claude.ai and other clients</h3>
<p>Add a custom connector with the URL <code>{BASE}/mcp</code>. The server's instructions tell the assistant how to run an intake. Tools (all read-only: each is a pure function of its arguments):</p>
<ul>{tools}</ul>
<h3>On the wire, 2026-07-28</h3>
<p>No handshake. Every request names its revision in <code>_meta</code> and mirrors the method (and, for <code>tools/call</code>, the tool name) in headers that must match the body:</p>
{code(headers, "text")}
{code(ex["mcp_call_body"])}
{code(call_view)}
<p>Start with <code>server/discover</code> for the supported revisions and instructions. Errors: <code>-32022</code> (unsupported revision, with the supported list), <code>-32020</code> (header and body disagree), <code>-32602</code> (missing <code>_meta</code>, bad arguments, unknown tool).</p>
<h3>On the wire, handshake revisions</h3>
{code(ex["mcp_legacy"])}
<p>Then <code>tools/list</code> and <code>tools/call</code> with an <code>MCP-Protocol-Version</code> header. No <code>Mcp-Session-Id</code> is issued: the state travels in the tool arguments.</p>

<h2 id="code">6. Code</h2>
<h3>JavaScript</h3>
{code(js, "js")}
<h3>Python, standard library only</h3>
{code(py, "python")}
<h3>curl</h3>
{code(curl, "text")}

<h2 id="rules">7. Integration checklist</h2>
<ul>
<li>Show the <code>note</code> with every result, and say the tool is not a government site and not an application.</li>
<li>Never present "doesn't look likely" or "can't tell" as a denial. Offer the official way to apply for every program, including those.</li>
<li>Present values as rough yearly ranges if eligible, never as an amount the person will receive.</li>
<li>Treat <code>hand_off</code> as advice to talk to a navigator (211 is free); it does not refer anyone.</li>
<li>Keep <code>state</code> out of URLs, analytics and logs; it is the person's answers.</li>
<li>Ask about the household as the questions word it; the engine does not ask about immigration status, which can change eligibility.</li>
</ul>

<h2 id="limits">8. Errors, limits and versions</h2>
<p>Errors are JSON: <code>{{"error": {{"code", "message"}}}}</code>. A fact that can't be read doesn't fail the call (<a href="#facts">section 2</a>).</p>
<div class="tbl"><table>
<tr><th>Status</th><th>When</th></tr>
<tr><td>200</td><td>success; MCP protocol errors are JSON-RPC errors in a 200 for the handshake revisions</td></tr>
<tr><td>202</td><td>MCP notification accepted</td></tr>
<tr><td>400</td><td><code>bad_json</code>, <code>bad_request</code>, <code>bad_state</code> (damaged state, or one written for another question set); MCP 2026-07-28 errors</td></tr>
<tr><td>403</td><td>MCP only: a browser <code>Origin</code> this server doesn't accept</td></tr>
<tr><td>404, 405</td><td>no such route; wrong method (MCP is POST only)</td></tr>
<tr><td>413, 415</td><td>body over {MAX_BODY // 1024} KB; not <code>application/json</code></td></tr>
<tr><td>500</td><td><code>internal</code>; nothing about the failure is returned, the server log keeps it</td></tr>
</table></div>
<div class="tbl"><table>
<tr><th>Limit</th><th>Value</th></tr>
<tr><td>Request body</td><td>{MAX_BODY // 1024} KB</td></tr>
<tr><td>Facts in one call</td><td>{MAX_FACTS}</td></tr>
<tr><td>State string</td><td>2,000 characters</td></tr>
<tr><td>Amounts</td><td>numbers from 0 to 1,000,000,000</td></tr>
<tr><td>Latency</td><td>about 300 ms of server time a step; the first call after a quiet spell adds about a second</td></tr>
</table></div>
<p><b>Cross-origin use.</b> The JSON API answers any origin (no credentials are involved). <code>/mcp</code> serves clients that send no <code>Origin</code> (server-side assistants and tools) and browsers from this site or localhost; ask in the repository if a browser client elsewhere needs it.</p>
<p><b>Versions.</b> The engine is {VERSION}; <code>/healthz</code> reports it with the commit and the model and rules hashes. A state stays readable across deployments while the questions and their options are unchanged. A release that changes them makes older states <code>bad_state</code>: start again (the public screener does this for you). Breaking changes to the API move the major version. This is a free research service with no uptime guarantee; it is monitored every 30 minutes.</p>
<p>Full schema: <a href="openapi.json">openapi.json</a>. Evaluation, decisions and their tests: <a href="https://github.com/egehazar/decision-intake" target="_blank" rel="noopener">the repository</a>.</p>
</main>
</body>
</html>
"""
    open(OUT, "w", encoding="utf-8").write(page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
