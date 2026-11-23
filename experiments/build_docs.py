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


def examples():
    app = App(static_dir=None, log=None)
    start = call(app, "/v1/step", {})
    facts = [{"question": "children", "choice": "2"}, {"question": "state", "choice": "TX"},
             {"question": "earned_income", "amount": {"value": 2400, "period": "month", "approximate": True}}]
    step = call(app, "/v1/step", {"state": start["state"], "facts": facts})
    answers = {"household_size": "3", "child_under5": "no", "infant_or_pregnant": "no", "senior": "no", "disabled": "no",
               "married": "no", "self_employed": "no", "employer_insurance": "no", "college": "no", "tenure": "rent",
               "social_security": "none", "other_income": "none", "housing_cost": "$1,000 to $1,500", "childcare": "none",
               "assets": "under $2,000"}
    state, nxt = step["state"], step["next"]
    while nxt["action"] == "ask":
        q = nxt["question"]["id"]
        r = call(app, "/v1/step", {"state": state, "facts": [{"question": q, "choice": answers[q]}]})
        state, nxt = r["state"], r["next"]
    bad = call(app, "/v1/step", {"state": state[:-1] + ("0" if state[-1] != "0" else "1")})
    return start, facts, step, state, nxt, bad


def code(obj, lang="json"):
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False)
    return f'<pre tabindex="0"><code class="{lang}">{html.escape(text)}</code></pre>'


def main():
    start, facts, step, state, finish, bad = examples()
    start_view = {"state": start["state"], "readings": [], "next": {
        "action": "ask", "question": start["next"]["question"], "why": start["next"]["why"], "progress": start["next"]["progress"]}}
    step_view = {"state": step["state"], "readings": step["readings"], "next": {
        k: step["next"][k] for k in ("action", "question", "why")}}
    finish_view = {"action": finish["action"], "recommend": finish["recommend"][:2] + ["…"], "unsure": finish.get("unsure"),
                   "progress": finish["progress"], "note": finish["note"]}
    tools = "".join(f"<li><code>{t['name']}</code>: {html.escape(t['description'])}</li>" for t in TOOLS_STATELESS)
    js = """const BASE = "{BASE}";
async function step(state, facts) {
  const r = await fetch(BASE + "/v1/step", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({state, facts})
  });
  if (!r.ok) throw new Error((await r.json()).error.message);
  return r.json();
}

let {state, next} = await step(undefined, []);        // start
while (next.action === "ask") {
  const answer = await askThePerson(next.question);  // your UI: a choice, an amount, or "don't know"
  ({state, next} = await step(state, [answer]));
}
show(next.recommend, next.unsure, next.action === "hand_off");"""
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
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>DecisionIntake Developer Guide</title>
<meta name="description" content="Plug the DecisionIntake benefits-screening engine into a screener or an AI assistant: the JSON API, the state string, MCP.">
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
<header><div class="wrap"><a href="./">Quick Benefits Check</a><span><a href="openapi.json">openapi.json</a> · <a href="https://github.com/egehazar/decision-intake" target="_blank" rel="noopener">source</a></span></div></header>
<main class="wrap">
<h1>Developer guide</h1>
<p class="lede">DecisionIntake decides what to ask next in a US benefits screening, how far to trust each answer, and when it knows enough to recommend programmes or should hand the case to a person. Your product keeps the conversation; the engine keeps the decision. Engine {VERSION}; every example below is a real response from this service.</p>

<nav class="toc" aria-label="Contents"><b>Contents</b><ol>
<li><a href="#loop">The loop</a></li><li><a href="#facts">Facts</a></li><li><a href="#responses">Responses</a></li>
<li><a href="#state">The state string</a></li><li><a href="#mcp">AI assistants (MCP)</a></li><li><a href="#code">Code</a></li>
<li><a href="#limits">Errors and limits</a></li></ol></nav>

<h2 id="loop">1. The loop</h2>
<p>One endpoint does the work. Send what you know, get back the next step, repeat until the step is <code>finish</code> or <code>hand_off</code>. The service keeps nothing between calls: every response carries a <code>state</code> string, and you send it back with the next call.</p>
<div class="tbl"><table>
<tr><th>Call</th><th>Body</th><th>Returns</th></tr>
<tr><td><code>POST /v1/step</code></td><td><code>{{"state"?, "facts"?}}</code></td><td>the new <code>state</code>, how each fact was read, and <code>next</code>: ask, finish or hand_off</td></tr>
<tr><td><code>POST /v1/assessment</code></td><td><code>{{"state"}}</code></td><td>every programme's probability, tier and yearly value range; the facts so far; how well the data covers this household</td></tr>
<tr><td><code>GET /v1/questions</code></td><td>-</td><td>the ten programmes and all 18 questions with their options</td></tr>
<tr><td><code>POST /mcp</code></td><td>JSON-RPC</td><td>the same engine as tools for AI assistants</td></tr>
</table></div>
<h3>Start</h3>
{code({"facts": []})}
{code(start_view)}

<h2 id="facts">2. Facts</h2>
<p>Record anything the person said, asked or not: a fact for a question that already has one replaces it (a correction). Three kinds:</p>
<div class="tbl"><table>
<tr><th>Kind</th><th>Example</th><th>Notes</th></tr>
<tr><td>choice</td><td><code>{{"question": "children", "choice": "2"}}</code></td><td>the option's text; numbers map onto ranges like "4 or more"; true/false onto yes/no</td></tr>
<tr><td>amount</td><td><code>{{"question": "earned_income", "amount": {{"value": 2400, "period": "month", "approximate": true}}}}</code></td><td>as said: a value or <code>low</code>/<code>high</code>, a period (year, month, week, none); the engine does the arithmetic and spreads approximate amounts over neighbouring ranges</td></tr>
<tr><td>don't know</td><td><code>{{"question": "assets", "dont_know": true}}</code></td><td>no evidence; the belief is unchanged</td></tr>
</table></div>
<h3>Three facts at once</h3>
{code({"state": start["state"], "facts": facts})}
{code(step_view)}
<p><code>why.worth_per_year</code> is the expected cost of mistakes, in dollars a year, that the next question saves; it is asked because that is more than <code>question_cost</code>. <code>mostly_settles</code> says which programmes it is about, useful for telling the person why you ask.</p>

<h2 id="responses">3. Responses</h2>
<p>When no remaining question is worth asking, <code>next.action</code> is <code>finish</code>, or <code>hand_off</code> when the doubt left is still costly enough that a person should review.</p>
{code(finish_view)}
<div class="tbl"><table>
<tr><th>Field</th><th>Meaning</th></tr>
<tr><td><code>recommend</code></td><td>programmes worth applying for; <code>tier</code> is <code>likely</code> (probability at least 0.5) or <code>worth checking</code>; <code>value_if_eligible_per_year</code> a range in dollars</td></tr>
<tr><td><code>unsure</code></td><td>programmes the engine won't call from these answers: few training households look like this person and the programme keeps a real chance. Show them as "we couldn't tell, you can still apply", never as unlikely</td></tr>
<tr><td><code>progress.support</code></td><td>how many of the 48,657 training households fit these answers before the support floor; under 10 is <code>thin</code></td></tr>
<tr><td><code>note</code></td><td>what this is and isn't; show it</td></tr>
</table></div>
<p class="note">A screening estimate from PolicyEngine-US's 2026 rules, not an eligibility decision. Always offer the official way to apply, including for programmes that don't look likely. The <a href="https://github.com/egehazar/decision-intake/blob/master/MODEL_CARD.md" target="_blank" rel="noopener">model card</a> says where it is wrong.</p>

<h2 id="state">4. The state string</h2>
<p>The facts so far, readable, with a check so a mangled copy is refused instead of misread:</p>
{code(state, "text")}
<p>Choices are option text (spaces as <code>_</code>), amounts are <code>~</code>-prefixed when approximate with <code>..</code> for ranges and <code>/period</code>, don't-know is <code>?</code>, and <code>#</code> ends with four hex digits of a CRC. It is not encrypted or signed: it holds what the person said, so treat it like the answers themselves. Send it in POST bodies, never in URLs, and don't log it. The service doesn't: its logs hold route, status and timing only.</p>

<h2 id="mcp">5. AI assistants (MCP)</h2>
<p>The same engine speaks the Model Context Protocol over Streamable HTTP at <code>{{BASE}}/mcp</code> (the stateless 2026-07-28 revision, and 2025-11-25 back to 2024-11-05 for clients that still use the handshake). The assistant does the talking and turns the person's words into facts; the engine decides what to ask and when to stop. In a registered test, Claude made 41% fewer costly mistakes with these tools than without them.</p>
<h3>Claude Code</h3>
{code("claude mcp add --transport http decisionintake {BASE}/mcp", "text")}
<h3>Claude.ai and other clients</h3>
<p>Add a custom connector with the URL <code>{{BASE}}/mcp</code>. Tools:</p>
<ul>{tools}</ul>

<h2 id="code">6. Code</h2>
<h3>JavaScript</h3>
{code(js, "js")}
<h3>Python, standard library only</h3>
{code(py, "python")}
<h3>curl</h3>
{code(curl, "text")}

<h2 id="limits">7. Errors and limits</h2>
<p>Errors are JSON: <code>{{"error": {{"code", "message"}}}}</code>. A fact that can't be read doesn't fail the call: its reading carries an <code>error</code> saying how to fix it, and valid facts in the same call still go in.</p>
{code(bad)}
<div class="tbl"><table>
<tr><th>Limit</th><th>Value</th></tr>
<tr><td>Request body</td><td>{MAX_BODY // 1024} KB</td></tr>
<tr><td>Facts in one call</td><td>{MAX_FACTS}</td></tr>
<tr><td>State string</td><td>2,000 characters</td></tr>
<tr><td>Amounts</td><td>numbers from 0 to 1,000,000,000</td></tr>
</table></div>
<p>Full schema: <a href="openapi.json">openapi.json</a>. Evaluation, decisions and their tests: <a href="https://github.com/egehazar/decision-intake" target="_blank" rel="noopener">the repository</a>.</p>
</main>
<script>
"use strict";
for (const el of document.querySelectorAll("pre code, p code"))
  el.textContent = el.textContent.split("{{BASE}}").join(location.origin);
</script>
</body>
</html>
"""
    open(OUT, "w", encoding="utf-8").write(page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
