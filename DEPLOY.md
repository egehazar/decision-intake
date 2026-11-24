# Deploying DecisionIntake

The service is one WSGI app (`src/decisionintake/web.py`) plus static files (`public/`). Vercel runs it
as is: `vercel.json` sends `/v1/*`, `/mcp`, `/healthz` and `/openapi.json` to one Python function
(`api/index.py`) and serves the screener from the CDN. It keeps nothing between requests, so there is
no database, no secret and no migration.

## Production (D38)

Public since 2026-11-23 20:21 UTC at https://decision-intake.vercel.app: Vercel Authentication now
protects preview deployments only (Settings → Deployment Protection → Standard, previews). Every push to
master deploys to production. `.github/workflows/monitor.yml` runs after every push (it first waits until
`/healthz` reports that commit) and on a 30-minute schedule: the page, health with the rules, a step, a
whole screening, the deployed model and rules hashes against the repository, and the MCP connector in both
protocol eras; a failure fails the run and GitHub emails the owner. The monitor reached GitHub only on
2026-11-24, and its schedule had not fired in its first four hours; the push trigger is the reliable one.

## Listing in the MCP Registry

`server.json` describes the hosted connector (`io.github.egehazar/decision-intake`, Streamable HTTP at
https://decision-intake.vercel.app/mcp). `.github/workflows/publish-mcp.yml` publishes it when a version
tag is pushed: it checks the endpoint answers MCP, verifies the pinned `mcp-publisher` release's signature,
validates `server.json`, logs in with GitHub OIDC (no secret) and publishes.

```bash
git tag v1.0.0 && git push origin v1.0.0
```

To list a new version later: bump `version` in `server.json`, commit, tag `v<version>`, push the tag.

## Setting up a deployment (how this one was staged, D36)

1. On vercel.com: **Add New → Project → Import** the repository. Framework preset: Other. No build
   command, no environment variables.
2. A new project starts with standard protection, which leaves the production address public. To stage
   privately first, set **Settings → Deployment Protection → Vercel Authentication: All Deployments**
   before the first build; this project was staged that way (D36) and opened to the public in D38.
3. Check the deployment (replace `$URL`):

```bash
curl -s $URL/healthz                       # ok, version, 14 statutory limits, commit, bundle/rules hashes
curl -s -X POST $URL/v1/step -H 'Content-Type: application/json'   -d '{"facts":[{"question":"state","choice":"TX"},{"question":"household_size","choice":"2"}]}'
curl -s -X POST $URL/mcp -H 'Content-Type: application/json'   -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'      # three tools
```

   With protection on, pass a protection-bypass token as the `x-vercel-protection-bypass` header.
4. Open `$URL/` and complete a screening; change an answer; clear the answers.

## Release gate (D36-D38)

From GPT's round-6 review; each item is checked in DECISIONS.md or the model card. The two open items
were waived when the site went public (D38): the page labels the Spanish as AI-assisted.

- [x] D32 passed: the rules cost nothing on 2,000 fresh households
- [x] Every rule's exceptions explained (`results/contradiction_ledger.md`) and the eligible households it suppresses counted
- [x] Thin-support profiles (few training households like the person) handled: "can't tell", not "unlikely" (D33, D35)
- [x] Rules year, engine version and benefit timing shown on the page
- [x] No definitive ineligibility claim; an official application path for every programme, including unlikely ones
- [x] Non-government, informational-use disclaimer; privacy notice
- [x] Keyboard and phone width checked: a screening done with real key presses only on the live site (Tab,
      arrows, typing, Enter; focus moves to each new question), axe-core clean in light and dark, no
      overflow at 375 px
- [x] Screen-reader pass on the live site, through Chrome's accessibility tree (what NVDA, JAWS, Narrator
      and VoiceOver read): landmarks and skip link; focus lands on each new question, announced with its
      number and what it helps decide; options are labelled radio buttons grouped by the question;
      errors are alerts; results focus their heading; every link says where it goes and that it opens a
      new tab. Fixed on the way: the badge symbol is no longer read ("check mark Likely"), the apply
      buttons announce the new tab, the question's number and purpose are tied to it.
- [ ] Optional: a two-minute listen with a real screen reader (Narrator: Win+Ctrl+Enter) to hear it
- [ ] Spanish: a native speaker reads the Spanish screens (the translation is AI-assisted; the page says so)
- [x] No facts in URLs, logs or analytics (state travels in POST bodies only)
- [x] Request limits in the app (64 KB body, 40 facts, 2,000-character state); the Hobby plan has no
      billing to run up (usage limits cap it)
- [x] Rate limit at the edge (Vercel Firewall, rule "Rate limit /v1/* and /mcp"): 120 requests a minute per
      IP address on the compute routes, 429 beyond it; checked with a 125-request burst (120 x 200, 5 x 429),
      the page and /healthz unaffected (DECISIONS D39, launch audit)
- [x] Cold and warm latency measured on the real deployment: engine boot 1.1 s, /v1/step about 300 ms
      median server time (89 ms to 1.1 s); on a desktop CPU 109-125 ms median (D31, D36)
- [x] Rollback rehearsed (D36): production moved back to the previous deployment in seconds, served it,
      and was promoted forward again

## When something breaks

- **The monitor fails** (GitHub emails the owner): open the failed run to see which step. `/healthz`
  down or 5xx: check Vercel → Deployments → the production deployment → Logs (or the Vercel connector's
  runtime logs) for the traceback the service writes to stderr; if the latest deployment is the cause,
  roll back (below). Only the MCP steps failing: a protocol change; run `pytest tests/test_contract.py`
  against the deployed commit.
- **`/healthz` shows the wrong commit or hashes**: the deployment is not the commit you meant; promote
  the right one.
- **Hobby usage limits reached**: Vercel pauses the project instead of billing. The screener and API are
  down until the limit resets or the plan changes; nothing is lost, because nothing is stored.

## Rollback

Every deploy is immutable on Vercel. To roll back: **Deployments → the last good deployment → Promote**
(or `vercel rollback`). After a rollback Vercel stops moving production to new pushes until a deployment
is promoted again, so finish with **Promote** on the deployment you want live. The engine, its rules and the screener ship together in one deployment, so a
rollback restores all three at once; `/healthz` shows which version is live.

## Running it elsewhere

```bash
pip install numpy
PYTHONPATH=src python -m decisionintake.web --host 0.0.0.0 --port 8080
```

Any WSGI server works (`gunicorn 'decisionintake.web:create_app()'`). Browsers may call `/mcp` from the
same origin and from localhost; set `DI_ALLOWED_ORIGINS` (comma-separated, or `*`) to allow others.
Requests without an Origin header (server-side MCP clients) are always served.
