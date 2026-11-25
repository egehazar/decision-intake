"""Where to apply, by state (D53): each state's own page for SNAP, Medicaid and CHIP, TANF and the state tax credits.

    python experiments/apply_links.py sources   rebuild experiments/apply_links_sources/candidates.json from the
                                                federal directories (Internet Archive copies; the live FNS and
                                                Medicaid.gov pages refuse requests from outside the US)
    python experiments/apply_links.py check     fetch every candidate, keep the first that passes for each state
                                                and programme, write src/decisionintake/apply_links.json
                                                [--report PATH] also writes a Markdown report (the links workflow
                                                puts it in the run summary) [--dry] writes nothing to src/

A candidate passes when it answers 200, is not an error, captcha, block or maintenance page, and, if it redirects
to a different page, the page it lands on still names the programme (an old link that now lands on an agency's
home page or an unrelated page fails). Run `check` from the US: many state sites refuse other countries, so a
check from elsewhere passes far fewer links (it never passes a bad one). A state and programme with no passing
candidate has no entry, and the screener keeps the national page.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import gzip
import html
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "apply_links_sources")
CANDIDATES = os.path.join(SRC_DIR, "candidates.json")
REVIEW = os.path.join(SRC_DIR, "review.json")  # links read and excluded by hand, each with its reason
OUT = os.path.join(HERE, "..", "src", "decisionintake", "apply_links.json")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
GZIP_MAGIC = bytes([0x1F, 0x8B])

STATES = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
          "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
          "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
          "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
          "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
          "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
          "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
          "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
          "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
          "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}

SOURCES = {
    "snap": {"name": "USDA FNS, SNAP State Directory of Resources",
             "url": "https://www.fns.usda.gov/snap/state-directory"},
    "medicaid": {"name": "HealthCare.gov, Medicaid & CHIP: apply through your state Medicaid agency; then "
                         "Medicaid.gov, Where can people get help with Medicaid & CHIP",
                 "url": "https://www.healthcare.gov/medicaid-chip/getting-medicaid-chip/",
                 "also": "https://www.medicaid.gov/about-us/where-can-people-get-help-medicaid-chip/index.html"},
    "tanf": {"name": "HHS ACF Office of Family Assistance, Help for families (state TANF programs)",
             "url": "https://www.acf.hhs.gov/ofa/map/about/help-families"},
    "state_tax": {"name": "Federation of Tax Administrators, member agencies", "url": "https://taxadmin.org/fta-members/"},
}

# Words a page about the programme carries, for links that redirect elsewhere (state names for programmes vary).
KEYWORDS = {
    "snap": r"snap|food|nutrition|calfresh|foodshare|basic food|3squares|apply for benefits|public benefits",
    "medicaid": r"medicaid|medi-cal|coverage|husky|masshealth|soonercare|tenncare|kancare|ahcccs|apple health|"
                r"med-?quest|apply|application|chip|benefits|health[- ]care",
    "tanf": r"tanf|cash|temporary assistance|works|work first|family investment|reach up|fitap|"
            r"family independence|families first|calworks|k-tap|transitional",
    "state_tax": r"revenue|tax|treasury|comptroller|franchise",
}
BAD_TITLE = re.compile(r"not found|404|error|captcha|bot manager|blocked|access denied|unauthori[sz]ed|"
                       r"maintenance|unavailable|forbidden|just a moment|attention required|events?", re.I)
BAD_URL = re.compile(r"/maint|/error|checkip|perfdrive|captcha|/404|/event/", re.I)
RENEWAL = re.compile(r"renew|keep-your|staycovered|staywell|medicaid-changes", re.I)
# Portals for people already enrolled (Medicaid.gov lists some as "Enrollment"): not where a new applicant starts.
MEMBER_PORTAL = re.compile(r"memberpublic|sp_beneficiary|/member/|beneficiary", re.I)
# A short page that says it is missing, though it answers 200 (a "soft 404").
SOFT_404 = re.compile(r"url is invalid|pointer does not exist|page (?:you requested |you are looking for )?"
                      r"(?:was |could )?not (?:be )?found|no longer available|has been (?:moved|removed)|"
                      r"page not found|error 404|404 error", re.I)  # not a bare 404: Atlanta numbers start 404


# ---------------------------------------------------------------------------------------------- sources
def _get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or b.startswith(GZIP_MAGIC):
                    b = gzip.decompress(b)
                return b.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - the archive drops connections now and then
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"could not fetch {url}")


def _archived(url, near="20261120"):
    """The raw page of the archive's snapshot closest to `near`, and that snapshot's timestamp. The availability
    API sometimes answers with nothing for a page it holds, so ask it twice, then let the archive redirect."""
    for _ in range(2):
        meta = json.loads(_get(f"https://archive.org/wayback/available?url={url}&timestamp={near}"))
        snap = meta.get("archived_snapshots", {}).get("closest")
        if snap:
            return _get(f"http://web.archive.org/web/{snap['timestamp']}id_/{url}"), snap["timestamp"]
        time.sleep(2)
    req = urllib.request.Request(f"http://web.archive.org/web/{near[:4]}id_/{url}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        ts = re.search(r"/web/(\d{14})", r.geturl())
        b = r.read()
    b = gzip.decompress(b) if b.startswith(GZIP_MAGIC) else b
    return b.decode("utf-8", "replace"), ts.group(1) if ts else near[:4]


def build_sources():
    cands = {ab: {} for ab in STATES.values()}
    snapshots = {}
    # SNAP: one directory page per state, each with "State SNAP Website" and "State Applications".
    index, ts = _archived(SOURCES["snap"]["url"])
    snapshots["snap_index"] = ts
    entries = re.findall(r'<li><a href="/snap-directory-entry/([a-z-]+)">([^<]+)</a></li>', index)
    for slug, name in dict((s, n) for s, n in entries).items():
        if name not in STATES:
            continue
        page, ts = _archived(f"https://www.fns.usda.gov/snap-directory-entry/{slug}")
        snapshots[f"snap_{slug}"] = ts

        def section(cls):
            i = page.find(cls)
            if i < 0:
                return []
            j = page.find("fns-contact-field", i + len(cls))
            return [html.unescape(u) for u in
                    re.findall(r'<a href="(https?://[^"]+)"', page[i:j if j > 0 else i + 5000])]
        site, apps = section("fnscontact-snap-state-snap-info"), section("fnscontact-snap-state-applications")
        cands[STATES[name]]["snap"] = [u for u in site[:1] + apps[:1] if not u.endswith(".pdf")]
        time.sleep(1)
    # Medicaid and CHIP, first: HealthCare.gov's "apply directly through your state Medicaid agency" list, read
    # live from the page's data (Gatsby page-data; the state picker is drawn in the browser from it).
    hc = _get("https://www.healthcare.gov/assets/learn/page-data/medicaid-chip/getting-medicaid-chip/page-data.json")
    snapshots["medicaid_healthcare_gov"] = dt.date.today().isoformat()
    hc_links = {}
    for text in _strings(json.loads(hc)):
        if "state_name:" not in text:
            continue
        for name, body in re.findall(r"- state_name: ([^\r\n]+)\r?\n\s+text: (.*?)(?=\r?\n\s*- state_name:|\Z)",
                                     text, re.S):
            m = re.search(r'href="(https?://[^"]+)"', body)
            if m and name.strip() in STATES:
                hc_links[STATES[name.strip()]] = html.unescape(m.group(1))
    # then Medicaid.gov's table, a block per state (agency, contacts, eligibility, enrollment).
    page, ts = _archived(SOURCES["medicaid"]["also"])
    snapshots["medicaid"] = ts
    blocks = re.split(r'<h3 id="([A-Z]{2})">', page)[1:]
    for k in range(0, len(blocks), 2):
        ab, b = blocks[k], blocks[k + 1]
        b = b[:b.find("<h3 ")] if "<h3 " in b else b[:6000]
        if ab not in cands:
            continue
        links = [(html.unescape(u), t) for u, t in re.findall(r'<a href="(https?://[^"]+)"[^>]*>([^<]+)</a>', b)]
        pick = lambda label: [u for u, t in links if t.strip() == label]  # noqa: E731
        order = [u for u in pick("Enrollment") + pick("Eligibility")
                 if not RENEWAL.search(u) and not MEMBER_PORTAL.search(u) and not u.lower().endswith(".pdf")]
        agency = [links[0][0]] if links else []
        # A Medicaid.gov page about applying first; then HealthCare.gov's agency link, which avoids the member
        # portals some Medicaid.gov "Enrollment" links point to (Alaska's, North Carolina's); then the rest.
        applying = [u for u in order if "apply" in u.lower()]
        cands[ab]["medicaid"] = list(dict.fromkeys(applying + ([hc_links[ab]] if ab in hc_links else [])
                                                   + order + agency))
    # TANF: the program's own page, then the state's general benefits site.
    page, ts = _archived(SOURCES["tanf"]["url"])
    snapshots["tanf"] = ts
    by_host = {}
    for part in re.split(r"<h3[^>]*>", page)[1:]:
        head = part[:part.find("</h3>")]
        m = re.search(r'<a href="([^"]+)"[^>]*><span>([^<]+)</span>', head)
        if not m:
            continue
        rest = part[part.find("</h3>"):part.find("</h3>") + 3000]
        gb = re.search(r'General Benefits: </span><a href="([^"]+)"', rest)
        name = html.unescape(m.group(2)).replace("\xa0", " ").strip()
        ab = next((STATES[n] for n in sorted(STATES, key=len, reverse=True) if name.lower().startswith(n.lower())),
                  None)
        by_host.setdefault(urlparse(m.group(1)).netloc.lower(), []).append(
            (ab, name, [html.unescape(m.group(1))] + ([html.unescape(gb.group(1))] if gb else [])))
    # Programmes not named after their state ("Work First New Jersey", "Colorado Works" does start with it):
    # place them by the state domain of their page.
    for host, rows in by_host.items():
        for ab, name, urls in rows:
            if ab is None:
                ab = "MA" if host.endswith("mass.gov") else next(
                    (STATES[n] for n in sorted(STATES, key=len, reverse=True) if _state_domain(host, n, STATES[n])),
                    None)
            if ab:
                cands[ab]["tanf"] = urls
    # State tax agencies: the income tax agency where a state lists several.
    page = _get(SOURCES["state_tax"]["url"])
    snapshots["state_tax"] = dt.date.today().isoformat()
    seg = page[page.find("View a full listing"):]
    agencies = [(html.unescape(t).strip(), u) for u, t in
                re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>\s*([^<]{3,90}?)\s*</a>', seg) if "taxadmin" not in u]
    prefer = {"CA": "Franchise Tax Board", "DC": None}
    for name, ab in STATES.items():
        rows = [(t, u) for t, u in agencies if name.lower() in t.lower() and "transportation" not in t.lower()
                and "motor" not in t.lower() and "safety" not in t.lower() and "city" not in t.lower()
                and "philadelphia" not in t.lower()]
        if ab in prefer and prefer[ab]:
            rows = [r for r in rows if prefer[ab] in r[0]]
        if ab == "WV":
            rows = [r for r in agencies if r[0].startswith("West Virginia")]
        elif ab == "VA":
            rows = [r for r in rows if not r[0].startswith("West")]
        elif ab == "KS":
            rows = [r for r in rows if "Arkansas" not in r[0]]
        elif ab == "WA":
            rows = [r for r in rows if "D.C." not in r[0]]
        if rows:
            cands[ab]["state_tax"] = [rows[0][1]]
    # DC's income tax is the Office of Tax and Revenue, inside the CFO's site that FTA lists.
    cands["DC"]["state_tax"] = ["https://otr.cfo.dc.gov/", "https://cfo.dc.gov/"]
    # Hand review, then a site's home page after its specific pages (an agency's front page is the last resort).
    review = json.load(open(REVIEW, encoding="utf-8"))
    for ab, progs in cands.items():
        for prog, urls in progs.items():
            skip = review.get(ab, {}).get(prog, {}).get("skip", [])
            urls = [u for u in urls if not any(k in u for k in skip)]
            progs[prog] = sorted(urls, key=lambda u: prog != "state_tax" and urlparse(u).path.strip("/") == "")
    out = {"_about": "Candidate links, best first, read from the federal directories in SOURCES; "
                     "experiments/apply_links.py check verifies them. Archive snapshot timestamps per source.",
           "sources": SOURCES, "snapshots": snapshots, "states": cands}
    with open(CANDIDATES, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    n = {p: sum(1 for s in cands.values() if s.get(p)) for p in SOURCES}
    print(f"wrote {CANDIDATES}: states with candidates {n}")


def _strings(o):
    """Every string inside a parsed JSON value."""
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)


def _state_domain(host, name, code):
    """True when `host` is that state's own domain (.state.xx.us, xx.gov, or the state's name in the domain)."""
    h = host.lower()
    squashed = name.lower().replace(" ", "")
    return (h.endswith(f".{code.lower()}.gov") or f".state.{code.lower()}.us" in h or h.endswith(f".{code.lower()}.us")
            or squashed in h.replace("-", ""))


# ---------------------------------------------------------------------------------------------- check
def fetch(url, insecure=False):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,*/*",
                                                   "Accept-Language": "en-US,en;q=0.8"})
        ctx = ssl.create_default_context()
        if insecure:
            ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            body = r.read(300_000)
            if r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            body = body.decode("utf-8", "replace")
            t = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
            title = html.unescape(re.sub(r"\s+", " ", t.group(1)).strip()) if t else ""
            text = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", body, flags=re.S | re.I)
            text = re.sub(r"\s+", " ", html.unescape(text)).strip()
            soft = len(text) < 3000 and bool(SOFT_404.search(text))
            return {"status": r.status, "final": r.geturl(), "title": title[:120], "soft_404": soft}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "final": e.url, "title": ""}
    except urllib.error.URLError as e:
        # A server that leaves out an intermediate certificate: browsers fetch it and show the page; Python's
        # check stops. Look again without the check, and say so. (Expired or mismatched certificates still fail.)
        if insecure or "unable to get local issuer certificate" not in str(e.reason):
            return {"status": 0, "final": url, "title": "", "error": f"URLError: {str(e.reason)[:100]}"}
        r = fetch(url, insecure=True)
        r["tls"] = "incomplete certificate chain (browsers complete it)"
        return r
    except Exception as e:  # noqa: BLE001
        return {"status": 0, "final": url, "title": "", "error": f"{type(e).__name__}: {str(e)[:100]}"}


def verdict(programme, url, r):
    """(passes, reason) for one fetched candidate."""
    if r["status"] != 200:
        return False, f"status {r['status']}" + (f" ({r['error']})" if r.get("error") else "")
    if BAD_TITLE.search(r["title"]) or BAD_URL.search(r["final"]):
        return False, f"error or block page: {r['title'] or r['final']}"
    if r.get("soft_404"):
        return False, f"the page says it is missing: {r['final']}"
    a, b = urlparse(url), urlparse(r["final"])
    moved = (a.netloc.lower().removeprefix("www."), a.path.rstrip("/").lower()) != \
            (b.netloc.lower().removeprefix("www."), b.path.rstrip("/").lower())
    if moved and b.path.strip("/") == "" and a.path.strip("/") != "":
        return False, f"redirects to the site's home page: {r['final']}"
    if moved and not re.search(KEYWORDS[programme], f"{r['title']} {b.path}", re.I):
        return False, f"redirects to an unrelated page: {r['final']} ({r['title']})"
    return True, "redirected, still about the programme" if moved else "ok"


def check(report=None, dry=False):
    data = json.load(open(CANDIDATES, encoding="utf-8"))
    jobs = sorted({u for s in data["states"].values() for urls in s.values() for u in urls})
    with cf.ThreadPoolExecutor(12) as ex:
        res = dict(zip(jobs, ex.map(fetch, jobs)))
    today = dt.date.today().isoformat()
    links, rows, tally = {}, [], {p: [0, 0] for p in SOURCES}
    for ab, progs in sorted(data["states"].items()):
        for prog, urls in progs.items():
            chosen, tried = None, []
            for u in urls:
                ok, why = verdict(prog, u, res[u])
                tried.append((u, ok, why))
                if ok:
                    chosen = {"url": res[u]["final"], "title": res[u]["title"], "checked": today}
                    break
            tally[prog][0 if chosen else 1] += 1
            if chosen:
                links.setdefault(ab, {})[prog] = chosen
            rows.append((ab, prog, chosen, tried))
    out = {"_about": "Where to apply in each state: the first candidate (experiments/apply_links_sources) that "
                     "passed experiments/apply_links.py check, per state and programme. CHIP uses the Medicaid "
                     "link (one application covers both, 42 CFR 435.907). No entry: the national page is shown.",
           "sources": data["sources"], "checked": today, "links": links}
    if not dry:
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
    summary = ", ".join(f"{p} {a}/{a + b}" for p, (a, b) in tally.items())
    print(f"checked {len(jobs)} links on {today}; states with a passing link: {summary}")
    if report:
        lines = [f"## Where-to-apply links, checked {today}", "", f"States with a passing link: {summary}.", "",
                 "| State | Programme | Chosen | Page title | Failed candidates |", "|---|---|---|---|---|"]
        for ab, prog, chosen, tried in rows:
            failed = "<br>".join(f"{u} ({why})" for u, ok, why in tried if not ok)
            lines.append(f"| {ab} | {prog} | {chosen['url'] if chosen else '-'} | "
                         f"{(chosen or {}).get('title', '').replace('|', '/')} | {failed.replace('|', '/')} |")
        lines += ["", "<details><summary>apply_links.json</summary>", "", "```json",
                  json.dumps(out, indent=1, ensure_ascii=False), "```", "</details>"]
        with open(report, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return out


def compare(served_path):
    """After `check`: exit 1 if a link the service serves (the committed file, copied to served_path before the
    check rewrote it) no longer has any passing link. New links are reported, not failed on."""
    served = json.load(open(served_path, encoding="utf-8"))["links"]
    now = json.load(open(OUT, encoding="utf-8"))["links"]
    lost = [(ab, p, v["url"]) for ab, ps in served.items() for p, v in ps.items() if p not in now.get(ab, {})]
    gained = [(ab, p) for ab, ps in now.items() for p in ps if p not in served.get(ab, {})]
    moved = [(ab, p, v["url"], now[ab][p]["url"]) for ab, ps in served.items() for p, v in ps.items()
             if p in now.get(ab, {}) and now[ab][p]["url"] != v["url"]]
    for ab, p, u in lost:
        print(f"LOST {ab} {p}: {u} no longer passes and no other candidate does")
    for ab, p, old, new in moved:
        print(f"MOVED {ab} {p}: {old} -> {new}")
    print(f"{len(gained)} state and programme pairs could gain a link: "
          + ", ".join(f"{ab} {p}" for ab, p in gained[:60]))
    if lost:
        sys.exit(1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sources":
        build_sources()
    elif cmd == "compare":
        compare(sys.argv[2])
    elif cmd == "check":
        rep = sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else None
        check(rep, dry="--dry" in sys.argv)
    else:
        print(__doc__)
        sys.exit(2)
