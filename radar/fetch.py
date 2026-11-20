#!/usr/bin/env python3
"""Collect recent papers from Hugging Face Daily Papers and arXiv -> docs/data/papers.json (stdlib only).

Papers that are trending on Hugging Face but have no linked code are enriched with two extra signals:
  * GitHub repositories whose README mentions the arXiv id (community reimplementations)
  * Hugging Face models tagged with the paper
and get a `status`: official | community | open | unchecked.  "open" means we found nothing; it is a
lead to verify by hand, not proof that no code exists.
"""
import json, os, re, sys, time, datetime as dt, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "papers.json"
DAYS = 14
ARXIV_CATS = ["cs.CL", "cs.LG", "cs.AI", "cs.CV"]
MIN_UPVOTES = 5          # only enrich papers with some traction
MAX_ENRICH = 40          # per run (GitHub search is rate limited)
REENRICH_HOURS = 24
ENRICH_VERSION = 2       # bump when the enrichment logic changes so cached results are recomputed
# Repos that merely list many papers (daily arXiv trackers, awesome-lists) are not implementations.
AGGREGATOR = re.compile(r"arxiv|daily|awesome|paper|digest|survey|reading|list|feed|tracker|news|summar|weekly|note|collection", re.I)
GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
UA = {"User-Agent": "paper-radar/0.2 (open-source research digest)"}
PROMISE = re.compile(r"(code|models?|weights|data(sets?)?)( and \w+)? (will|to) be (made )?(publicly )?(released|available|open-?sourced)"
                     r"|coming soon|we will (open-?source|release|make)", re.I)

def get(url, tries=3, headers=None):
    for i in range(tries):
        try:
            h = dict(UA, **(headers or {}))
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            print(f"warn {url[:90]}: HTTP {e.code}", file=sys.stderr)
            if e.code in (403, 404, 422):
                return None
            time.sleep(3 * (i + 1))
        except Exception as e:
            print(f"warn {url[:90]}: {e}", file=sys.stderr)
            time.sleep(3 * (i + 1))
    return None

def hf_papers():
    out = {}
    today = dt.date.today()
    for d in range(DAYS):
        day = (today - dt.timedelta(days=d)).isoformat()
        raw = get(f"https://huggingface.co/api/daily_papers?date={day}&limit=100")
        if not raw:
            continue
        for it in json.loads(raw):
            p = it.get("paper", {})
            pid = p.get("id")
            if not pid:
                continue
            out[pid] = {
                "id": pid, "title": it.get("title") or p.get("title"),
                "summary": it.get("summary") or p.get("summary", ""),
                "authors": [a.get("name") for a in p.get("authors", [])][:8],
                "date": (p.get("publishedAt") or it.get("publishedAt") or "")[:10],
                "upvotes": p.get("upvotes", 0), "code": p.get("githubRepo"),
                "stars": p.get("githubStars"), "project": p.get("projectPage"),
                "tldr": p.get("ai_summary"), "source": "hf",
                "url": f"https://huggingface.co/papers/{pid}",
            }
    return out

NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
GH = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+")

def arxiv_papers():
    out = {}
    for cat in ARXIV_CATS:
        q = urllib.parse.urlencode({"search_query": f"cat:{cat}", "sortBy": "submittedDate",
                                    "sortOrder": "descending", "max_results": 150})
        raw = get(f"https://export.arxiv.org/api/query?{q}")
        time.sleep(3)  # arXiv API etiquette
        if not raw:
            continue
        for e in ET.fromstring(raw).findall("a:entry", NS):
            aid = re.sub(r"v\d+$", "", e.findtext("a:id", "", NS).rsplit("/abs/", 1)[-1])
            summ = " ".join(e.findtext("a:summary", "", NS).split())
            comment = e.findtext("ar:comment", "", NS) or ""
            m = GH.search(summ + " " + comment)
            out[aid] = {
                "id": aid, "title": " ".join(e.findtext("a:title", "", NS).split()),
                "summary": summ,
                "authors": [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)][:8],
                "date": e.findtext("a:published", "", NS)[:10], "upvotes": 0,
                "code": m.group(0) if m else None, "stars": None, "project": None, "tldr": None,
                "source": "arxiv", "url": f"https://arxiv.org/abs/{aid}",
            }
    return out

# ---- enrichment: "is there really no code yet?" ---------------------------------------------------
def hf_models(pid):
    raw = get(f"https://huggingface.co/api/models?filter=arxiv:{pid}&limit=20")
    if raw is None:
        return None
    ms = json.loads(raw)
    return [{"id": m["id"], "downloads": m.get("downloads", 0), "likes": m.get("likes", 0)} for m in ms[:3]], len(ms)

def gh_repos(pid):
    """Community repos mentioning the arXiv id. Returns None if GitHub search is unavailable/rate-limited."""
    h = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        h["Authorization"] = f"Bearer {GH_TOKEN}"
    q = urllib.parse.quote(f'"{pid}" in:readme')      # quoted: unquoted ids match unrelated numbers
    raw = get(f"https://api.github.com/search/repositories?q={q}&sort=stars&per_page=10", tries=1, headers=h)
    if raw is None:
        return None
    items = [i for i in json.loads(raw).get("items", [])
             if not AGGREGATOR.search(i["full_name"].split("/", 1)[1] + " " + (i.get("description") or ""))]
    return [{"name": i["full_name"], "stars": i["stargazers_count"],
             "license": (i.get("license") or {}).get("spdx_id"), "url": i["html_url"]} for i in items[:3]]

def enrich(papers, previous):
    now = dt.datetime.now(dt.timezone.utc)
    pause = 2.2 if GH_TOKEN else 6.5     # GitHub search: 30/min with a token, 10/min without
    gh_ok = True
    cands = sorted((p for p in papers if p["source"] == "hf" and not p["code"] and p["upvotes"] >= MIN_UPVOTES),
                   key=lambda p: -p["upvotes"])[:MAX_ENRICH]
    for p in papers:
        p["status"] = "official" if p["code"] else "unchecked"
        p["promised"] = bool(PROMISE.search(p["summary"]))
    for p in cands:
        old = previous.get(p["id"], {})
        if old.get("enriched_at") and old.get("gh_checked") and old.get("enrich_version") == ENRICH_VERSION and \
           (now - dt.datetime.fromisoformat(old["enriched_at"])).total_seconds() < REENRICH_HOURS * 3600:
            for k in ("gh_repos", "hf_models", "hf_model_count", "enriched_at", "gh_checked", "enrich_version"):
                p[k] = old.get(k)
        else:
            m = hf_models(p["id"])
            g = gh_repos(p["id"]) if gh_ok else None
            if g is None:
                gh_ok = False
            p["hf_models"], p["hf_model_count"] = m if m else ([], 0)
            p["gh_repos"] = g or []
            p["gh_checked"] = g is not None
            p["enriched_at"] = now.isoformat(timespec="seconds")
            p["enrich_version"] = ENRICH_VERSION
            if gh_ok:
                time.sleep(pause)
        found = bool(p.get("gh_repos") or p.get("hf_model_count"))
        p["status"] = "community" if found else ("open" if p.get("gh_checked") else "unchecked")
    return len(cands)

def main():
    previous = {}
    if OUT.exists():
        try:
            previous = {p["id"]: p for p in json.loads(OUT.read_text()).get("papers", [])}
        except Exception:
            pass
    papers = arxiv_papers()
    papers.update(hf_papers())  # HF data (upvotes, code, tldr) wins on overlap
    cutoff = (dt.date.today() - dt.timedelta(days=DAYS)).isoformat()
    items = sorted((p for p in papers.values() if p["date"] >= cutoff),
                   key=lambda p: (p["date"], p["upvotes"]), reverse=True)
    if not items:
        sys.exit("no papers fetched; leaving existing data untouched")
    n = enrich(items, previous)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "count": len(items), "papers": items}, ensure_ascii=False))
    st = {}
    for p in items:
        st[p["status"]] = st.get(p["status"], 0) + 1
    print(f"wrote {len(items)} papers; enriched {n}; status {st}")

if __name__ == "__main__":
    main()
