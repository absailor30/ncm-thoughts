#!/usr/bin/env python3
"""Collect recent papers from Hugging Face Daily Papers and arXiv -> docs/data/papers.json (stdlib only)."""
import json, re, sys, time, datetime as dt, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "papers.json"
DAYS = 14
ARXIV_CATS = ["cs.CL", "cs.LG", "cs.AI", "cs.CV"]
UA = {"User-Agent": "paper-radar/0.1 (open-source research digest)"}

def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception as e:
            print(f"warn {url}: {e}", file=sys.stderr)
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

def main():
    papers = arxiv_papers()
    papers.update(hf_papers())  # HF data (upvotes, code, tldr) wins on overlap
    cutoff = (dt.date.today() - dt.timedelta(days=DAYS)).isoformat()
    items = sorted((p for p in papers.values() if p["date"] >= cutoff),
                   key=lambda p: (p["date"], p["upvotes"]), reverse=True)
    if not items:
        sys.exit("no papers fetched; leaving existing data untouched")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "count": len(items), "papers": items}, ensure_ascii=False))
    print(f"wrote {len(items)} papers")

if __name__ == "__main__":
    main()
