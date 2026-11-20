#!/usr/bin/env python3
"""Hacker News stories and brand-new GitHub repos about AI -> docs/data/extras.json (stdlib only)."""
import datetime as dt, json, os, sys, time, urllib.parse, urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "extras.json"
DAYS = 7
GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
UA = {"User-Agent": "paper-radar/0.2 (open-source research digest)"}
HN_QUERIES = ["LLM", "language model", "diffusion model", "transformer", "open-source model", "arxiv", "fine-tuning", "AI agent"]
GH_TOPICS = ["llm", "language-model", "diffusion", "machine-learning", "ai-agents", "transformers", "rag"]

def get_json(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=dict(UA, **(headers or {}))), timeout=30) as r:
            return json.load(r)
    except Exception as e:
        print(f"warn {url[:90]}: {e}", file=sys.stderr)
        return None

def hacker_news():
    cutoff = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DAYS)).timestamp())
    seen = {}
    for q in HN_QUERIES:
        url = ("https://hn.algolia.com/api/v1/search?tags=story&hitsPerPage=40&query=" + urllib.parse.quote(q) +
               f"&numericFilters=created_at_i>{cutoff},points>15")
        for h in (get_json(url) or {}).get("hits", []):
            seen[h["objectID"]] = {
                "id": h["objectID"], "title": h.get("title"), "url": h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
                "hn_url": f"https://news.ycombinator.com/item?id={h['objectID']}", "points": h.get("points", 0),
                "comments": h.get("num_comments", 0), "author": h.get("author"), "date": (h.get("created_at") or "")[:10]}
        time.sleep(0.3)
    return sorted(seen.values(), key=lambda x: -x["points"])[:60]

def new_repos():
    since = (dt.date.today() - dt.timedelta(days=DAYS)).isoformat()
    h = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        h["Authorization"] = f"Bearer {GH_TOKEN}"
    pause = 2.2 if GH_TOKEN else 6.5
    seen = {}
    for t in GH_TOPICS:
        q = urllib.parse.quote(f"created:>{since} topic:{t} stars:>20")
        data = get_json(f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=30", h)
        if data is None:
            break  # rate limited; keep what we have
        for i in data.get("items", []):
            seen[i["full_name"]] = {
                "name": i["full_name"], "description": i.get("description"), "stars": i["stargazers_count"],
                "language": i.get("language"), "license": (i.get("license") or {}).get("spdx_id"),
                "date": i["created_at"][:10], "url": i["html_url"], "topics": i.get("topics", [])[:6]}
        time.sleep(pause)
    return sorted(seen.values(), key=lambda x: -x["stars"])[:60]

def main():
    hn, repos = hacker_news(), new_repos()
    if not hn and not repos:
        sys.exit("nothing fetched; leaving existing extras untouched")
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "hn": hn or old.get("hn", []), "repos": repos or old.get("repos", [])}, ensure_ascii=False))
    print(f"wrote {len(hn)} HN stories, {len(repos)} repos")

if __name__ == "__main__":
    main()
