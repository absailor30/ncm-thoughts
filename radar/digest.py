#!/usr/bin/env python3
"""Build a weekly digest from the current radar data -> docs/data/digests/<YYYY-Www>.json + index.json.

Template-based (no AI-written prose): ranked lists plus topic counts, so every line is traceable to data.
"""
import datetime as dt, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
OUT = DATA / "digests"

def tag(p, topics):
    t = (p["title"] + " " + p["summary"]).lower()
    return [k for k, ws in topics.items() if any(w in t for w in ws)]

def main():
    papers = json.loads((DATA / "papers.json").read_text())["papers"]
    extras = json.loads((DATA / "extras.json").read_text()) if (DATA / "extras.json").exists() else {}
    topics = json.loads((ROOT / "radar" / "topics.json").read_text())
    today = dt.date.today()
    since = (today - dt.timedelta(days=7)).isoformat()
    week = [p for p in papers if p["date"] >= since]
    keep = lambda p: {k: p.get(k) for k in ("id", "title", "url", "date", "upvotes", "stars", "code", "status", "promised", "tldr")} | \
        {"community": [r["name"] for r in p.get("gh_repos", [])], "hf_models": p.get("hf_model_count", 0)}
    opportunities = sorted((p for p in week if p["status"] == "open"), key=lambda p: -p["upvotes"])[:10]
    with_code = sorted((p for p in week if p["code"] and p["source"] == "hf"), key=lambda p: -(p["upvotes"] + (p.get("stars") or 0) / 10))[:10]
    counts = {}
    for p in week:
        for t in tag(p, topics):
            counts[t] = counts.get(t, 0) + 1
    iso = today.isocalendar()
    digest = {
        "week": f"{iso[0]}-W{iso[1]:02d}", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "window": [since, today.isoformat()], "papers_seen": len(week),
        "papers_with_code": sum(1 for p in week if p["code"]),
        "opportunities": [keep(p) for p in opportunities],
        "top_with_code": [keep(p) for p in with_code],
        "new_repos": (extras.get("repos") or [])[:8], "hn": (extras.get("hn") or [])[:8],
        "topic_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "note": "Auto-generated from ranked data; 'opportunity' = trending on Hugging Face with no official code, GitHub repo or HF model found. Verify before building.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{digest['week']}.json").write_text(json.dumps(digest, ensure_ascii=False))
    weeks = sorted((f.stem for f in OUT.glob("20*.json")), reverse=True)
    (OUT / "index.json").write_text(json.dumps(weeks))
    print("wrote digest", digest["week"], "| opportunities", len(opportunities), "| archive", weeks)

if __name__ == "__main__":
    main()
