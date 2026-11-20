import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "radar"))
import fetch, digest  # noqa: E402


def test_promise_detector():
    assert fetch.PROMISE.search("Code and models will be released soon.")
    assert fetch.PROMISE.search("Our weights will be made publicly available")
    assert not fetch.PROMISE.search("We release no code and evaluate on three benchmarks.")


def test_aggregator_repos_filtered():
    for bad in ("someone/my_arXiv_daily", "x/awesome-llm-papers", "y/paper-reading-notes"):
        assert fetch.AGGREGATOR.search(bad)
    assert not fetch.AGGREGATOR.search("lab/ncm-mlx")


def test_topics_valid_and_synced():
    t = json.loads((ROOT / "radar" / "topics.json").read_text())
    assert len(t) >= 10 and all(isinstance(v, list) and v for v in t.values())
    assert t == json.loads((ROOT / "docs" / "data" / "topics.json").read_text()), "run: cp radar/topics.json docs/data/"


def test_digest_tagging():
    t = {"Agents": ["agent"], "Code": ["code generation"]}
    assert digest.tag({"title": "An agent for code generation", "summary": ""}, t) == ["Agents", "Code"]


def test_shipped_data_is_well_formed():
    d = json.loads((ROOT / "docs" / "data" / "papers.json").read_text())
    assert d["count"] == len(d["papers"]) > 0
    assert {p["status"] for p in d["papers"]} <= {"official", "community", "open", "unchecked"}
