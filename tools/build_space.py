#!/usr/bin/env python3
"""Copy the ncm_lite package into space/ so the folder can be uploaded to a Hugging Face Space as-is."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "ncm_lite", ROOT / "space" / "ncm_lite"
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copy(ROOT / "LICENSE", ROOT / "space" / "LICENSE")
shutil.copy(ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "space" / "THIRD_PARTY_NOTICES.md")
print("space/ ready:", sum(1 for _ in (ROOT / "space").rglob("*") if _.is_file()), "files")
