import os
import posixpath
import subprocess
import sys
from pathlib import Path

import pytest

from ncm_lite import NCM, TASKS
from ncm_lite.cli import main
from ncm_lite.core import UNCONDITIONAL, CONDITIONAL_TEXT, _load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("task", sorted(TASKS))
def test_config_ships_for_every_task(task):
    cfg = _load_config(task)
    assert cfg.checkpoints.generator and cfg.checkpoints.tokenizer


def test_every_text_task_is_classified():
    text_tasks = {t for t in TASKS if not t.startswith("sudoku")}
    assert text_tasks == UNCONDITIONAL | CONDITIONAL_TEXT


def test_checkpoint_paths_resolve_inside_repo():
    # lm1b stores the tokenizer one level above the generator ('../tokenizer.pt').
    cfg = _load_config("lm1b-small")
    assert posixpath.normpath(posixpath.join(TASKS["lm1b-small"], cfg.checkpoints.tokenizer)) == "lm1b/tokenizer.pt"


def test_unknown_task_and_sudoku_rejected():
    with pytest.raises(ValueError):
        NCM.from_pretrained("nope")
    with pytest.raises(NotImplementedError):
        NCM.from_pretrained("sudoku-easy")


def test_cli_list(capsys):
    assert main(["list"]) == 0
    assert "tinystories" in capsys.readouterr().out


def test_upstream_copy_is_in_sync_with_vendor():
    """ncm_lite/_upstream must equal a fresh run of tools/sync_upstream.py (no hand edits)."""
    before = {p: p.read_text() for p in (ROOT / "ncm_lite" / "_upstream").rglob("*.py")}
    subprocess.run([sys.executable, str(ROOT / "tools" / "sync_upstream.py")], check=True, capture_output=True)
    after = {p: p.read_text() for p in (ROOT / "ncm_lite" / "_upstream").rglob("*.py")}
    assert before == after


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("NCM_LITE_SLOW"), reason="downloads ~580 MB; set NCM_LITE_SLOW=1")
def test_tinystories_end_to_end():
    m = NCM.from_pretrained("tinystories", device="cpu")
    a = m.generate(n=2, seed=0, return_levels=True)
    b = m.generate(n=2, seed=0)
    assert len(a) == 2 and all(x.text.strip() for x in a)
    assert len(a[0].levels) == 12          # one decode per coarse-to-fine level
    assert [x.text for x in a] == [x.text for x in b]  # same seed -> same output
    with pytest.raises(ValueError):
        m.generate(prompt="hello")
