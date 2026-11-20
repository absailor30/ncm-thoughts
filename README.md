# ncm-lite

Easy-to-use Python wrapper and CLI for **Next-Concept Modeling (NCM)**, the fast coarse-to-fine text generator from
[arXiv:2610.14162](https://arxiv.org/abs/2610.14162) (Kumbong et al., Stanford / Together AI).

Normal language models write one token at a time. NCM writes a rough version of the whole text first, then refines
it in about 12 parallel steps. `ncm-lite` downloads the official released checkpoints and gives you one call to run them.

> Status: **0.1, unofficial.** Not affiliated with the paper's authors. The checkpoints are small research models,
> so expect repetition, odd wording and made-up facts (see Limitations).

## Install
```bash
pip install -e .            # from a clone of this repo; PyPI release not published yet
```
CPU works (about 15 s for two TinyStories samples on 4 cores, after the one-time download). A GPU is used automatically if present.

## Use
```python
from ncm_lite import NCM

m = NCM.from_pretrained("tinystories")            # ~580 MB download once, then cached
print(m.generate(n=2, seed=0)[0])                 # unconditional story

out = m.generate(n=1, seed=0, return_levels=True)[0]
for i, t in enumerate(out.levels, 1):             # watch the text sharpen, level by level
    print(i, t[:80])

s = NCM.from_pretrained("xsum")                   # conditional: summarize an article
print(s.generate(prompt="<article text>")[0])
```
```bash
ncm-lite list
ncm-lite generate tinystories -n 2 --seed 0 --levels
ncm-lite generate xsum --prompt "<article text>"
```
Tasks: `tinystories`, `owt`, `lm1b-small|medium|large` (unconditional), `xsum` (summarize), `wmt14` (German to English).
Sudoku checkpoints exist upstream but are not exposed in 0.1.

Web demo: `pip install -e ".[demo]" && python examples/demo.py` (the app builds and its generate function runs; I have not clicked through the UI in a browser).

## Limitations (read this)
- Output quality is that of the paper's ~90-100M parameter models. In my XSum test a summary invented "Oxford" and a "car lane" that were not in the article.
- Only the paper's released tasks work; there is no general chat or instruction following.
- No training code here. For training from scratch see community work such as [amar683/tiny-ncm](https://github.com/amar683/tiny-ncm) (not reviewed in depth).
- Speed claims in the paper (10-31x vs diffusion LMs, on an H100) were not reproduced here.

## Repo layout
- `ncm_lite/`: the package. `_upstream/` is a mechanically rewritten copy of the official code (do not edit; regenerate with `python tools/sync_upstream.py`).
- `vendor/ncm/`: untouched copy of [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT).
- `docs/`, `radar/`, `.github/workflows/radar.yml`: the Paper Radar site (new AI papers, refreshed every 2 hours).
- `tests/`: `pytest` (set `NCM_LITE_SLOW=1` to also run the real-checkpoint test).

## Licence and credit
MIT, see `LICENSE` and `THIRD_PARTY_NOTICES.md`. All credit for the method and checkpoints goes to the NCM authors. If you use this, cite the paper:
```bibtex
@misc{kumbong2026nextconcept,
  title={Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation},
  author={Hermann Kumbong and Ishan S. Khare and Iddah Mlauzi and Andrew Woen and Charles Ding and E. Kelly Buchanan and Simran Arora and Azalia Mirhoseini and Christopher R\'e},
  year={2026}, eprint={2610.14162}, archivePrefix={arXiv}, primaryClass={cs.CL}
}
```
