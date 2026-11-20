# ncm-lite

Easy-to-use Python wrapper and CLI for **Next-Concept Modeling (NCM)**, the fast coarse-to-fine text generator from
[arXiv:2610.14162](https://arxiv.org/abs/2610.14162) (Kumbong et al., Stanford / Together AI).

Normal language models write one token at a time. NCM writes a rough version of the whole text first, then refines
it in about 12 parallel steps. `ncm-lite` downloads the official released checkpoints and gives you one call to run them.

**Try it in your browser (no install): https://huggingface.co/spaces/absailor/ncm-lite-demo**

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

Hosted demo: [absailor/ncm-lite-demo](https://huggingface.co/spaces/absailor/ncm-lite-demo) (source in `space/`; rebuild with `python tools/build_space.py`). Local version: `pip install -e ".[demo]" && python examples/demo.py`.

## Limitations (read this)
- Output quality is that of the paper's ~90-100M parameter models. In my XSum test a summary invented "Oxford" and a "car lane" that were not in the article.
- Only the paper's released tasks work; there is no general chat or instruction following.
- No training code here. For training from scratch see community work such as [amar683/tiny-ncm](https://github.com/amar683/tiny-ncm) (not reviewed in depth).
- Speed claims in the paper (10-31x vs diffusion LMs, on an H100) were not reproduced here.

## Repo layout
- `ncm_lite/`: the package. `_upstream/` is a mechanically rewritten copy of the official code (do not edit; regenerate with `python tools/sync_upstream.py`).
- `vendor/ncm/`: untouched copy of [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT).
- `docs/`, `radar/`, `.github/workflows/`: the Paper Radar site (https://ncm-thoughts-docs.vercel.app). New AI papers refreshed every 2 hours, a "No code yet" list of trending papers with no code/repo/model found, Hacker News and new-repo feeds, and a weekly digest (Mondays).
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
