# ncm-thoughts

Two things live here:

1. **Paper Radar** (`docs/`, `radar/`, `.github/workflows/radar.yml`): a GitHub Pages site that lists new AI papers from
   Hugging Face Daily Papers and arXiv, with code links, topic filters and a Refresh button.
   A GitHub Action rebuilds `docs/data/papers.json` every 2 hours.
2. **`vendor/ncm/`**: vendored copy of the official MIT-licensed [HazyResearch/ncm](https://github.com/HazyResearch/ncm)
   (Next-Concept Modeling, arXiv:2610.14162), the base for the planned `ncm-lite` library.

## Enable the site
Settings → Pages → Source: *Deploy from a branch* → pick the branch (default branch recommended) and folder `/docs`.
Scheduled workflows only run on the **default branch**, so merge this work there for the 2-hourly updates.

## Run locally
```bash
python3 radar/fetch.py && python3 -m http.server -d docs 8000
```
