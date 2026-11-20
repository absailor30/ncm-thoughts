# Third-party notices

## HazyResearch/ncm (MIT)
`vendor/ncm/` is an unmodified copy of https://github.com/HazyResearch/ncm (commit c702742), and
`ncm_lite/_upstream/` is a mechanical copy of its `models/`, `tokenization/` and `utils/` folders with only the
import paths rewritten (see `tools/sync_upstream.py`). Copyright (c) 2025 hermann kumbong. Licence text: `vendor/ncm/LICENSE`.

Model weights are downloaded at runtime from https://huggingface.co/hazyresearch/ncm-checkpoints (licence tag on the Hub: MIT) and are not
distributed in this repository.

## Paper
Kumbong, Khare, Mlauzi, Woen, Ding, Buchanan, Arora, Mirhoseini, Ré. *Next-Concept Modeling: Fast, Parallel
Coarse-to-Fine Text Generation.* arXiv:2610.14162, 2026.
