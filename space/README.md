---
title: ncm-lite demo
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: Watch text sharpen step by step with Next-Concept Modeling
---

# ncm-lite demo

Unofficial demo of **Next-Concept Modeling** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162)) using the
authors' released checkpoints ([hazyresearch/ncm-checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints)).
Text is written coarse-to-fine: the whole passage is sketched first, then refined in ~12 parallel steps.

Small research models: expect odd wording and invented facts. Not affiliated with the paper's authors.
