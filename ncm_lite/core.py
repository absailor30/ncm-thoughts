"""High-level API over the official NCM checkpoints released by HazyResearch (MIT)."""
from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass, field
from importlib import resources
from typing import Optional, Sequence

import torch
from omegaconf import OmegaConf

REPO_ID = "hazyresearch/ncm-checkpoints"

# task name -> sub-folder inside the Hugging Face checkpoint repo
TASKS = {
    "tinystories": "tinystories",
    "owt": "owt",
    "lm1b-small": "lm1b/small",
    "lm1b-medium": "lm1b/medium",
    "lm1b-large": "lm1b/large",
    "xsum": "xsum",
    "wmt14": "wmt14",
    "sudoku-easy": "sudoku/easy",
    "sudoku-medium": "sudoku/medium",
    "sudoku-hard": "sudoku/hard",
}
UNCONDITIONAL = {"tinystories", "owt", "lm1b-small", "lm1b-medium", "lm1b-large"}
CONDITIONAL_TEXT = {"xsum", "wmt14"}  # xsum: article -> summary, wmt14: German -> English
DESCRIPTIONS = {
    "tinystories": "unconditional short stories (smallest download, best first try)",
    "owt": "unconditional web-style text (OpenWebText)",
    "lm1b-small": "unconditional news sentences", "lm1b-medium": "unconditional news sentences",
    "lm1b-large": "unconditional news sentences",
    "xsum": "summarize a news article (prompt = article)",
    "wmt14": "translate German to English (prompt = German text)",
    "sudoku-*": "Sudoku solving; not exposed in ncm-lite 0.1 (use the official repo)",
}


@dataclass
class Generation:
    text: str
    levels: list = field(default_factory=list)  # text decoded after each coarse-to-fine level (if requested)

    def __str__(self):
        return self.text


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _load_config(task: str):
    path = resources.files("ncm_lite").joinpath("configs", f"{task}.yaml")
    return OmegaConf.load(str(path))


def _fetch(task: str, cfg, revision: Optional[str], local_dir: Optional[str]) -> str:
    """Download exactly the files this task needs; return the local checkpoint dir that mirrors the repo layout."""
    from huggingface_hub import hf_hub_download

    sub = TASKS[task]
    wanted = [cfg.checkpoints.generator, cfg.checkpoints.tokenizer]
    wanted += [cfg.checkpoints[k] for k in ("bpe_vocab", "bpe_merges") if cfg.checkpoints.get(k)]
    root = None
    for rel in wanted:
        repo_path = posixpath.normpath(posixpath.join(sub, rel))  # lm1b: '../tokenizer.pt' lives one level up
        local = hf_hub_download(REPO_ID, repo_path, revision=revision, local_dir=local_dir)
        if rel == cfg.checkpoints.generator:
            root = os.path.dirname(local)
    return root


class NCM:
    """Next-Concept Modeling generator.

    >>> m = NCM.from_pretrained("tinystories")        # downloads ~580 MB once, then cached
    >>> m.generate(n=2, seed=0)                       # two stories
    >>> m.generate(n=1, return_levels=True)[0].levels # watch the text sharpen level by level
    """

    def __init__(self, task, cfg, tk, net, device, ck_dir):
        self.task, self.cfg, self.tk, self.net, self.device, self.ck_dir = task, cfg, tk, net, device, ck_dir
        self.eos = tk.pre_tokenizer.eos_token_id
        self.pad = tk.pre_tokenizer.pad_token_id
        self.prefix_len = int(cfg.model.prefix_len)
        self._src_tok = None
        self._l0_p = None

    # ---- loading ------------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, task: str = "tinystories", device: str = "auto", revision: Optional[str] = None,
                        local_dir: Optional[str] = None) -> "NCM":
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; choose one of {sorted(TASKS)}")
        if task.startswith("sudoku"):
            raise NotImplementedError("Sudoku uses a special tokenizer and is not exposed in ncm-lite 0.1.")
        cfg = _load_config(task)
        ck = _fetch(task, cfg, revision, local_dir)
        dev = _resolve_device(device)

        # Imported lazily so `import ncm_lite` stays fast and works before torch heavy-init.
        from ncm_lite._upstream.models.next_concept import NextConceptTransformer
        from ncm_lite._upstream.tokenization.vqvae_tokenizer import VQVAETokenizer

        cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ck, cfg.checkpoints.tokenizer)
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            if cfg.checkpoints.get(name):
                cfg.tokenizer.pre_tokenizer[key] = os.path.join(ck, cfg.checkpoints[name])
        tk = VQVAETokenizer.from_config(cfg.tokenizer)
        tk.vqvae = tk.vqvae.to(dev).eval()
        net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(dev)
        sd = torch.load(os.path.join(ck, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
        net.load_state_dict(sd["model"], strict=True)
        net.eval()
        obj = cls(task, cfg, tk, net, dev, ck)
        obj._sd_inference = sd.get("inference", {}) if isinstance(sd, dict) else {}
        return obj

    # ---- helpers ------------------------------------------------------------------------------------------
    def _unigram(self):
        """Level-0 code distribution used to start unconditional sampling (stored inside the generator checkpoint)."""
        if self._l0_p is None:
            spec = self.cfg.generation.get("level0_unigram")
            self._l0_p = None
            if spec:
                obj = torch.load(os.path.join(self.ck_dir, str(spec).split(":")[0]), map_location="cpu", weights_only=False)
                for k in str(spec).split(":", 1)[1].split("."):
                    obj = obj[k]
                self._l0_p = obj.float()
        return self._l0_p

    def _prefix(self, prompt: str):
        if self._src_tok is None:
            from transformers import AutoTokenizer
            self._src_tok = AutoTokenizer.from_pretrained(str(self.cfg.tokenizer.pre_tokenizer.model_id))
        ids = self._src_tok(prompt)["input_ids"][: self.prefix_len]
        return (ids + [self.pad] * self.prefix_len)[: self.prefix_len]

    def _decode(self, ids):
        ids = list(ids)
        if bool(self.cfg.generation.get("cut_at_eos", True)) and self.eos in ids:
            ids = ids[: ids.index(self.eos)]
        return self.tk.pre_tokenizer.decode([t for t in ids if t not in (self.eos, self.pad)], skip_special_tokens=True)

    # ---- generation ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: Optional[str | Sequence[str]] = None, n: int = 1, seed: Optional[int] = None,
                 temperature: Optional[float] = None, top_k: Optional[int] = None, top_p: Optional[float] = None,
                 guidance: Optional[float] = None, return_levels: bool = False) -> list[Generation]:
        """Generate text.

        Unconditional tasks (tinystories, owt, lm1b-*): leave `prompt` empty and set `n`.
        Conditional tasks (xsum, wmt14): pass one prompt string or a list of them.
        `guidance` is classifier-free guidance (each guided step costs two function evaluations).
        """
        G = self.cfg.generation
        temp = float(temperature if temperature is not None else G.get("temperature", 1.0))
        k = int(top_k if top_k is not None else G.get("top_k", 1))
        p = float(top_p if top_p is not None else G.get("top_p", 0.0))
        cfg_scale = float(guidance if guidance is not None else G.get("cfg_scale", 0.0))
        if seed is not None:
            torch.manual_seed(seed)

        dev = self.device
        if self.task in UNCONDITIONAL:
            if prompt:
                raise ValueError(f"task {self.task!r} is unconditional; omit `prompt`")
            B = int(n)
            prefix = torch.full((B, 1), self.eos, device=dev)
            l0 = None
            uni = self._unigram()
            if uni is not None:
                L0 = self.tk.vqvae.vq.levels[0]
                l0 = torch.multinomial(uni, B * L0, replacement=True).view(B, L0).to(dev)
        else:
            if not prompt:
                raise ValueError(f"task {self.task!r} needs a `prompt`")
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            prefix = torch.tensor([self._prefix(x) for x in prompts], device=dev)
            l0 = None

        codes = self.net.generate(l0, prefix, temperature=temp, top_k=k, top_p=p,
                                  cfg_scale=cfg_scale, cfg_null_eos=self.eos)
        _, zl = self.tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        final = self.tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
        per_level = [self.tk.vqvae.decoder(z).argmax(-1).tolist() for z in zl] if return_levels else []
        return [Generation(self._decode(final[i]), [self._decode(lv[i]) for lv in per_level]) for i in range(len(final))]
