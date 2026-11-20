import argparse
import sys

from .core import DESCRIPTIONS, TASKS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ncm-lite", description="Next-Concept Modeling: fast coarse-to-fine text generation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show available tasks")
    g = sub.add_parser("generate", help="generate text")
    g.add_argument("task", choices=sorted(TASKS))
    g.add_argument("--prompt", help="input text (needed for xsum / wmt14)")
    g.add_argument("-n", type=int, default=1, help="number of samples (unconditional tasks)")
    g.add_argument("--seed", type=int)
    g.add_argument("--device", default="auto")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--top-p", type=float)
    g.add_argument("--guidance", type=float, help="classifier-free guidance scale")
    g.add_argument("--levels", action="store_true", help="also print the text after each coarse-to-fine level")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for t in sorted(TASKS):
            print(f"{t:14s} {DESCRIPTIONS.get(t, DESCRIPTIONS.get('sudoku-*') if t.startswith('sudoku') else '')}")
        return 0

    from .core import NCM
    try:
        m = NCM.from_pretrained(a.task, device=a.device)
        outs = m.generate(prompt=a.prompt, n=a.n, seed=a.seed, temperature=a.temperature, top_k=a.top_k,
                          top_p=a.top_p, guidance=a.guidance, return_levels=a.levels)
    except (ValueError, NotImplementedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for i, o in enumerate(outs):
        if a.levels:
            for j, t in enumerate(o.levels, 1):
                print(f"[level {j}/{len(o.levels)}] {t}")
            print("---")
        print(o.text)
        if i < len(outs) - 1:
            print("\n=====\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
