"""Hugging Face Space: watch NCM refine text level by level (CPU)."""
import gradio as gr
from ncm_lite import NCM

_MODELS = {}


def model(task):
    if task not in _MODELS:
        _MODELS[task] = NCM.from_pretrained(task, device="cpu")
    return _MODELS[task]


def _pack(outs):
    levels = outs[0].levels
    return outs[0].text, levels, gr.update(maximum=len(levels), value=len(levels), interactive=True), levels[-1]


def story(seed, temperature):
    return _pack(model("tinystories").generate(n=1, seed=int(seed), temperature=float(temperature), return_levels=True))


def summarize(article, seed):
    if not article or not article.strip():
        raise gr.Error("Paste a news article first.")
    return _pack(model("xsum").generate(prompt=article, seed=int(seed), return_levels=True))


def show_level(levels, i):
    return levels[int(i) - 1] if levels else ""


with gr.Blocks(title="ncm-lite demo") as demo:
    gr.Markdown(
        "# ncm-lite demo\nText is generated **coarse to fine**: a rough sketch of the whole passage first, then "
        "~12 parallel refinement steps. Use the slider to replay the steps.\n\n"
        "*Small research models (about 90M parameters): expect odd wording and invented facts. Unofficial demo of "
        "[Next-Concept Modeling](https://arxiv.org/abs/2610.14162). The first click downloads the model (~1 min).*"
    )
    levels = gr.State([])
    with gr.Tabs():
        with gr.Tab("Write a story"):
            seed = gr.Number(0, label="Seed (change for a different story)", precision=0)
            temp = gr.Slider(0.3, 1.5, 1.0, step=0.05, label="Temperature (higher = more random)")
            b1 = gr.Button("Generate story", variant="primary")
        with gr.Tab("Summarize an article"):
            art = gr.Textbox(lines=8, label="News article")
            seed2 = gr.Number(0, label="Seed", precision=0)
            b2 = gr.Button("Summarize", variant="primary")
    final = gr.Textbox(label="Final text", lines=8)
    step = gr.Slider(1, 12, value=12, step=1, label="Refinement step to display", interactive=False)
    replay = gr.Textbox(label="Text at the selected step", lines=8)
    outs = [final, levels, step, replay]
    b1.click(story, [seed, temp], outs)
    b2.click(summarize, [art, seed2], outs)
    step.change(show_level, [levels, step], replay)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
