"""Tiny web demo: pip install 'ncm-lite[demo]' && python examples/demo.py"""
import gradio as gr
from ncm_lite import NCM

model = NCM.from_pretrained("tinystories", device="auto")


def run(seed, temperature, guidance_unused):
    out = model.generate(n=1, seed=int(seed), temperature=float(temperature), return_levels=True)[0]
    steps = "\n\n".join(f"Level {i}/{len(out.levels)}:\n{t}" for i, t in enumerate(out.levels, 1))
    return out.text, steps


with gr.Blocks(title="ncm-lite") as demo:
    gr.Markdown("# ncm-lite\nText is generated coarse-to-fine: the full story is refined in ~12 parallel steps.")
    seed = gr.Number(0, label="Seed", precision=0)
    temp = gr.Slider(0.3, 1.5, 1.0, label="Temperature")
    gr.Button("Generate").click(run, [seed, temp, gr.State(0)], [gr.Textbox(label="Story", lines=8),
                                                                gr.Textbox(label="Refinement steps", lines=20)])

if __name__ == "__main__":
    demo.launch()
