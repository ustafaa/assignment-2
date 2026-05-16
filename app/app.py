"""Phase 6: Gradio demo with two tabs (Report + QA).

UI design notes:
- Soft theme; wider container so reports + retrieved previews fit.
- Each tab has TWO modes: "single model" and "side-by-side compare".
- Retrieved reports are shown inline with a per-report preview, not just IDs.
- All model loading is lazy: opening the app doesn't touch GPU.

Launches with share=True (per config.demo.share) so the Colab share link is
publicly reachable.

Run:
    python app/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_CFG_PATH = _REPO_ROOT / "config.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Lazy imports so opening the app doesn't load every model at import time.
def _report_medgemma(img):
    from src.report_mode import generate_report_medgemma
    return generate_report_medgemma(img)


def _report_clip(img):
    from src.report_mode import generate_report_clip
    return generate_report_clip(img)


def _qa_colpali(q, k):
    from src.qa_mode import qa_colpali_rag
    return qa_colpali_rag(q, top_k=k)


def _qa_text(q, k):
    from src.qa_mode import qa_text_rag
    return qa_text_rag(q, top_k=k)


REPORT_MODELS = {
    "MedGemma 1.5-4B (multimodal)": _report_medgemma,
    "OpenCLIP ViT-B/32 (retrieval)": _report_clip,
}

QA_RETRIEVERS = {
    "MiniLM-L6 (text bi-encoder)": _qa_text,
    "ColPali v1.3 (vision late-interaction)": _qa_colpali,
}

SAMPLE_QUESTIONS = [
    "Is there evidence of pleural effusion?",
    "Describe the cardiac silhouette.",
    "Are there any signs of pneumothorax on the left side?",
    "Is there consolidation visible in either lung?",
    "Comment on the position of any endotracheal tube or central line.",
]


def _truncate(s: str, n: int = 320) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def _format_retrieved(retrieved: list[dict]) -> str:
    if not retrieved:
        return "_No reports retrieved._"
    lines: list[str] = []
    for i, r in enumerate(retrieved, 1):
        score = r.get("score", r.get("similarity", 0.0))
        lines.append(f"**[{i}]** &nbsp; id `{r['id']}` &nbsp; · &nbsp; score `{score:.3f}`")
        lines.append("")
        lines.append(f"> {_truncate(r['report'])}")
        lines.append("")
    return "\n".join(lines)


# ---------------- Report tab handlers ---------------- #

def run_report(image, model_name: str):
    """Single-model report generation."""
    if image is None:
        return "Please upload an image.", "", ""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    fn = REPORT_MODELS[model_name]
    try:
        result = fn(image.convert("RGB"))
    except Exception as e:
        return f"Error: {e}", "", ""
    detail = ""
    extras = result.get("extras", {})
    if "retrieved_id" in extras:
        detail = (
            f"Nearest training patient: `{extras['retrieved_id']}` &nbsp;·&nbsp; "
            f"cosine `{extras['similarity']:.3f}`"
        )
    return result["report"], f"{result['latency_s']:.2f} s", detail


def run_report_compare(image):
    """Run BOTH report models on the same image."""
    if image is None:
        empty = ("Please upload an image.", "")
        return *empty, *empty, ""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    img = image.convert("RGB")
    try:
        med = _report_medgemma(img)
    except Exception as e:
        med = {"report": f"Error: {e}", "latency_s": 0.0, "extras": {}}
    try:
        clip = _report_clip(img)
    except Exception as e:
        clip = {"report": f"Error: {e}", "latency_s": 0.0, "extras": {}}
    clip_extras = clip.get("extras", {})
    clip_detail = ""
    if "retrieved_id" in clip_extras:
        clip_detail = (
            f"Nearest training patient: `{clip_extras['retrieved_id']}` &nbsp;·&nbsp; "
            f"cosine `{clip_extras['similarity']:.3f}`"
        )
    return (
        med["report"],
        f"{med['latency_s']:.2f} s",
        clip["report"],
        f"{clip['latency_s']:.2f} s",
        clip_detail,
    )


# ---------------- QA tab handlers ---------------- #

def run_qa(question: str, retriever_name: str, top_k: int):
    """Single-retriever RAG QA."""
    q = (question or "").strip()
    if not q:
        return "Please enter a question.", "_No reports retrieved._", ""
    fn = QA_RETRIEVERS[retriever_name]
    try:
        result = fn(q, top_k)
    except Exception as e:
        return f"Error: {e}", "_No reports retrieved._", ""
    return result["answer"], _format_retrieved(result["retrieved"]), f"{result['latency_s']:.2f} s"


def run_qa_compare(question: str, top_k: int):
    """Run BOTH retrievers and answer with each."""
    q = (question or "").strip()
    if not q:
        msg = "Please enter a question."
        return msg, "_–_", "", msg, "_–_", ""
    try:
        a = _qa_text(q, top_k)
    except Exception as e:
        a = {"answer": f"Error: {e}", "retrieved": [], "latency_s": 0.0}
    try:
        b = _qa_colpali(q, top_k)
    except Exception as e:
        b = {"answer": f"Error: {e}", "retrieved": [], "latency_s": 0.0}
    return (
        a["answer"], _format_retrieved(a["retrieved"]), f"{a['latency_s']:.2f} s",
        b["answer"], _format_retrieved(b["retrieved"]), f"{b['latency_s']:.2f} s",
    )


# ---------------- UI ---------------- #

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.report-output textarea {
    font-family: 'JetBrains Mono','SF Mono','Menlo',monospace !important;
    font-size: 0.86rem !important; line-height: 1.55 !important;
}
.section-header { font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; }
footer { display: none !important; }
"""


def _build_ui() -> gr.Blocks:
    cfg = _load_config()
    default_k = cfg["retrieval"]["top_k"]

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Chest X-Ray Multi-Modal System", theme=theme, css=CSS) as demo:
        gr.Markdown(
            """
            # Chest X-Ray Multi-Modal Intelligence System

            **DSAI 413 — Assignment 2.** Image-to-report generation and RAG QA over a 300-report MIMIC-CXR subset.

            **Mandatory models:** `google/medgemma-1.5-4b-it` · `vidore/colpali-v1.3-merged`
            **Baselines:** OpenCLIP ViT-B/32 · sentence-transformers MiniLM-L6
            """
        )

        with gr.Tabs():
            # ============== REPORT TAB ============== #
            with gr.Tab("Report generation"):
                gr.Markdown(
                    "Upload a chest X-ray. **MedGemma** writes a structured Findings/Impression "
                    "report from scratch (~10 s). The **OpenCLIP retrieval** baseline returns the "
                    "report of the nearest training image verbatim (~50 ms)."
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        img_input = gr.Image(type="pil", label="Chest X-ray", height=380)
                        model_select = gr.Dropdown(
                            list(REPORT_MODELS.keys()),
                            value=list(REPORT_MODELS.keys())[0],
                            label="Model",
                        )
                        with gr.Row():
                            single_btn = gr.Button("Generate report", variant="primary")
                            compare_btn = gr.Button("Run both side-by-side")

                    with gr.Column(scale=3, min_width=400):
                        report_out = gr.Textbox(
                            label="Generated report",
                            lines=15,
                            elem_classes=["report-output"],
                            show_copy_button=True,
                        )
                        with gr.Row():
                            report_latency = gr.Textbox(
                                label="Latency", interactive=False, max_lines=1, scale=1
                            )
                            report_detail = gr.Markdown(value="", elem_id="report_detail")

                with gr.Accordion("Side-by-side comparison", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("##### MedGemma 1.5-4B", elem_classes=["section-header"])
                            med_out = gr.Textbox(
                                lines=13, elem_classes=["report-output"],
                                label="", show_copy_button=True,
                            )
                            med_lat = gr.Textbox(label="Latency", interactive=False, max_lines=1)
                        with gr.Column():
                            gr.Markdown("##### OpenCLIP retrieval", elem_classes=["section-header"])
                            clip_out = gr.Textbox(
                                lines=13, elem_classes=["report-output"],
                                label="", show_copy_button=True,
                            )
                            clip_lat = gr.Textbox(label="Latency", interactive=False, max_lines=1)
                    compare_detail = gr.Markdown(value="")

                single_btn.click(
                    run_report,
                    inputs=[img_input, model_select],
                    outputs=[report_out, report_latency, report_detail],
                )
                compare_btn.click(
                    run_report_compare,
                    inputs=[img_input],
                    outputs=[med_out, med_lat, clip_out, clip_lat, compare_detail],
                )

            # ============== QA TAB ============== #
            with gr.Tab("QA over reports (RAG)"):
                gr.Markdown(
                    "Ask a clinical question. The retriever pulls the top-`k` most relevant reports "
                    "from the 300-report index; **MedGemma** then answers using ONLY those reports "
                    "as context. Retrieved reports are previewed below the answer."
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        q_input = gr.Textbox(
                            label="Question",
                            placeholder="e.g. Is there a pleural effusion? Describe the cardiac silhouette.",
                            lines=2,
                        )
                        gr.Examples(
                            examples=[[q] for q in SAMPLE_QUESTIONS],
                            inputs=[q_input],
                            label="Sample questions",
                        )
                        with gr.Row():
                            retriever_select = gr.Dropdown(
                                list(QA_RETRIEVERS.keys()),
                                value=list(QA_RETRIEVERS.keys())[0],
                                label="Retriever",
                            )
                            k_slider = gr.Slider(
                                minimum=1, maximum=10, value=default_k, step=1,
                                label="top_k",
                            )
                        with gr.Row():
                            qa_btn = gr.Button("Ask", variant="primary")
                            qa_compare_btn = gr.Button("Compare retrievers")
                        qa_latency = gr.Textbox(label="Latency", interactive=False, max_lines=1)

                    with gr.Column(scale=3, min_width=400):
                        answer_out = gr.Textbox(
                            label="Answer",
                            lines=9,
                            elem_classes=["report-output"],
                            show_copy_button=True,
                        )
                        retrieved_out = gr.Markdown(label="Retrieved reports")

                with gr.Accordion("Side-by-side retriever comparison", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("##### MiniLM-L6 (text)", elem_classes=["section-header"])
                            txt_ans = gr.Textbox(
                                label="Answer", lines=8,
                                elem_classes=["report-output"], show_copy_button=True,
                            )
                            txt_lat = gr.Textbox(label="Latency", interactive=False, max_lines=1)
                            txt_ret = gr.Markdown(label="Retrieved")
                        with gr.Column():
                            gr.Markdown("##### ColPali v1.3 (vision)", elem_classes=["section-header"])
                            cp_ans = gr.Textbox(
                                label="Answer", lines=8,
                                elem_classes=["report-output"], show_copy_button=True,
                            )
                            cp_lat = gr.Textbox(label="Latency", interactive=False, max_lines=1)
                            cp_ret = gr.Markdown(label="Retrieved")

                qa_btn.click(
                    run_qa,
                    inputs=[q_input, retriever_select, k_slider],
                    outputs=[answer_out, retrieved_out, qa_latency],
                )
                qa_compare_btn.click(
                    run_qa_compare,
                    inputs=[q_input, k_slider],
                    outputs=[txt_ans, txt_ret, txt_lat, cp_ans, cp_ret, cp_lat],
                )

        gr.Markdown(
            "<sub>First query per pipeline triggers a lazy model + index load — expect ~60 s "
            "for MedGemma 4-bit and ~15 s for ColPali. Subsequent queries are fast.</sub>"
        )

    return demo


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    cfg = _load_config()
    demo = _build_ui()
    demo.queue()  # serialize requests so models aren't re-entered concurrently
    demo.launch(
        share=cfg["demo"].get("share", True),
        server_name=cfg["demo"].get("server_name", "0.0.0.0"),
        server_port=cfg["demo"].get("server_port", 7860),
    )


if __name__ == "__main__":
    main()
