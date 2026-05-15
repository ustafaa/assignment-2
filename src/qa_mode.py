"""Phase 4: RAG QA mode.

Two retrievers + a single MedGemma answer step. Common return shape:
    {
      "retriever": str,
      "answer": str,
      "retrieved": [{"id", "report", "score", "image_path"}, ...],
      "latency_s": float,
    }

Run as `python -m src.qa_mode` to smoke-test both retrievers on 5 questions
from data/sample/qa_dataset.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_CFG_PATH = _REPO_ROOT / "config.yaml"


QA_ANSWER_PROMPT_TEMPLATE = (
    "Use ONLY the reports below to answer. If the answer is not in the reports, "
    "say so explicitly.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_context(retrieved: list[dict]) -> str:
    """Render retrieved reports into a single prompt-ready context block."""
    parts: list[str] = []
    for i, r in enumerate(retrieved, 1):
        parts.append(f"[Report {i} | id={r['id']}]\n{r['report']}")
    return "\n\n".join(parts)


def _answer(retrieved: list[dict], question: str, max_new_tokens: int = 256) -> str:
    from src.models import medgemma
    prompt = QA_ANSWER_PROMPT_TEMPLATE.format(
        context=_format_context(retrieved),
        question=question,
    )
    return medgemma.generate(None, prompt, max_new_tokens=max_new_tokens).strip()


def qa_colpali_rag(question: str, top_k: Optional[int] = None) -> dict:
    """Retrieve top-k report pages with ColPali, then answer with MedGemma."""
    from src.models import colpali_retriever
    cfg = _load_config()
    if top_k is None:
        top_k = cfg["retrieval"]["top_k"]
    t0 = time.perf_counter()
    retrieved = colpali_retriever.query(question, top_k=top_k)
    answer = _answer(retrieved, question)
    return {
        "retriever": cfg["models"]["colpali"]["hf_id"],
        "answer": answer,
        "retrieved": retrieved,
        "latency_s": time.perf_counter() - t0,
    }


def qa_text_rag(question: str, top_k: Optional[int] = None) -> dict:
    """Retrieve top-k reports with MiniLM, then answer with MedGemma."""
    from src.models import text_retriever
    cfg = _load_config()
    if top_k is None:
        top_k = cfg["retrieval"]["top_k"]
    t0 = time.perf_counter()
    retrieved = text_retriever.query(question, top_k=top_k)
    answer = _answer(retrieved, question)
    return {
        "retriever": cfg["models"]["text_retriever"]["hf_id"],
        "answer": answer,
        "retrieved": retrieved,
        "latency_s": time.perf_counter() - t0,
    }


def _truncate(s: str, n: int = 300) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def _main() -> None:
    # Line-buffered stdout so Colab `!python -m src.qa_mode` shows progress live.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    cfg = _load_config()
    qa_path = _REPO_ROOT / cfg["data"]["qa_dataset"]
    if not qa_path.exists():
        print(f"QA dataset not found at {qa_path}; run `python data/build_qa_dataset.py` first.")
        return
    with open(qa_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    if not ds:
        print("QA dataset is empty.")
        return

    samples: list[tuple[str, str, str]] = []
    for entry in ds:
        if not entry.get("qa_pairs"):
            continue
        pair = entry["qa_pairs"][0]
        samples.append((entry["report_id"], pair["q"], pair["a"]))
        if len(samples) >= 5:
            break

    print(f"Smoke-testing {len(samples)} questions on both retrievers.")
    print("(First call per retriever loads the model + index.)")

    for i, (rid, q, gold_a) in enumerate(samples):
        print(f"\n=== [{i+1}/{len(samples)}] gold_report_id={rid} ===")
        print(f"Q: {q}")
        print(f"gold A: {gold_a}")
        for fn, label in [(qa_text_rag, "TEXT"), (qa_colpali_rag, "COLPALI")]:
            try:
                r = fn(q)
                top_ids = [str(x["id"])[:12] for x in r["retrieved"]]
                gold_hit = any(str(x["id"]) == str(rid) for x in r["retrieved"])
                print(
                    f"--- {label}  ({r['latency_s']:.1f}s, "
                    f"gold_in_top{len(r['retrieved'])}={gold_hit}, top={top_ids}) ---"
                )
                print(_truncate(r["answer"]))
            except Exception as e:
                print(f"{label} error: {e}")


if __name__ == "__main__":
    _main()
