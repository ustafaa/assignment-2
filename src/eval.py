"""Phase 5: Evaluation harness.

Report mode (image -> report):
  - MedGemma vs OpenCLIP retrieval baseline
  - Metrics: ROUGE-L, BERTScore F1
  - Test set: manifest_test.csv (default 100 imgs, override via config)

QA mode (question -> answer over reports):
  - MedGemma+ColPali vs MedGemma+MiniLM-text
  - Metrics: Recall@3 (retrieval) + LLM-judge correct/partial/wrong (answer)
  - Test set: qa_dataset.json (default 50 items, override via config)

Outputs:
  - results/comparison.json (full per-item detail + aggregates)
  - results/comparison.md  (Markdown summary table)

Usage:
    python -m src.eval                     # full run
    python -m src.eval --skip-report       # only QA scoring
    python -m src.eval --skip-qa           # only report scoring
    python -m src.eval --limit-report 10   # 10 images
    python -m src.eval --limit-qa 5        # 5 QA items
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_CFG_PATH = _REPO_ROOT / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("eval")


JUDGE_PROMPT = (
    "Score a predicted answer against the gold answer for a chest X-ray question.\n\n"
    "Scoring rubric:\n"
    "  1 = correct  (matches the gold answer's meaning)\n"
    "  2 = partial  (partially matches or hedges)\n"
    "  3 = wrong    (contradicts the gold answer or is unrelated)\n\n"
    "Question: {question}\n"
    "Predicted: {predicted}\n"
    "Gold: {gold}\n\n"
    "Output your verdict on a single line in EXACTLY this format:\n"
    "VERDICT: <digit 1, 2, or 3>"
)


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------- Judge parsing ------------------------------ #

_VERDICT_LINE_RE = re.compile(r"VERDICT:\s*([123])", re.IGNORECASE)
_VERDICT_WORD_RE = re.compile(r"\b(correct|partial|wrong)\b", re.IGNORECASE)
_DIGIT_TO_VERDICT = {"1": "correct", "2": "partial", "3": "wrong"}


def _parse_verdict(raw: str) -> str:
    """Robust verdict extraction:
       1) primary: an explicit `VERDICT: <1|2|3>` line
       2) fallback: the LAST correct/partial/wrong word in the output
          (handles thinking-mode preambles that conclude with the verdict word)
    """
    m = _VERDICT_LINE_RE.search(raw)
    if m:
        return _DIGIT_TO_VERDICT[m.group(1)]
    words = _VERDICT_WORD_RE.findall(raw)
    if words:
        return words[-1].lower()
    return "unparseable"


# ---------------------------- Report scoring ------------------------------ #

def _score_report(n_limit: Optional[int], save_path: Path) -> dict:
    """Run MedGemma + CLIP retrieval on test set; collect predictions for metrics."""
    from src.report_mode import generate_report_medgemma, generate_report_clip

    cfg = _load_config()
    test_csv = _REPO_ROOT / cfg["data"]["manifest_test"]
    n_total = cfg["eval"]["report_test_n"]
    df = pd.read_csv(test_csv)
    n = min(n_limit or n_total, len(df))
    df = df.head(n).reset_index(drop=True)
    log.info("Report scoring: %d test images from %s", len(df), test_csv.name)

    items: list[dict] = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="report mode"):
        img_path = _REPO_ROOT / row["image_path"]
        gold = str(row["report"])
        record: dict = {"id": row["id"], "image_path": str(row["image_path"]), "gold": gold}
        try:
            with Image.open(img_path) as raw:
                img = raw.convert("RGB").copy()
        except Exception as e:
            record["error"] = f"image_load: {e}"
            items.append(record)
            continue

        try:
            r_clip = generate_report_clip(img)
            record["clip"] = {
                "report": r_clip["report"],
                "latency_s": r_clip["latency_s"],
                "retrieved_id": r_clip["extras"].get("retrieved_id"),
                "similarity": r_clip["extras"].get("similarity"),
            }
        except Exception as e:
            record["clip"] = {"error": str(e)}

        try:
            r_med = generate_report_medgemma(img)
            record["medgemma"] = {
                "report": r_med["report"],
                "latency_s": r_med["latency_s"],
            }
        except Exception as e:
            record["medgemma"] = {"error": str(e)}

        items.append(record)
        if (i + 1) % 5 == 0:
            _save_partial(save_path, {"report_mode_items": items})

    return {"items": items}


def _compute_report_metrics(items: list[dict]) -> dict:
    """ROUGE-L + BERTScore for both models against gold reports."""
    from rouge_score import rouge_scorer

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def _collect(model_key: str):
        preds, refs, latencies = [], [], []
        for it in items:
            m = it.get(model_key)
            if not isinstance(m, dict) or "report" not in m:
                continue
            preds.append(m["report"])
            refs.append(it["gold"])
            latencies.append(m.get("latency_s", 0.0))
        return preds, refs, latencies

    out: dict = {}
    for model_key, label in [("medgemma", "medgemma"), ("clip", "clip_retrieval")]:
        preds, refs, latencies = _collect(model_key)
        if not preds:
            out[label] = {"n": 0}
            continue
        rouge_f1 = [rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
        bert_f1_mean: Optional[float] = None
        try:
            from bert_score import score as bertscore
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _, _, F1 = bertscore(preds, refs, lang="en", device=device, verbose=False)
            bert_f1_mean = float(F1.mean().item())
        except Exception as e:
            log.warning("BERTScore failed for %s: %s", label, e)

        out[label] = {
            "n": len(preds),
            "rouge_l_f1_mean": float(sum(rouge_f1) / len(rouge_f1)),
            "bertscore_f1_mean": bert_f1_mean,
            "latency_s_mean": float(sum(latencies) / len(latencies)) if latencies else None,
        }
    return out


# ----------------------------- QA scoring --------------------------------- #

def _score_qa(n_limit: Optional[int], save_path: Path) -> dict:
    """For each QA item: run both retrievers, capture answers + retrieved IDs."""
    from src.qa_mode import qa_colpali_rag, qa_text_rag

    cfg = _load_config()
    qa_path = _REPO_ROOT / cfg["data"]["qa_dataset"]
    with open(qa_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    n_total = cfg["eval"]["qa_test_n"]

    flat: list[tuple[str, str, str]] = []
    for entry in ds:
        for pair in entry.get("qa_pairs", []):
            flat.append((entry["report_id"], pair["q"], pair["a"]))
    n = min(n_limit or n_total, len(flat))
    flat = flat[:n]
    log.info("QA scoring: %d questions", n)

    items: list[dict] = []
    for i, (rid, q, gold_a) in enumerate(tqdm(flat, desc="qa mode")):
        record: dict = {"gold_report_id": rid, "question": q, "gold_answer": gold_a}
        for fn, key in [(qa_text_rag, "text"), (qa_colpali_rag, "colpali")]:
            try:
                r = fn(q)
                top_ids = [str(x["id"]) for x in r["retrieved"]]
                record[key] = {
                    "answer": r["answer"],
                    "retrieved_ids": top_ids,
                    "gold_in_top_k": str(rid) in top_ids,
                    "latency_s": r["latency_s"],
                }
            except Exception as e:
                record[key] = {"error": str(e)}
        items.append(record)
        if (i + 1) % 5 == 0:
            _save_partial(save_path, {"qa_mode_items": items})

    return {"items": items}


def _judge_qa(qa_items: list[dict], save_path: Path) -> list[dict]:
    """Use MedGemma to score each predicted answer correct/partial/wrong vs gold."""
    from src.models import medgemma

    for i, rec in enumerate(tqdm(qa_items, desc="judge")):
        q = rec["question"]
        gold = rec["gold_answer"]
        for key in ("text", "colpali"):
            m = rec.get(key)
            if not isinstance(m, dict) or "answer" not in m:
                continue
            prompt = JUDGE_PROMPT.format(question=q, predicted=m["answer"], gold=gold)
            try:
                # 128 tokens: room for MedGemma thinking-mode preamble + VERDICT line.
                # Short anyway; doesn't materially change total judge runtime.
                raw = medgemma.generate(None, prompt, max_new_tokens=128)
            except Exception as e:
                m["judge"] = {"error": str(e)}
                continue
            verdict = _parse_verdict(raw)
            m["judge"] = {"verdict": verdict, "raw": raw[:200]}
        if (i + 1) % 5 == 0:
            _save_partial(save_path, {"qa_mode_items": qa_items})
    return qa_items


def _compute_qa_metrics(items: list[dict]) -> dict:
    """Aggregate Recall@k + judge counts per retriever."""
    out: dict = {}
    for key, label in [("text", "minilm-text"), ("colpali", "colpali")]:
        n = 0
        hits = 0
        verdicts = {"correct": 0, "partial": 0, "wrong": 0, "unparseable": 0, "error": 0}
        latencies = []
        for rec in items:
            m = rec.get(key)
            if not isinstance(m, dict) or "answer" not in m:
                verdicts["error"] += 1
                continue
            n += 1
            if m.get("gold_in_top_k"):
                hits += 1
            j = m.get("judge", {})
            if "verdict" in j:
                verdicts[j["verdict"]] = verdicts.get(j["verdict"], 0) + 1
            elif "error" in j:
                verdicts["error"] += 1
            else:
                verdicts["unparseable"] += 1
            latencies.append(m.get("latency_s", 0.0))
        out[label] = {
            "n": n,
            "recall_at_k": (hits / n) if n else None,
            "judge_counts": verdicts,
            "judge_accuracy": (verdicts["correct"] / n) if n else None,
            "latency_s_mean": (sum(latencies) / len(latencies)) if latencies else None,
        }
    return out


# ----------------------------- Output ------------------------------------- #

def _save_partial(path: Path, payload: dict) -> None:
    """Merge `payload` into the existing JSON at `path` (incremental save)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def _write_markdown(comparison: dict, md_path: Path) -> None:
    lines: list[str] = [
        "# Comparison Results",
        "",
        f"_Generated: {comparison.get('generated_at')}_",
        "",
    ]

    rm = comparison.get("report_mode", {})
    if rm.get("metrics"):
        sample_n = next(iter(rm["metrics"].values()), {}).get("n", "?")
        lines.append(f"## Report Generation (n={sample_n})")
        lines.append("")
        lines.append("| Model | ROUGE-L F1 | BERTScore F1 | Latency mean (s) |")
        lines.append("|---|---:|---:|---:|")
        for label, m in rm["metrics"].items():
            rouge_v = m.get("rouge_l_f1_mean")
            bert_v = m.get("bertscore_f1_mean")
            lat_v = m.get("latency_s_mean")
            lines.append(
                f"| {label} | "
                f"{'-' if rouge_v is None else f'{rouge_v:.3f}'} | "
                f"{'-' if bert_v is None else f'{bert_v:.3f}'} | "
                f"{'-' if lat_v is None else f'{lat_v:.2f}'} |"
            )
        lines.append("")

    qa = comparison.get("qa_mode", {})
    if qa.get("metrics"):
        sample_n = next(iter(qa["metrics"].values()), {}).get("n", "?")
        lines.append(f"## QA RAG (n={sample_n})")
        lines.append("")
        lines.append(
            "| Retriever | Recall@k | Judge accuracy | correct | partial | wrong | unparseable+err | latency mean (s) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for label, m in qa["metrics"].items():
            r = m.get("recall_at_k")
            ja = m.get("judge_accuracy")
            jc = m.get("judge_counts", {})
            lat_v = m.get("latency_s_mean")
            lines.append(
                f"| {label} | "
                f"{'-' if r is None else f'{r:.3f}'} | "
                f"{'-' if ja is None else f'{ja:.3f}'} | "
                f"{jc.get('correct', 0)} | {jc.get('partial', 0)} | {jc.get('wrong', 0)} | "
                f"{jc.get('unparseable', 0) + jc.get('error', 0)} | "
                f"{'-' if lat_v is None else f'{lat_v:.2f}'} |"
            )
        lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------- Main --------------------------------------- #

def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--limit-report", type=int, default=None)
    ap.add_argument("--limit-qa", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_config()
    json_out = _REPO_ROOT / cfg["eval"]["comparison_json"]
    md_out = _REPO_ROOT / cfg["eval"]["comparison_md"]
    comparison: dict = {
        "generated_at": _now_iso(),
        "config_snapshot": {
            "data": cfg["data"],
            "models": cfg["models"],
            "retrieval": cfg["retrieval"],
            "eval": cfg["eval"],
        },
    }

    if not args.skip_report:
        log.info("=== Report mode scoring ===")
        rep_data = _score_report(args.limit_report, json_out)
        log.info("Computing report metrics (ROUGE-L + BERTScore)...")
        rep_metrics = _compute_report_metrics(rep_data["items"])
        comparison["report_mode"] = {"items": rep_data["items"], "metrics": rep_metrics}
        _save_partial(json_out, comparison)

    if not args.skip_qa:
        log.info("=== QA mode scoring ===")
        qa_data = _score_qa(args.limit_qa, json_out)
        if not args.skip_judge:
            log.info("=== LLM-judge over QA answers ===")
            qa_data["items"] = _judge_qa(qa_data["items"], json_out)
        qa_metrics = _compute_qa_metrics(qa_data["items"])
        comparison["qa_mode"] = {"items": qa_data["items"], "metrics": qa_metrics}
        _save_partial(json_out, comparison)

    log.info("Writing Markdown summary -> %s", md_out)
    _write_markdown(comparison, md_out)
    log.info("Done. JSON: %s", json_out)
    log.info("       MD : %s", md_out)


if __name__ == "__main__":
    main()
