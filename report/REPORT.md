# Chest X-Ray Multi-Modal Intelligence System

**DSAI 413 — Assignment 2**

---

## 1. Problem

Two operating modes over the MIMIC-CXR dataset:

1. **Report generation** — given a chest X-ray image, produce a radiology report (Findings + Impression).
2. **Question answering (RAG)** — given a natural-language clinical question, retrieve relevant indexed reports and generate a grounded answer.

The work compares one **mandatory model** against one **lighter baseline** in each mode, all zero-shot, on a 400-sample MIMIC-CXR subset (300 indexed, 100 held-out test).

| Mode | Mandatory | Baseline |
|---|---|---|
| Report generation | MedGemma 1.5-4B-IT (multimodal LM) | OpenCLIP ViT-B/32 nearest-neighbor retrieval |
| QA retrieval | ColPali v1.3 (vision-LM late interaction) | sentence-transformers MiniLM-L6 (text bi-encoder) |
| QA answer | MedGemma 1.5-4B-IT (text-only with retrieved context) | — |

---

## 2. Architecture

```mermaid
flowchart TB
    subgraph Data
        K[Kaggle: MIMIC-CXR<br/>simhadrisadaram] --> D[data/download.py]
        D --> M[manifest.csv<br/>id, image_path, report,<br/>subject_id, view_used]
        M --> MI[manifest_index.csv<br/>300 train]
        M --> MT[manifest_test.csv<br/>100 test]
    end

    subgraph "Report Mode"
        IMG[Chest X-ray]
        IMG --> MG[MedGemma 4-bit]
        MG --> R1[Report]
        IMG --> CLIP[OpenCLIP ViT-B/32]
        CLIP --> FAISS_IMG[FAISS IndexFlatIP<br/>over 300 imgs]
        FAISS_IMG --> R2[Nearest report]
    end

    subgraph "QA RAG Mode"
        Q[Question] --> CP[ColPali v1.3<br/>multi-vector]
        MI -.render report PNG.-> CPI[ColPali index<br/>doc_embeddings.pt]
        CP --> CPI
        CPI --> TOPK1[top-3 reports]
        Q --> ML[MiniLM-L6]
        MI -.encode text.-> MLI[FAISS IndexFlatIP<br/>text_index.faiss]
        ML --> MLI
        MLI --> TOPK2[top-3 reports]
        TOPK1 --> MG2[MedGemma context+Q]
        TOPK2 --> MG2
        MG2 --> ANS[Answer]
    end

    subgraph Evaluation
        R1 & R2 --> METR1[ROUGE-L + BERTScore]
        ANS --> METR2[Recall@3 + LLM-judge]
    end
```

The same `MedGemma` singleton serves three jobs (report generation, RAG answering, LLM-as-judge in eval), avoiding repeated 4-bit loads.

---

## 3. Model choices

### MedGemma 1.5-4B-IT (`google/medgemma-1.5-4b-it`)
- Mandatory per the assignment.
- Loaded in **4-bit nf4** with `bitsandbytes` so it fits alongside ColPali on a Colab T4 (15 GB).
- Multimodal: handles image+text for report generation; text-only for QA answering and judging.
- Generation: greedy decoding (`do_sample=False`) for reproducibility.

### ColPali v1.3 (`vidore/colpali-v1.3-merged`)
- Mandatory per the assignment. We use the `-merged` variant (LoRA pre-merged into the base PaliGemma weights) to avoid PEFT adapter-key remap incompatibilities with current transformers.
- 4-bit nf4 quantized.
- Reports are rendered as single-page PNGs (PIL + DejaVu Sans 20px) at 1024×1280, then embedded as multi-vector representations. Queries are scored via late-interaction MaxSim through `processor.score_multi_vector`.

### Baselines
- **OpenCLIP ViT-B/32** (`openai` weights) for image-to-report retrieval. L2-normalized 512-d embeddings, FAISS `IndexFlatIP` over the 300 train images.
- **sentence-transformers/all-MiniLM-L6-v2** for text-to-report retrieval. L2-normalized 384-d embeddings, FAISS `IndexFlatIP` over the 300 train reports.

---

## 4. Dataset and QA generation

### MIMIC-CXR subset
- **Source:** `simhadrisadaram/mimic-cxr-dataset` via `kagglehub`. Two CSVs (`mimic_cxr_aug_train.csv`, `mimic_cxr_aug_validate.csv`) with ~65k patient-rows.
- **Each row = one patient**, with stringified Python lists in `image`, `view`, `AP`, `PA`, `Lateral`, `text`. Parsed via `ast.parse` (safe literal eval) into Python lists.
- **Filtering funnel** (logged by `data/download.py --inspect-csv`):

  | Stage | Count |
  |---|---:|
  | Total rows | 65,086 |
  | All lists parseable | 65,086 |
  | Exactly-one non-empty report (`'Findings:' in t OR len > 50`) | 32,680 |
  | Has ≥ 1 PA or AP image | 31,563 |
  | Image file resolves on disk | **22,783 (eligible pool)** |

- **Sampling:** 400 patients drawn with `seed=42`, view priority `PA > AP > other`. Split into 300 index / 100 test (re-shuffled with same seed).
- **Manifest schema:** `id, image_path, report, subject_id, view_used`.

### QA dataset
- For 25 reports sampled from `manifest_index.csv`, MedGemma is prompted (text-only) to emit a strict JSON list of 3 QA pairs per report — one yes/no, one open-ended, one location/laterality.
- The parser is defensive: tries direct `json.loads`, strips ` ```json ` fences, falls back to `json.JSONDecoder().raw_decode` scanning every `[` (survives MedGemma 1.5's `<unused94>thought` preambles).
- Output: `data/sample/qa_dataset.json` with `{report_id, image_path, report, qa_pairs:[{q, a}, …]}`.
- 25 reports × 3 pairs = **75 QA items**, comfortably covering the 50-item eval budget after parse-failure attrition.

---

## 5. Evaluation methodology

### Report mode (n=100 test images)
- **ROUGE-L F1** (per-pair, then mean) via `rouge_score`.
- **BERTScore F1** (vectorized over all pairs) via `bert_score`, `lang="en"` (RoBERTa-large encoder).

### QA RAG mode (n=50 questions, drawn from `qa_dataset.json`)
- **Recall@3**: fraction of questions where the gold patient's report appears among the top-3 retrieved IDs.
- **LLM-as-judge**: for each (question, predicted answer, gold answer) triple, MedGemma is asked to emit `VERDICT: <1|2|3>` (1=correct / 2=partial / 3=wrong). Parser uses the explicit `VERDICT:` regex with a last-word fallback to handle thinking-mode preambles. **Judge accuracy** = correct / n.

### Honest limitations of the protocol
- **Same model is both system-under-test AND judge.** Standard LLM-as-judge weakness; the judge may favor outputs in MedGemma's style. The Recall@k metric is independent of this.
- **Generic clinical questions** (e.g. "Is there a pleural effusion?") have no patient-specific anchor, so retrieval Recall@k is expected to be low across both retrievers (~0). This biases the comparison toward the answering step rather than retrieval quality per se.

---

## 6. Results

_Numbers below come from `results/comparison.json` (populated by `python -m src.eval`)._

### Report Generation (n=100)

| Model | ROUGE-L F1 | BERTScore F1 | Latency mean (s) |
|---|---:|---:|---:|
| MedGemma 1.5-4B | _<from comparison.md>_ | _<from comparison.md>_ | _<from comparison.md>_ |
| OpenCLIP retrieval | _<from comparison.md>_ | _<from comparison.md>_ | _<from comparison.md>_ |

### QA RAG (n=50)

| Retriever | Recall@3 | Judge accuracy | correct | partial | wrong | unparseable+err | latency mean (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ColPali v1.3 | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ |
| MiniLM-L6 text | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ |

### Qualitative observations from the smoke run (n=5)
- MedGemma's reports follow the requested `FINDINGS: … IMPRESSION: …` structure consistently.
- It accurately catches obvious findings (sternotomy wires, patient rotation, mild cardiomegaly) but under-calls subtle ones; default verdict is "No acute cardiopulmonary process."
- CLIP retrieval surfaces structurally similar X-rays (most CXRs look broadly similar), so its retrieved reports often share boilerplate even when patient-specific findings differ. ROUGE-L sees through this; BERTScore less so (similar vocabulary).
- ColPali retrieval similarities are well-separated in score, but the questions are too generic for top-3 to land on the specific gold patient. Both retrievers fail Recall@3 ≈ symmetrically.

---

## 7. Limitations

1. **Subset, not full MIMIC-CXR.** 400 pairs is statistically small; results indicate trend, not absolute capability. The full dataset has ~250k images.
2. **Zero-shot only.** No domain fine-tuning of MedGemma or ColPali on MIMIC-CXR; an in-domain finetune would likely materially improve both modes.
3. **LLM-as-judge with same model.** MedGemma judging its own outputs is a known bias source; cross-model judges (e.g. GPT-4, Claude) would be more rigorous if API access were in scope.
4. **Rendered-report ColPali pages are synthetic**, not native scanned documents. ColPali was trained on real document pages; basic PIL typography on a blank canvas may not exercise its layout-understanding strength.
5. **Generic clinical questions.** Auto-generated QA pairs are not patient-specific (no anchor to identify which patient's report is gold), so retrieval Recall@k is structurally low regardless of retriever quality.
6. **MedGemma 1.5 thinking-mode preambles.** The `<unused94>thought` token leaks structured reasoning before JSON or verdict outputs. Mitigated by defensive parsers and bumped `max_new_tokens`, but ~5–15% of judge outputs still need fallback parsing.
7. **Single Colab T4 footprint.** MedGemma 4-bit + ColPali 4-bit fits tightly in 15 GB VRAM; bf16 ColPali would not coexist. On larger GPUs (A100), 8-bit or bf16 ColPali would give cleaner numbers.

---

## 8. Future work

- **Patient-grounded QA generation:** include patient-specific markers (view, modality, any imaging finding token) in generated questions so retrieval has something to anchor on.
- **Cross-model judges:** add a second judge (e.g. Claude or GPT-4) to debias the LLM-judge accuracy.
- **In-domain fine-tune:** LoRA-tune MedGemma on a subset of MIMIC-CXR (image + report pairs) and re-evaluate; this is the lever most likely to move ROUGE-L meaningfully.
- **Stronger baseline:** swap MiniLM for a clinical-text encoder (e.g. `emilyalsentzer/Bio_ClinicalBERT` mean-pooled) — better in-domain semantics for medical text.
- **Native PDF rendering for ColPali:** convert reports to actual PDF (e.g. via `reportlab`) so ColPali sees document-like layouts rather than plain text on white.
- **Confidence calibration:** have MedGemma emit uncertainty alongside the answer; couple with retrieval similarity to abstain when neither retrieval nor confidence support an answer.

---

## Appendix: Repo layout

```
chest-xray-system/
├── README.md, config.yaml, requirements.txt, .env.example
├── data/
│   ├── download.py            # Kaggle pull + filter + 400-sample split
│   ├── build_qa_dataset.py    # MedGemma generates 3 QA pairs per report
│   └── sample/                # gitignored at-runtime data + indexes
├── src/
│   ├── models/
│   │   ├── medgemma.py           # 4-bit MedGemma singleton
│   │   ├── colpali_retriever.py  # ColPali index + multi-vector scoring
│   │   ├── clip_retriever.py     # OpenCLIP image retrieval
│   │   └── text_retriever.py     # MiniLM text retrieval
│   ├── report_mode.py            # generate_report_medgemma / _clip
│   ├── qa_mode.py                # qa_colpali_rag / qa_text_rag
│   └── eval.py                   # ROUGE + BERTScore + judge
├── app/app.py                    # Gradio demo (Report / QA tabs)
├── notebooks/run_on_colab.ipynb  # Colab launch flow
├── results/                      # comparison.json + comparison.md
└── report/REPORT.md              # this document
```
