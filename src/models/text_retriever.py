"""sentence-transformers text-RAG baseline (MiniLM-L6 + FAISS IndexFlatIP).

Same interface as colpali_retriever:
    build_index(manifest_path=None) -> None
    query(question: str, top_k: int = 3) -> list[dict]

Embeddings are L2-normalized so inner-product == cosine similarity.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import faiss
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG_PATH = _REPO_ROOT / "config.yaml"

_state: dict = {"model": None, "index": None, "manifest": None}
_lock = threading.RLock()


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_model() -> SentenceTransformer:
    if _state["model"] is not None:
        return _state["model"]
    with _lock:
        if _state["model"] is not None:
            return _state["model"]
        cfg = _load_config()
        model_id = cfg["models"]["text_retriever"]["hf_id"]
        model = SentenceTransformer(model_id, device=_device())
        _state["model"] = model
        return model


def _index_paths() -> tuple[Path, Path]:
    cfg = _load_config()
    idx_path = _REPO_ROOT / cfg["models"]["text_retriever"]["index_path"]
    manifest_snapshot = idx_path.with_suffix(".manifest.csv")
    return idx_path, manifest_snapshot


def build_index(manifest_path: Optional[Path] = None) -> None:
    cfg = _load_config()
    if manifest_path is None:
        manifest_path = _REPO_ROOT / cfg["data"]["manifest_index"]
    idx_path, manifest_snapshot = _index_paths()
    model = _ensure_model()

    df = pd.read_csv(manifest_path)
    reports = df["report"].astype(str).tolist()
    print(f"[text] embedding {len(reports)} reports...", flush=True)
    embs = model.encode(
        reports,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via L2-normalized inner product
    index.add(embs)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    df.to_csv(manifest_snapshot, index=False)
    _state["index"] = index
    _state["manifest"] = df
    print(f"[text] index written: {idx_path} ({index.ntotal} vectors, dim={embs.shape[1]})")


def _ensure_index() -> tuple:
    if _state["index"] is not None and _state["manifest"] is not None:
        return _state["index"], _state["manifest"]
    with _lock:
        if _state["index"] is not None and _state["manifest"] is not None:
            return _state["index"], _state["manifest"]
        idx_path, manifest_snapshot = _index_paths()
        if idx_path.exists() and manifest_snapshot.exists():
            _state["index"] = faiss.read_index(str(idx_path))
            _state["manifest"] = pd.read_csv(manifest_snapshot)
        else:
            build_index()
        return _state["index"], _state["manifest"]


def query(question: str, top_k: int = 3) -> list[dict]:
    index, manifest = _ensure_index()
    model = _ensure_model()
    emb = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    sims, idxs = index.search(emb, top_k)
    out: list[dict] = []
    for sim, idx in zip(sims[0], idxs[0]):
        row = manifest.iloc[int(idx)]
        out.append({
            "id": row["id"],
            "report": row["report"],
            "score": float(sim),
            "image_path": row["image_path"],
        })
    return out


if __name__ == "__main__":
    build_index()
