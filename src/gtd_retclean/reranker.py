from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_pool import build_candidate_pool
from .config import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_RERANKER_MODEL,
)


def late_interaction_score(query_token_vectors: np.ndarray, doc_token_vectors: np.ndarray) -> float:
    """Score a query/document pair with ColBERT-style MaxSim late interaction."""
    if query_token_vectors.ndim != 2 or doc_token_vectors.ndim != 2:
        raise ValueError("Late interaction expects 2D token embedding matrices.")
    if query_token_vectors.shape[0] == 0 or doc_token_vectors.shape[0] == 0:
        return 0.0

    similarities = query_token_vectors @ doc_token_vectors.T
    return float(similarities.max(axis=1).mean())


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class BaseReranker:
    method_name = "base"

    def rerank(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class TokenOverlapReranker(BaseReranker):
    """A lightweight fallback reranker for offline environments."""

    method_name = "token_overlap"

    def score_pair(self, query_text: str, document_text: str) -> float:
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return 0.0

        document_tokens = _tokenize(document_text)
        return float(len(query_tokens & document_tokens)) / float(len(query_tokens))

    def rerank(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            document_text = str(candidate.get("summary", ""))
            enriched = dict(candidate)
            enriched["reranker_score"] = self.score_pair(query_text, document_text)
            enriched["reranker_method"] = self.method_name
            scored.append(enriched)

        ranked = sorted(
            scored,
            key=lambda record: (
                -record["reranker_score"],
                -record.get("reciprocal_rank_score", 0.0),
                str(record.get("eventid", "")),
            ),
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        for rank, candidate in enumerate(ranked, start=1):
            candidate["rerank_rank"] = rank
        return ranked


class TransformerLateInteractionEncoder:
    """Encode token embeddings that a late-interaction reranker can consume.

    If the checkpoint contains a ColBERT projection layer (`linear.weight`),
    apply it so local ColBERT models can be used directly from disk.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Late interaction reranking requires transformers and torch."
            ) from exc

        model_path = Path(model_name)
        local_files_only = model_path.exists()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self._model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)
        self.max_length = max_length
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}
        self._projection = self._load_projection_matrix(model_path if local_files_only else None)

    def _load_projection_matrix(self, model_path: Path | None):
        if model_path is None or not model_path.exists():
            return None

        safetensors_path = model_path / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file
            except ImportError:
                return None

            state = load_file(str(safetensors_path))
            if "linear.weight" in state:
                return state["linear.weight"].to(self.device)

        torch_path = model_path / "pytorch_model.bin"
        if torch_path.exists():
            state = self._torch.load(torch_path, map_location=self.device)
            if "linear.weight" in state:
                return state["linear.weight"].to(self.device)

        return None

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        uncached_texts = [text for text in texts if text not in self._cache]

        for start in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[start : start + self.batch_size]
            if not batch:
                continue

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}

            with self._torch.inference_mode():
                model_outputs = self._model(**encoded)
                token_embeddings = model_outputs.last_hidden_state

            attention_mask = encoded["attention_mask"].bool()
            input_ids = encoded["input_ids"]
            special_mask = self._torch.zeros_like(attention_mask)
            for special_id in self._tokenizer.all_special_ids:
                special_mask |= input_ids == special_id
            valid_mask = attention_mask & ~special_mask

            output_dim = self._projection.shape[0] if self._projection is not None else token_embeddings.shape[-1]
            for text, token_matrix, row_mask in zip(batch, token_embeddings, valid_mask):
                selected = token_matrix[row_mask]
                if selected.shape[0] == 0:
                    array = np.zeros((1, output_dim), dtype="float32")
                else:
                    if self._projection is not None:
                        selected = selected @ self._projection.T
                    selected = self._torch.nn.functional.normalize(selected, p=2, dim=-1)
                    array = selected.detach().cpu().numpy().astype("float32")
                self._cache[text] = array

        for text in texts:
            outputs.append(self._cache[text])
        return outputs


class LateInteractionReranker(BaseReranker):
    """ColBERT-style reranker built on token-level MaxSim scoring."""

    method_name = "late_interaction"

    def __init__(
        self,
        encoder: Any | None = None,
        model_name: str = DEFAULT_RERANKER_MODEL,
        max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> None:
        self.encoder = encoder or TransformerLateInteractionEncoder(
            model_name=model_name,
            max_length=max_length,
        )

    def rerank(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        query_tokens = self.encoder.encode_texts([query_text])[0]
        document_texts = [str(candidate.get("summary", "")) for candidate in candidates]
        document_tokens = self.encoder.encode_texts(document_texts)

        scored: list[dict[str, Any]] = []
        for candidate, token_matrix in zip(candidates, document_tokens):
            enriched = dict(candidate)
            enriched["reranker_score"] = late_interaction_score(query_tokens, token_matrix)
            enriched["reranker_method"] = self.method_name
            scored.append(enriched)

        ranked = sorted(
            scored,
            key=lambda record: (
                -record["reranker_score"],
                -record.get("reciprocal_rank_score", 0.0),
                str(record.get("eventid", "")),
            ),
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        for rank, candidate in enumerate(ranked, start=1):
            candidate["rerank_rank"] = rank
        return ranked


def create_reranker(
    backend: str,
    model_name: str = DEFAULT_RERANKER_MODEL,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> BaseReranker:
    """Factory so later weeks can swap reranker implementations cleanly."""
    normalized = backend.strip().lower()
    if normalized == "late_interaction":
        return LateInteractionReranker(model_name=model_name, max_length=max_length)
    if normalized == "token_overlap":
        return TokenOverlapReranker()
    raise ValueError(f"Unsupported reranker backend: {backend}")


def rerank_retrieval_results(
    retrieval_results: list[dict[str, Any]],
    reranker: BaseReranker,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank each query's merged candidate pool and preserve retrieval context."""
    reranked_outputs: list[dict[str, Any]] = []

    for result in retrieval_results:
        candidate_pool = result.get("candidate_pool") or build_candidate_pool(
            es_candidates=result.get("es_candidates", []),
            faiss_candidates=result.get("faiss_candidates", []),
        )
        reranked = reranker.rerank(
            query_text=str(result.get("query_summary", "")),
            candidates=candidate_pool,
            top_k=top_k,
        )

        enriched = dict(result)
        enriched["candidate_pool"] = candidate_pool
        enriched["reranked_candidates"] = reranked
        enriched["reranker_method"] = reranker.method_name
        reranked_outputs.append(enriched)

    return reranked_outputs
