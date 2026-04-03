from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data_prep import UNKNOWN_GNAME_MARKERS


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STRUCTURED_FIELDS: tuple[tuple[str, str, float], ...] = (
    ("country_txt", "country", 0.16),
    ("region_txt", "region", 0.08),
    ("provstate", "province/state", 0.09),
    ("city", "city", 0.07),
    ("iyear", "year", 0.05),
    ("attacktype1_txt", "attack type", 0.12),
    ("targtype1_txt", "target type", 0.10),
    ("targsubtype1_txt", "target subtype", 0.06),
    ("weaptype1_txt", "weapon type", 0.08),
    ("weapsubtype1_txt", "weapon subtype", 0.05),
    ("corp1", "target organization", 0.07),
    ("target1", "target", 0.07),
)
_PROMPT_FIELDS: tuple[tuple[str, str], ...] = (
    ("eventid", "Event ID"),
    ("iyear", "Year"),
    ("country_txt", "Country"),
    ("region_txt", "Region"),
    ("provstate", "Province/State"),
    ("city", "City"),
    ("attacktype1_txt", "Attack Type"),
    ("targtype1_txt", "Target Type"),
    ("targsubtype1_txt", "Target Subtype"),
    ("weaptype1_txt", "Weapon Type"),
    ("weapsubtype1_txt", "Weapon Subtype"),
    ("corp1", "Target Organization"),
    ("target1", "Target"),
    ("summary", "Summary"),
    ("gname", "Group"),
)
_FIELD_LABELS = {field: label for field, label, _ in _STRUCTURED_FIELDS}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _require_local_model_dir(model_name: str, backend_name: str) -> Path:
    model_path = Path(model_name)
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"{backend_name} expected a local model at '{model_path}', but it was not found. "
            "Run scripts/download_reasoner_models.py first."
        )
    return model_path


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _field_similarity(query_value: Any, candidate_value: Any) -> float | None:
    query_text = _normalize_text(query_value)
    candidate_text = _normalize_text(candidate_value)
    if not query_text or not candidate_text:
        return None
    if query_text.lower() == candidate_text.lower():
        return 1.0

    query_tokens = _tokenize(query_text)
    candidate_tokens = _tokenize(candidate_text)
    if not query_tokens or not candidate_tokens:
        return 0.0
    shared = len(query_tokens & candidate_tokens)
    return float(shared) / float(max(len(query_tokens), len(candidate_tokens)))


def _year_similarity(query_value: Any, candidate_value: Any) -> float | None:
    query_text = _normalize_text(query_value)
    candidate_text = _normalize_text(candidate_value)
    if not query_text or not candidate_text:
        return None

    try:
        query_year = int(float(query_text))
        candidate_year = int(float(candidate_text))
    except ValueError:
        return _field_similarity(query_text, candidate_text)

    if query_year == candidate_year:
        return 1.0
    if abs(query_year - candidate_year) == 1:
        return 0.5
    return 0.0


def _summary_similarity(query_text: str, candidate_text: str) -> float | None:
    query_tokens = _tokenize(_normalize_text(query_text))
    candidate_tokens = _tokenize(_normalize_text(candidate_text))
    if not query_tokens or not candidate_tokens:
        return None
    return float(len(query_tokens & candidate_tokens)) / float(len(query_tokens))


def build_incident_profile(record: dict[str, Any]) -> dict[str, str]:
    """Build a compact incident profile that matcher/extractor backends can reuse."""
    profile: dict[str, str] = {}
    for field, _ in _PROMPT_FIELDS:
        source_value = record.get(field)
        if field == "summary" and not source_value:
            source_value = record.get("query_summary")
        value = _normalize_text(source_value)
        if value:
            profile[field] = value
    return profile


def incident_profile_to_text(record: dict[str, Any]) -> str:
    """Render an incident profile as a stable prompt block for local transformer backends."""
    profile = build_incident_profile(record)
    lines: list[str] = []
    for field, label in _PROMPT_FIELDS:
        value = profile.get(field)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _resolve_query_record(result: dict[str, Any]) -> dict[str, Any]:
    query_record = result.get("query_record")
    if isinstance(query_record, dict) and query_record:
        return dict(query_record)

    fallback: dict[str, Any] = {
        "eventid": result.get("eventid"),
        "summary": result.get("query_summary", ""),
    }
    for field, _ in _PROMPT_FIELDS:
        if field in fallback:
            continue
        if field in result:
            fallback[field] = result.get(field)
    return fallback


class BaseMatcher:
    method_name = "base_matcher"

    def match(
        self,
        query_record: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class FieldAwareMatcher(BaseMatcher):
    """Offline matcher that scores structured tactical/geographic alignment."""

    method_name = "field_weighted"

    def _base_score_candidate(
        self,
        query_record: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        enriched = dict(candidate)
        query_summary = _normalize_text(query_record.get("summary") or query_record.get("query_summary"))
        candidate_summary = _normalize_text(candidate.get("summary"))

        summary_score = _summary_similarity(query_summary, candidate_summary)

        feature_scores: dict[str, float] = {}
        reasons: list[str] = []
        weighted_score = 0.0
        possible_weight = 0.0

        for field, label, weight in _STRUCTURED_FIELDS:
            if field == "iyear":
                similarity = _year_similarity(query_record.get(field), candidate.get(field))
            else:
                similarity = _field_similarity(query_record.get(field), candidate.get(field))

            if similarity is None:
                continue

            feature_scores[field] = similarity
            possible_weight += weight
            weighted_score += weight * similarity

            candidate_value = _normalize_text(candidate.get(field))
            if similarity >= 0.99:
                reasons.append(f"same {label}: {candidate_value}")
            elif similarity >= 0.55:
                reasons.append(f"similar {label}: {candidate_value}")

        structured_score = (weighted_score / possible_weight) if possible_weight else None

        prior_score = 0.0
        if candidate.get("rerank_rank") is not None:
            prior_score = max(prior_score, 1.0 / max(_safe_float(candidate.get("rerank_rank"), 1.0), 1.0))
        if candidate.get("candidate_rank") is not None:
            prior_score = max(prior_score, 1.0 / max(_safe_float(candidate.get("candidate_rank"), 1.0), 1.0))
        prior_score = max(prior_score, min(1.0, _safe_float(candidate.get("reciprocal_rank_score"), 0.0)))

        weighted_components = {
            "summary": (summary_score, 0.40),
            "structured": (structured_score, 0.50),
            "prior": (prior_score, 0.10),
        }
        active_components = [
            (name, score, weight)
            for name, (score, weight) in weighted_components.items()
            if score is not None and score > 0.0
        ]
        if not active_components:
            matcher_score = 0.0
        else:
            total_weight = sum(weight for _, _, weight in active_components)
            matcher_score = sum(score * weight for _, score, weight in active_components) / total_weight

        if summary_score is not None and summary_score >= 0.35:
            reasons.append(f"summary overlap score {summary_score:.2f}")
        if prior_score >= 0.5:
            reasons.append("already highly ranked by retrieval/reranking")

        enriched["match_features"] = feature_scores
        enriched["summary_match_score"] = float(summary_score or 0.0)
        enriched["structured_match_score"] = float(structured_score or 0.0)
        enriched["prior_match_score"] = float(prior_score)
        enriched["matcher_score"] = float(matcher_score)
        enriched["matcher_method"] = self.method_name
        enriched["matcher_evidence"] = reasons[:6]
        return enriched

    def _finalize_ranking(
        self,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            candidates,
            key=lambda record: (
                -_safe_float(record.get("matcher_score"), 0.0),
                -_safe_float(record.get("structured_match_score"), 0.0),
                -_safe_float(record.get("summary_match_score"), 0.0),
                -_safe_float(record.get("reranker_score"), 0.0),
                -_safe_float(record.get("reciprocal_rank_score"), 0.0),
                str(record.get("eventid", "")),
            ),
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        for rank, record in enumerate(ranked, start=1):
            record["matcher_rank"] = rank
        return ranked

    def match(
        self,
        query_record: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        scored = [self._base_score_candidate(query_record, candidate) for candidate in candidates]
        return self._finalize_ranking(scored, top_k=top_k)


class LocalCrossEncoderMatcher(FieldAwareMatcher):
    """Cross-encoder matcher for practical local reranking-style pair scoring."""

    method_name = "cross_encoder_matcher"

    def __init__(
        self,
        model_name: str,
        max_length: int = 256,
        device: str | None = None,
        scorer: Any | None = None,
    ) -> None:
        self.max_length = max_length
        self._custom_scorer = scorer

        if scorer is not None:
            return

        if not model_name.strip():
            raise ValueError("cross_encoder_matcher requires a model name or local checkpoint path.")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "cross_encoder_matcher requires sentence-transformers to be installed."
            ) from exc

        model_path = _require_local_model_dir(model_name, self.method_name)
        local_files_only = model_path.exists()
        self._model = CrossEncoder(
            model_name_or_path=model_name,
            max_length=max_length,
            device=device,
            local_files_only=local_files_only,
        )

    def _score_pairs(
        self,
        query_record: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[float]:
        if self._custom_scorer is not None:
            return [float(score) for score in self._custom_scorer(query_record, candidates)]

        query_text = incident_profile_to_text(query_record)
        pairs = [(query_text, incident_profile_to_text(candidate)) for candidate in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]

    def match(
        self,
        query_record: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        base_scored = [self._base_score_candidate(query_record, candidate) for candidate in candidates]
        cross_encoder_scores = self._score_pairs(query_record, candidates)

        for enriched, cross_encoder_score in zip(base_scored, cross_encoder_scores):
            heuristic_score = _safe_float(enriched.get("matcher_score"), 0.0)
            enriched["transformer_match_score"] = cross_encoder_score
            enriched["matcher_score"] = (0.7 * cross_encoder_score) + (0.2 * heuristic_score) + (
                0.1 * _safe_float(enriched.get("prior_match_score"), 0.0)
            )
            enriched["matcher_method"] = self.method_name
            if cross_encoder_score >= 0.7:
                enriched["matcher_evidence"] = [
                    "local cross-encoder strongly favored this pair",
                    *enriched.get("matcher_evidence", []),
                ][:6]

        return self._finalize_ranking(base_scored, top_k=top_k)


class LocalRobertaMatcher(FieldAwareMatcher):
    """Transformer matcher for locally hosted RoBERTa-style pair classification models."""

    method_name = "roberta_matcher"

    def __init__(
        self,
        model_name: str,
        max_length: int = 256,
        batch_size: int = 8,
        device: str | None = None,
        scorer: Any | None = None,
    ) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        self._custom_scorer = scorer

        if scorer is not None:
            return

        if not model_name.strip():
            raise ValueError("roberta_matcher requires a model name or local checkpoint path.")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "roberta_matcher requires transformers and torch to be installed."
            ) from exc

        model_path = _require_local_model_dir(model_name, self.method_name)
        local_files_only = model_path.exists()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self._model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)

    def _score_pairs(
        self,
        query_record: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[float]:
        if self._custom_scorer is not None:
            return [float(score) for score in self._custom_scorer(query_record, candidates)]

        query_text = incident_profile_to_text(query_record)
        scores: list[float] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            candidate_texts = [incident_profile_to_text(candidate) for candidate in batch]
            encoded = self._tokenizer(
                [query_text] * len(candidate_texts),
                candidate_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}

            with self._torch.inference_mode():
                logits = self._model(**encoded).logits

            if logits.ndim == 1 or logits.shape[-1] == 1:
                probs = self._torch.sigmoid(logits.reshape(-1))
            else:
                probs = self._torch.softmax(logits, dim=-1)[:, -1]
            scores.extend(float(item) for item in probs.detach().cpu().tolist())
        return scores

    def match(
        self,
        query_record: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        base_scored = [self._base_score_candidate(query_record, candidate) for candidate in candidates]
        transformer_scores = self._score_pairs(query_record, candidates)

        for enriched, transformer_score in zip(base_scored, transformer_scores):
            heuristic_score = _safe_float(enriched.get("matcher_score"), 0.0)
            enriched["transformer_match_score"] = transformer_score
            enriched["matcher_score"] = (0.65 * transformer_score) + (0.25 * heuristic_score) + (
                0.10 * _safe_float(enriched.get("prior_match_score"), 0.0)
            )
            enriched["matcher_method"] = self.method_name
            if transformer_score >= 0.7:
                enriched["matcher_evidence"] = [
                    "local RoBERTa matcher strongly favored this pair",
                    *enriched.get("matcher_evidence", []),
                ][:6]

        return self._finalize_ranking(base_scored, top_k=top_k)


class BaseExtractor:
    method_name = "base_extractor"

    def extract(
        self,
        query_record: dict[str, Any],
        matched_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _shared_feature_labels(candidates: list[dict[str, Any]]) -> list[str]:
    feature_counter: Counter[str] = Counter()
    for candidate in candidates:
        for field, score in (candidate.get("match_features") or {}).items():
            if score >= 0.75 and field in _FIELD_LABELS:
                feature_counter[_FIELD_LABELS[field]] += 1
    return [label for label, _ in feature_counter.most_common(3)]


class EvidenceVoteExtractor(BaseExtractor):
    """Offline extractor that votes over matched candidates by group support."""

    method_name = "group_vote"

    def __init__(self, max_supporting_candidates: int = 3) -> None:
        self.max_supporting_candidates = max_supporting_candidates

    def extract(
        self,
        query_record: dict[str, Any],
        matched_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not matched_candidates:
            return {
                "status": "no_candidates",
                "predicted_gname": None,
                "confidence": 0.0,
                "confidence_label": "low",
                "decision_type": "none",
                "supporting_event_ids": [],
                "supporting_groups": [],
                "group_scores": [],
                "extractor_rationale": "No matched candidates were available for the reasoner.",
                "extractor_method": self.method_name,
            }

        grouped_scores: defaultdict[str, float] = defaultdict(float)
        grouped_candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

        for candidate in matched_candidates:
            gname = _normalize_text(candidate.get("gname"))
            if not gname or gname.lower() in UNKNOWN_GNAME_MARKERS:
                continue

            score = _safe_float(candidate.get("matcher_score"), 0.0)
            if score <= 0.0:
                continue

            grouped_scores[gname] += score
            grouped_candidates[gname].append(candidate)

        if not grouped_scores:
            return {
                "status": "insufficient_evidence",
                "predicted_gname": None,
                "confidence": 0.0,
                "confidence_label": "low",
                "decision_type": "none",
                "supporting_event_ids": [],
                "supporting_groups": [],
                "group_scores": [],
                "extractor_rationale": "Matched candidates did not provide a usable known group label.",
                "extractor_method": self.method_name,
            }

        ranked_groups = sorted(
            grouped_scores.items(),
            key=lambda item: (-item[1], -len(grouped_candidates[item[0]]), item[0].lower()),
        )
        predicted_gname, predicted_score = ranked_groups[0]
        total_score = sum(grouped_scores.values())
        confidence = (predicted_score / total_score) if total_score else 0.0
        confidence_label = _confidence_label(confidence)

        supporting_candidates = sorted(
            grouped_candidates[predicted_gname],
            key=lambda candidate: (
                -_safe_float(candidate.get("matcher_score"), 0.0),
                -_safe_float(candidate.get("reranker_score"), 0.0),
                str(candidate.get("eventid", "")),
            ),
        )[: self.max_supporting_candidates]
        supporting_event_ids = [candidate.get("eventid") for candidate in supporting_candidates]
        supporting_groups = [candidate.get("gname") for candidate in supporting_candidates]
        shared_fields = _shared_feature_labels(supporting_candidates)

        rationale_parts = [
            f"{predicted_gname} received the strongest cumulative matcher support from {len(grouped_candidates[predicted_gname])} candidate tuple(s)."
        ]
        if shared_fields:
            rationale_parts.append(
                "Top supporting incidents align on "
                + ", ".join(shared_fields[:-1] + [shared_fields[-1]] if len(shared_fields) == 1 else shared_fields)
                + "."
            )
        top_evidence = supporting_candidates[0].get("matcher_evidence", []) if supporting_candidates else []
        if top_evidence:
            rationale_parts.append("Best match evidence: " + "; ".join(top_evidence[:3]) + ".")

        group_scores = [
            {
                "gname": gname,
                "score": float(score),
                "support_count": len(grouped_candidates[gname]),
            }
            for gname, score in ranked_groups
        ]

        return {
            "status": "predicted",
            "predicted_gname": predicted_gname,
            "confidence": float(confidence),
            "confidence_label": confidence_label,
            "decision_type": "group_consensus" if len(grouped_candidates[predicted_gname]) > 1 else "top_candidate",
            "supporting_event_ids": supporting_event_ids,
            "supporting_groups": supporting_groups,
            "group_scores": group_scores,
            "extractor_rationale": " ".join(rationale_parts),
            "extractor_method": self.method_name,
        }


class LocalLlamaExtractor(EvidenceVoteExtractor):
    """Extractor wrapper for locally hosted causal LLMs with JSON output."""

    method_name = "llama_extractor"

    def __init__(
        self,
        model_name: str,
        max_input_candidates: int = 3,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        device: str | None = None,
        generator: Any | None = None,
    ) -> None:
        super().__init__(max_supporting_candidates=max_input_candidates)
        self.max_input_candidates = max_input_candidates
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._custom_generator = generator

        if generator is not None:
            return

        if not model_name.strip():
            raise ValueError("llama_extractor requires a model name or local checkpoint path.")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("llama_extractor requires transformers and torch to be installed.") from exc

        model_path = _require_local_model_dir(model_name, self.method_name)
        local_files_only = model_path.exists()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
        self._model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)

    def _build_prompt(
        self,
        query_record: dict[str, Any],
        matched_candidates: list[dict[str, Any]],
    ) -> str:
        candidate_blocks: list[str] = []
        for index, candidate in enumerate(matched_candidates[: self.max_input_candidates], start=1):
            block = incident_profile_to_text(candidate)
            candidate_blocks.append(
                f"Candidate {index}\n{block}\nMatcher Score: {_safe_float(candidate.get('matcher_score'), 0.0):.3f}"
            )

        return (
            "You are inferring the missing perpetrator group name for a GTD incident.\n"
            "Only choose a group that appears in the candidate incidents.\n"
            "Return strict JSON with keys: predicted_gname, confidence_label, rationale, support_event_ids.\n\n"
            "Unknown Incident\n"
            f"{incident_profile_to_text(query_record)}\n\n"
            "Candidate Incidents\n"
            f"{chr(10).join(candidate_blocks)}"
        )

    def _generate(self, prompt: str) -> str:
        if self._custom_generator is not None:
            return str(self._custom_generator(prompt))

        encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()

    def _parse_json(self, raw_text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def extract(
        self,
        query_record: dict[str, Any],
        matched_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        heuristic = super().extract(query_record, matched_candidates)
        heuristic["extractor_method"] = self.method_name
        if heuristic["status"] != "predicted":
            return heuristic

        prompt = self._build_prompt(query_record, matched_candidates)
        raw_output = self._generate(prompt)
        parsed = self._parse_json(raw_output)
        if not parsed:
            heuristic["extractor_rationale"] = (
                heuristic["extractor_rationale"] + " Local LLaMA extractor fell back to evidence voting."
            )
            return heuristic

        predicted_gname = _normalize_text(parsed.get("predicted_gname")) or heuristic["predicted_gname"]
        support_event_ids = parsed.get("support_event_ids")
        if not isinstance(support_event_ids, list):
            support_event_ids = heuristic["supporting_event_ids"]

        confidence_label = _normalize_text(parsed.get("confidence_label")).lower() or heuristic["confidence_label"]
        if confidence_label not in {"high", "medium", "low"}:
            confidence_label = heuristic["confidence_label"]

        confidence = heuristic["confidence"]
        if confidence_label == "high":
            confidence = max(confidence, 0.75)
        elif confidence_label == "medium":
            confidence = max(confidence, 0.5)

        rationale = _normalize_text(parsed.get("rationale")) or heuristic["extractor_rationale"]
        heuristic.update(
            {
                "predicted_gname": predicted_gname,
                "confidence": float(confidence),
                "confidence_label": confidence_label,
                "supporting_event_ids": support_event_ids,
                "extractor_rationale": rationale,
            }
        )
        return heuristic


def create_matcher(
    backend: str,
    model_name: str = "",
    max_length: int = 256,
) -> BaseMatcher:
    """Factory so later local matcher checkpoints can plug in without pipeline rewrites."""
    normalized = backend.strip().lower()
    if normalized == "field_weighted":
        return FieldAwareMatcher()
    if normalized == "cross_encoder_matcher":
        return LocalCrossEncoderMatcher(model_name=model_name, max_length=max_length)
    if normalized == "roberta_matcher":
        return LocalRobertaMatcher(model_name=model_name, max_length=max_length)
    raise ValueError(f"Unsupported matcher backend: {backend}")


def create_extractor(
    backend: str,
    model_name: str = "",
) -> BaseExtractor:
    """Factory for reasoner extractors, including future local LLaMA checkpoints."""
    normalized = backend.strip().lower()
    if normalized == "group_vote":
        return EvidenceVoteExtractor()
    if normalized == "llama_extractor":
        return LocalLlamaExtractor(model_name=model_name)
    raise ValueError(f"Unsupported extractor backend: {backend}")


def reason_over_reranked_results(
    reranked_results: list[dict[str, Any]],
    matcher: BaseMatcher,
    extractor: BaseExtractor,
    candidate_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run the week 7-8 reasoner stage over reranked candidates."""
    return [
        reason_single_result(
            result=result,
            matcher=matcher,
            extractor=extractor,
            candidate_limit=candidate_limit,
        )
        for result in reranked_results
    ]


def reason_single_result(
    result: dict[str, Any],
    matcher: BaseMatcher,
    extractor: BaseExtractor,
    candidate_limit: int | None = None,
) -> dict[str, Any]:
    """Reason over a single reranked query result so scripts can checkpoint progress."""
    query_record = _resolve_query_record(result)
    candidates = result.get("reranked_candidates") or result.get("candidate_pool") or []
    if candidate_limit is not None:
        candidates = candidates[:candidate_limit]

    matched_candidates = matcher.match(
        query_record=query_record,
        candidates=candidates,
        top_k=candidate_limit,
    )
    decision = extractor.extract(query_record=query_record, matched_candidates=matched_candidates)

    enriched = dict(result)
    enriched["query_record"] = query_record
    enriched["matched_candidates"] = matched_candidates
    enriched["matcher_method"] = matcher.method_name
    enriched.update(decision)
    return enriched
