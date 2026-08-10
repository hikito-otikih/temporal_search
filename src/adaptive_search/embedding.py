"""Embedding contracts and optional local adapters for adaptive refinement.

The temporal algorithms only consume scalar similarity curves.  Keeping model
inference behind this module makes those algorithms deterministic and lets a
deployment run the encoder in-process or in a separate GPU worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
QUALITY_QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
QUALITY_QWEN_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_QWEN_INSTRUCTION = (
    "Retrieve video frames that visually depict the described event."
)


def is_immutable_hf_commit(revision: str | None) -> bool:
    """Return true only for a full Hugging Face/Git commit identifier."""

    return isinstance(revision, str) and bool(
        re.fullmatch(r"[0-9a-fA-F]{40}", revision)
    )


class EmbeddingConfigurationError(RuntimeError):
    """Raised when an optional model runtime is unavailable or inconsistent."""


@runtime_checkable
class TextImageEmbedder(Protocol):
    """Minimal dual-encoder contract used by the refinement stage."""

    model_id: str
    revision: str
    output_dimension: int
    preprocess_version: str
    instruction: str

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_images(self, images: Sequence[Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class SimilarityCurves:
    """Raw cosine curves for one event over ordered frames."""

    anchor: list[float]
    pre: list[float]
    post: list[float]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 vectors.

    Zero vectors are rejected instead of silently producing NaNs or misleading
    zero similarities.
    """

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings must be a rank-2 array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings must not contain zero vectors")
    return array / norms


def _siglip_feature_array(
    output: Any,
    *,
    expected_rows: int,
    expected_dimension: int,
    modality: str,
) -> np.ndarray:
    """Normalize SigLIP features across Transformers output conventions.

    Transformers 5 returns ``BaseModelOutputWithPooling`` from SigLIP2's
    ``get_*_features`` helpers, while older releases returned the pooled tensor
    directly. Callers force ``return_dict=True`` so a tuple cannot be confused
    with the rank-2 pooled representation.
    """

    feature_tensor = output if hasattr(output, "detach") else None
    if feature_tensor is None:
        feature_tensor = getattr(output, "pooler_output", None)
    if feature_tensor is None or not hasattr(feature_tensor, "detach"):
        raise EmbeddingConfigurationError(
            f"SigLIP2 {modality} encoder returned no pooled feature tensor"
        )

    try:
        array = np.asarray(
            feature_tensor.detach().float().cpu().numpy(),
            dtype=np.float32,
        )
    except Exception as exc:
        raise EmbeddingConfigurationError(
            f"could not convert SigLIP2 {modality} features to an array"
        ) from exc
    expected_shape = (expected_rows, expected_dimension)
    if array.shape != expected_shape:
        raise EmbeddingConfigurationError(
            f"SigLIP2 {modality} feature shape must be {expected_shape}, "
            f"got {array.shape}"
        )
    try:
        return l2_normalize(array)
    except ValueError as exc:
        raise EmbeddingConfigurationError(
            f"SigLIP2 {modality} features are not cosine-compatible"
        ) from exc


def truncate_mrl(vectors: np.ndarray, dimension: int) -> np.ndarray:
    """Truncate Matryoshka embeddings and normalize the truncated prefix.

    Qwen's full vector must be requested with ``normalize=False`` before this
    function is called.  Truncating an already normalized full vector without
    re-normalizing would not produce cosine-compatible vectors.
    """

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings must be a rank-2 array")
    if dimension <= 0 or dimension > array.shape[1]:
        raise ValueError(
            f"dimension must be in [1, {array.shape[1]}], got {dimension}"
        )
    return l2_normalize(array[:, :dimension])


def cosine_scores(text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarities after defensive normalization."""

    text = l2_normalize(text_embeddings)
    images = l2_normalize(image_embeddings)
    if text.shape[1] != images.shape[1]:
        raise ValueError("text and image embedding dimensions must match")
    return text @ images.T


def score_event_frames(
    embedder: TextImageEmbedder,
    *,
    anchor_query: str,
    pre_state_query: str,
    post_state_query: str,
    images: Sequence[Any],
    image_batch_size: int | None = None,
) -> SimilarityCurves:
    """Encode three observable state descriptions against the same frames."""

    if not images:
        return SimilarityCurves(anchor=[], pre=[], post=[])
    queries = [anchor_query, pre_state_query, post_state_query]
    if any(not query.strip() for query in queries):
        raise ValueError("anchor, pre-state and post-state queries must be non-empty")
    if (
        image_batch_size is not None
        and (
            isinstance(image_batch_size, bool)
            or not isinstance(image_batch_size, int)
            or image_batch_size <= 0
        )
    ):
        raise ValueError("image_batch_size must be a positive integer")
    text_embeddings = embedder.encode_texts(queries)
    batch_size = image_batch_size or len(images)
    matrices = []
    for start in range(0, len(images), batch_size):
        image_embeddings = embedder.encode_images(images[start : start + batch_size])
        matrices.append(cosine_scores(text_embeddings, image_embeddings))
    matrix = np.concatenate(matrices, axis=1)
    if matrix.shape != (3, len(images)):
        raise ValueError(
            "embedder returned an unexpected number of text/image embeddings"
        )
    return SimilarityCurves(
        anchor=matrix[0].astype(float).tolist(),
        pre=matrix[1].astype(float).tolist(),
        post=matrix[2].astype(float).tolist(),
    )


class Siglip2Embedder:
    """Lazy Transformers adapter for the practical 8-GB-GPU profile.

    ``torch`` and ``transformers`` are optional dependencies and are imported
    only when an instance is created.  Model weights are never downloaded at
    module import time.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SIGLIP2_MODEL,
        revision: str,
        device: str | None = None,
        torch_dtype: str = "float16",
        preprocess_version: str = "siglip2-auto-processor-v1",
    ) -> None:
        if not is_immutable_hf_commit(revision):
            raise EmbeddingConfigurationError(
                "revision must be a full 40-character hexadecimal commit"
            )
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional ML environment
            raise EmbeddingConfigurationError(
                "SigLIP2 requires the optional torch and transformers packages"
            ) from exc

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, torch_dtype, None)
        if dtype is None:
            raise EmbeddingConfigurationError(f"unsupported torch dtype: {torch_dtype}")

        self.model_id = model_id
        self.revision = revision
        self.preprocess_version = preprocess_version
        self.instruction = ""
        self._torch = torch
        self._device = resolved_device
        self._processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self._model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
        ).to(resolved_device).eval()
        projection_dim = getattr(self._model.config, "projection_dim", None)
        if projection_dim is None:
            text_config = getattr(self._model.config, "text_config", None)
            projection_dim = getattr(text_config, "projection_size", None)
        if not projection_dim:
            raise EmbeddingConfigurationError(
                "could not determine SigLIP2 projection dimension from model config"
            )
        self.output_dimension = int(projection_dim)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.output_dimension), dtype=np.float32)
        inputs = self._processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.inference_mode():
            features = self._model.get_text_features(**inputs, return_dict=True)
        return _siglip_feature_array(
            features,
            expected_rows=len(texts),
            expected_dimension=self.output_dimension,
            modality="text",
        )

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        if not images:
            return np.empty((0, self.output_dimension), dtype=np.float32)
        inputs = self._processor(images=list(images), return_tensors="pt").to(
            self._device
        )
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs, return_dict=True)
        return _siglip_feature_array(
            features,
            expected_rows=len(images),
            expected_dimension=self.output_dimension,
            modality="image",
        )


class Qwen3VLEmbedderAdapter:
    """Adapter around Qwen's official ``Qwen3VLEmbedder`` implementation.

    The official project currently exposes its own wrapper rather than a stable
    Transformers auto class.  The deployment creates that wrapper and injects
    it here, keeping this repository independent of a vendored model checkout.
    """

    def __init__(
        self,
        backend: Any,
        *,
        revision: str,
        output_dimension: int = 1024,
        model_id: str = QUALITY_QWEN_EMBEDDING_MODEL,
        preprocess_version: str = "qwen3-vl-official-v1",
        instruction: str = DEFAULT_QWEN_INSTRUCTION,
    ) -> None:
        if not revision or revision == "main":
            raise EmbeddingConfigurationError(
                "revision must be an immutable model commit for reproducible runs"
            )
        if output_dimension < 64 or output_dimension > 2048:
            raise EmbeddingConfigurationError(
                "Qwen3-VL-Embedding-2B output_dimension must be in [64, 2048]"
            )
        if not hasattr(backend, "process"):
            raise EmbeddingConfigurationError("backend must expose process(inputs, ...)")
        self._backend = backend
        self.model_id = model_id
        self.revision = revision
        self.output_dimension = output_dimension
        self.preprocess_version = preprocess_version
        self.instruction = instruction

    def _encode(self, inputs: list[dict[str, Any]]) -> np.ndarray:
        try:
            raw = self._backend.process(inputs, normalize=False)
        except TypeError as exc:
            raise EmbeddingConfigurationError(
                "official Qwen backend must support process(..., normalize=False)"
            ) from exc
        if hasattr(raw, "detach"):
            raw = raw.detach().float().cpu().numpy()
        return truncate_mrl(np.asarray(raw), self.output_dimension)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(
            [
                {"text": text, "instruction": self.instruction}
                for text in texts
            ]
        )

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        return self._encode([{"image": image} for image in images])


def image_embedding_cache_key(
    *,
    video_content_hash: str,
    pts_ms: int,
    model_id: str,
    revision: str,
    output_dimension: int,
    preprocess_version: str,
) -> str:
    """Build a stable image key; sampling policy is intentionally excluded."""

    if pts_ms < 0:
        raise ValueError("pts_ms must be non-negative")
    return _fingerprint(
        {
            "kind": "image_embedding",
            "video_content_hash": video_content_hash,
            "pts_ms": pts_ms,
            "model_id": model_id,
            "revision": revision,
            "output_dimension": output_dimension,
            "preprocess_version": preprocess_version,
        }
    )


def text_embedding_cache_key(
    *,
    text: str,
    instruction: str,
    model_id: str,
    revision: str,
    output_dimension: int,
    preprocess_version: str,
) -> str:
    return _fingerprint(
        {
            "kind": "text_embedding",
            "text": text,
            "instruction": instruction,
            "model_id": model_id,
            "revision": revision,
            "output_dimension": output_dimension,
            "preprocess_version": preprocess_version,
        }
    )


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()
