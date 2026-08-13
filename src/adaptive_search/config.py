"""Environment wiring for optional local YouCook2 live refinement.

Model weights and multimedia libraries are never loaded at API import time.
The default application remains model-free unless the operator explicitly sets
the YouCook2 root and an immutable SigLIP2 revision.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from threading import RLock
from typing import Any, Sequence

import numpy as np

from .embedding import (
    DEFAULT_SIGLIP2_MODEL,
    EmbeddingConfigurationError,
    Siglip2Embedder,
    TextImageEmbedder,
    is_immutable_hf_commit,
)
from .providers import FrameProvider, YouCook2FrameProvider
from .refinement_runtime import BoundaryRefinementRuntime
from .service import AdaptiveSearchService


@dataclass(frozen=True)
class RuntimeConfiguration:
    frame_provider: FrameProvider | None
    embedder: TextImageEmbedder | None
    provider_configured: bool
    model_configured: bool
    reason: str | None = None


class LazySiglip2Embedder:
    """Identity-first SigLIP2 adapter that loads weights on first inference."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        output_dimension: int = 768,
        device: str | None = None,
        torch_dtype: str = "float16",
        preprocess_version: str = "siglip2-auto-processor-v1",
    ) -> None:
        if not is_immutable_hf_commit(revision):
            raise ValueError(
                "ADAPTIVE_SIGLIP2_REVISION must be a full 40-character "
                "hexadecimal commit"
            )
        if output_dimension <= 0:
            raise ValueError("ADAPTIVE_SIGLIP2_DIMENSION must be positive")
        self.model_id = model_id
        self.revision = revision
        self.output_dimension = output_dimension
        self.preprocess_version = preprocess_version
        self.instruction = ""
        self._device = device
        self._torch_dtype = torch_dtype
        self._delegate: Siglip2Embedder | None = None
        self._lock = RLock()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._get_delegate().encode_texts(texts)

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        return self._get_delegate().encode_images(images)

    def _get_delegate(self) -> Siglip2Embedder:
        with self._lock:
            if self._delegate is None:
                try:
                    delegate = Siglip2Embedder(
                        model_id=self.model_id,
                        revision=self.revision,
                        device=self._device,
                        torch_dtype=self._torch_dtype,
                        preprocess_version=self.preprocess_version,
                    )
                except EmbeddingConfigurationError:
                    raise
                except Exception as exc:
                    raise EmbeddingConfigurationError(
                        "failed to initialize the pinned SigLIP2 runtime "
                        f"({type(exc).__name__})"
                    ) from exc
                if delegate.output_dimension != self.output_dimension:
                    raise RuntimeError(
                        "configured SigLIP2 dimension does not match loaded model: "
                        f"{self.output_dimension} != {delegate.output_dimension}"
                    )
                self._delegate = delegate
            return self._delegate


def runtime_from_environment() -> RuntimeConfiguration:
    data_root = os.getenv("YOUCOOK2_DATA_ROOT")
    metadata_root = os.getenv("YOUCOOK2_METADATA_ROOT")
    provider: FrameProvider | None = None
    reasons: list[str] = []
    if data_root:
        provider = YouCook2FrameProvider(
            data_root,
            metadata_root=metadata_root,
        )
        capability = provider.capabilities()
        if not capability.available:
            reasons.append(capability.reason or "YouCook2 provider is unavailable")

    revision = os.getenv("ADAPTIVE_SIGLIP2_REVISION")
    embedder: TextImageEmbedder | None = None
    if revision == "main":
        reasons.append("ADAPTIVE_SIGLIP2_REVISION must not be main")
    elif revision and not is_immutable_hf_commit(revision):
        reasons.append(
            "ADAPTIVE_SIGLIP2_REVISION must be a full 40-character "
            "hexadecimal commit"
        )
    elif revision:
        missing = [
            package
            for package in ("torch", "transformers")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            reasons.append(
                "missing live model dependencies: " + ", ".join(missing)
            )
        else:
            raw_dimension = os.getenv("ADAPTIVE_SIGLIP2_DIMENSION", "768")
            try:
                dimension = int(raw_dimension)
            except ValueError:
                reasons.append("ADAPTIVE_SIGLIP2_DIMENSION must be an integer")
            else:
                try:
                    embedder = LazySiglip2Embedder(
                        model_id=os.getenv(
                            "ADAPTIVE_SIGLIP2_MODEL", DEFAULT_SIGLIP2_MODEL
                        ),
                        revision=revision,
                        output_dimension=dimension,
                        device=os.getenv("ADAPTIVE_DEVICE") or None,
                        torch_dtype=os.getenv("ADAPTIVE_TORCH_DTYPE", "float16"),
                    )
                except ValueError as exc:
                    reasons.append(str(exc))

    return RuntimeConfiguration(
        frame_provider=provider,
        embedder=embedder,
        provider_configured=provider is not None,
        model_configured=embedder is not None,
        reason="; ".join(reasons) if reasons else None,
    )


def configure_from_environment(
    service: AdaptiveSearchService,
    refiner: BoundaryRefinementRuntime,
) -> RuntimeConfiguration:
    configuration = runtime_from_environment()
    if configuration.frame_provider is not None:
        service.frame_provider = configuration.frame_provider
        refiner.frame_provider = configuration.frame_provider
    refiner.embedder = configuration.embedder
    return configuration
