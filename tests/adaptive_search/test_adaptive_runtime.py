import os
import unittest
from unittest.mock import patch

from adaptive_search.refinement_runtime import BoundaryRefinementRuntime
from adaptive_search.schemas import RefinementHyperparameters
from adaptive_search.config import (
    LazySiglip2Embedder,
    configure_from_environment,
    runtime_from_environment,
)
from adaptive_search.service import AdaptiveSearchService


class AdaptiveRuntimeTests(unittest.TestCase):
    def test_default_import_does_not_configure_or_load_optional_runtime(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = runtime_from_environment()
        self.assertFalse(runtime.provider_configured)
        self.assertFalse(runtime.model_configured)
        self.assertIsNone(runtime.frame_provider)
        self.assertIsNone(runtime.embedder)

    def test_unpinned_revision_is_reported_without_crashing_api_startup(self):
        values = {"ADAPTIVE_SIGLIP2_REVISION": "main"}
        with patch.dict(os.environ, values, clear=True):
            runtime = runtime_from_environment()
        self.assertFalse(runtime.model_configured)
        self.assertIn("must not be main", runtime.reason)

    def test_missing_model_dependencies_are_reported_before_weight_loading(self):
        values = {"ADAPTIVE_SIGLIP2_REVISION": "a" * 40}
        with (
            patch.dict(os.environ, values, clear=True),
            patch("adaptive_search.config.importlib.util.find_spec", return_value=None),
        ):
            runtime = runtime_from_environment()
        self.assertFalse(runtime.model_configured)
        self.assertIn("torch", runtime.reason)
        self.assertIn("transformers", runtime.reason)

    def test_named_branch_is_not_accepted_as_an_immutable_revision(self):
        values = {"ADAPTIVE_SIGLIP2_REVISION": "release-v1"}
        with patch.dict(os.environ, values, clear=True):
            runtime = runtime_from_environment()
        self.assertFalse(runtime.model_configured)
        self.assertIn("40-character", runtime.reason)

    def test_configuration_leaves_default_provider_when_root_is_absent(self):
        service = AdaptiveSearchService()
        original = service.frame_provider
        refiner = BoundaryRefinementRuntime()
        with patch.dict(os.environ, {}, clear=True):
            runtime = configure_from_environment(service, refiner)
        self.assertIs(service.frame_provider, original)
        self.assertIsNone(refiner.embedder)
        self.assertFalse(runtime.provider_configured)

    def test_default_session_preprocess_matches_lazy_runtime(self):
        session_spec = RefinementHyperparameters().embedding_model
        runtime = LazySiglip2Embedder(
            model_id=session_spec.model_id,
            revision="a" * 40,
        )
        self.assertEqual(session_spec.model_id, runtime.model_id)
        self.assertEqual(session_spec.dimension, runtime.output_dimension)
        self.assertEqual(session_spec.preprocess, runtime.preprocess_version)


if __name__ == "__main__":
    unittest.main()
