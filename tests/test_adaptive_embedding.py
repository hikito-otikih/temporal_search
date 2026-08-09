import unittest
from types import SimpleNamespace

import numpy as np

from adaptive_search.embedding import (
    EmbeddingConfigurationError,
    Qwen3VLEmbedderAdapter,
    Siglip2Embedder,
    cosine_scores,
    image_embedding_cache_key,
    score_event_frames,
    truncate_mrl,
)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeProcessorBatch(dict):
    def to(self, device):
        return self


class FakeSiglipProcessor:
    def __call__(self, **kwargs):
        return FakeProcessorBatch()


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeTorch:
    @staticmethod
    def inference_mode():
        return FakeInferenceMode()


class FakeSiglipModel:
    def __init__(self):
        self.text_return_dict = None
        self.image_return_dict = None

    def get_text_features(self, **kwargs):
        self.text_return_dict = kwargs.get("return_dict")
        return SimpleNamespace(
            last_hidden_state=FakeTensor(np.zeros((2, 4, 2))),
            pooler_output=FakeTensor([[3.0, 4.0], [0.0, 2.0]]),
        )

    def get_image_features(self, **kwargs):
        self.image_return_dict = kwargs.get("return_dict")
        return FakeTensor([[5.0, 0.0]])


class FakeEmbedder:
    model_id = "fake"
    revision = "abc123"
    output_dimension = 3
    preprocess_version = "v1"
    instruction = ""

    def encode_texts(self, texts):
        self.texts = list(texts)
        return np.eye(3, dtype=np.float32)

    def encode_images(self, images):
        self.images = list(images)
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


class FakeQwenBackend:
    def __init__(self):
        self.normalize_values = []

    def process(self, inputs, normalize=True):
        self.normalize_values.append(normalize)
        rows = []
        for index, _ in enumerate(inputs):
            rows.append([index + 1.0] + [2.0] * 127)
        return np.asarray(rows, dtype=np.float32)


class BatchTrackingEmbedder:
    def __init__(self):
        self.batch_sizes = []

    def encode_texts(self, texts):
        return np.eye(3, dtype=np.float32)

    def encode_images(self, images):
        self.batch_sizes.append(len(images))
        return np.tile(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            (len(images), 1),
        )


class AdaptiveEmbeddingTests(unittest.TestCase):
    def test_siglip2_accepts_v5_model_output_and_v4_tensor(self):
        embedder = object.__new__(Siglip2Embedder)
        embedder.output_dimension = 2
        embedder._processor = FakeSiglipProcessor()
        embedder._model = FakeSiglipModel()
        embedder._torch = FakeTorch()
        embedder._device = "cpu"

        text = embedder.encode_texts(["first", "second"])
        images = embedder.encode_images(["frame"])

        self.assertTrue(embedder._model.text_return_dict)
        self.assertTrue(embedder._model.image_return_dict)
        np.testing.assert_allclose(text, [[0.6, 0.8], [0.0, 1.0]], atol=1e-7)
        np.testing.assert_allclose(images, [[1.0, 0.0]], atol=1e-7)

    def test_siglip2_rejects_output_without_pooled_features(self):
        embedder = object.__new__(Siglip2Embedder)
        embedder.output_dimension = 2
        embedder._processor = FakeSiglipProcessor()
        embedder._model = FakeSiglipModel()
        embedder._model.get_text_features = lambda **kwargs: SimpleNamespace(
            pooler_output=None
        )
        embedder._torch = FakeTorch()
        embedder._device = "cpu"

        with self.assertRaisesRegex(
            EmbeddingConfigurationError, "no pooled feature tensor"
        ):
            embedder.encode_texts(["query"])

    def test_score_event_frames_keeps_three_raw_curves_separate(self):
        embedder = FakeEmbedder()

        curves = score_event_frames(
            embedder,
            anchor_query="anchor",
            pre_state_query="pre",
            post_state_query="post",
            images=["f0", "f1", "f2"],
        )

        self.assertEqual(curves.anchor, [1.0, 0.0, 0.0])
        self.assertEqual(curves.pre, [0.0, 1.0, 0.0])
        self.assertEqual(curves.post, [0.0, 0.0, 1.0])
        self.assertEqual(embedder.texts, ["anchor", "pre", "post"])

    def test_cosine_scores_defensively_normalizes(self):
        text = np.asarray([[2.0, 0.0]], dtype=np.float32)
        images = np.asarray([[5.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        scores = cosine_scores(text, images)
        np.testing.assert_allclose(scores, [[1.0, 0.0]], atol=1e-7)

    def test_frame_scoring_microbatches_image_encoding(self):
        embedder = BatchTrackingEmbedder()
        curves = score_event_frames(
            embedder,
            anchor_query="anchor",
            pre_state_query="pre",
            post_state_query="post",
            images=list(range(5)),
            image_batch_size=2,
        )
        self.assertEqual(embedder.batch_sizes, [2, 2, 1])
        self.assertEqual(len(curves.anchor), 5)

    def test_mrl_prefix_is_renormalized_after_truncation(self):
        vectors = np.asarray([[3.0, 4.0, 100.0]], dtype=np.float32)
        truncated = truncate_mrl(vectors, 2)
        np.testing.assert_allclose(truncated, [[0.6, 0.8]], atol=1e-7)

    def test_qwen_adapter_requests_unnormalized_full_vectors(self):
        backend = FakeQwenBackend()
        adapter = Qwen3VLEmbedderAdapter(
            backend,
            revision="commit123",
            output_dimension=64,
        )

        result = adapter.encode_texts(["a", "b"])

        self.assertEqual(backend.normalize_values, [False])
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])

    def test_image_cache_key_uses_pts_and_model_identity(self):
        base = dict(
            video_content_hash="video-sha",
            pts_ms=1234,
            model_id="model",
            revision="commit",
            output_dimension=768,
            preprocess_version="processor-v1",
        )
        first = image_embedding_cache_key(**base)
        second = image_embedding_cache_key(**base)
        changed = image_embedding_cache_key(**{**base, "pts_ms": 1235})

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
