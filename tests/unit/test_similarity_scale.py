"""Unit tests for the shared episodic similarity scale (client/memory/similarity.py).

One config knob (EPISODIC_MEMORY.RETRIEVAL_THRESHOLD) gates both backends, so
FAISS inner products and Chroma squared-L2 distances must land on the SAME
cosine-in-[0,1] scale. These pin the conversions and the fingerprint bump that
migrates stores written under the old, incomparable scoring.
"""

import numpy as np
import pytest

from mnemoai.client.memory.similarity import (
    SCORING_VERSION,
    cosine_to_unit,
    l2_normalize,
    squared_l2_to_unit,
)


class TestL2Normalize:
    def test_unit_norm_rows(self):
        out = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])

    def test_already_normalized_is_stable(self):
        vec = np.array([[0.6, 0.8]], dtype=np.float32)
        assert np.allclose(l2_normalize(vec), vec)

    def test_zero_vector_does_not_produce_nan(self):
        out = l2_normalize(np.array([[0.0, 0.0]]))
        assert not np.isnan(out).any()

    def test_1d_input_supported(self):
        out = l2_normalize(np.array([3.0, 4.0]))
        assert out.shape == (2,)
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_output_is_float32_for_faiss(self):
        assert l2_normalize(np.array([[1.0, 2.0]], dtype=np.float64)).dtype == np.float32


class TestScaleConversions:
    def test_identical_vectors_score_one(self):
        assert cosine_to_unit(1.0) == pytest.approx(1.0)
        assert squared_l2_to_unit(0.0) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_half(self):
        # cos = 0 -> 0.5; unit vectors at 90 deg have squared-L2 distance 2.
        assert cosine_to_unit(0.0) == pytest.approx(0.5)
        assert squared_l2_to_unit(2.0) == pytest.approx(0.5)

    def test_opposite_vectors_score_zero(self):
        assert cosine_to_unit(-1.0) == pytest.approx(0.0)
        assert squared_l2_to_unit(4.0) == pytest.approx(0.0)

    def test_both_backends_agree_on_the_same_pair(self):
        a = l2_normalize(np.array([1.0, 2.0, 3.0]))
        b = l2_normalize(np.array([2.0, 1.0, 3.0]))
        cosine = float(np.dot(a, b))
        squared_distance = float(np.sum((a - b) ** 2))
        assert cosine_to_unit(cosine) == pytest.approx(
            squared_l2_to_unit(squared_distance), abs=1e-6
        )

    def test_scores_are_clamped(self):
        # Float drift or a legacy non-unit vector must not escape [0,1].
        assert cosine_to_unit(1.2) == 1.0
        assert cosine_to_unit(-1.5) == 0.0
        assert squared_l2_to_unit(9.0) == 0.0
        assert squared_l2_to_unit(-0.1) == 1.0


class TestFingerprintMigration:
    def test_scoring_version_is_in_the_store_fingerprints(self, monkeypatch):
        # Both stores must stamp the scoring version, so a store written with the
        # old raw-inner-product / 1/(1+d) scoring is reset once rather than
        # mixing two scales under one threshold.
        from mnemoai.client.memory.faiss_store import FAISSEpisodicStore

        embeddings = type("E", (), {"fingerprint": lambda self: "model-x@1024"})()
        fingerprint = FAISSEpisodicStore._embed_fingerprint(
            type("S", (), {"embeddings": embeddings})()
        )
        assert fingerprint == f"model-x@1024|score={SCORING_VERSION}"

    def test_fingerprint_falls_back_to_dimension(self):
        from mnemoai.client.memory.chroma_store import ChromaEpisodicStore

        embeddings = type("E", (), {"dim": 768})()
        fingerprint = ChromaEpisodicStore._embed_fingerprint(
            type("S", (), {"embeddings": embeddings})()
        )
        assert fingerprint == f"dim=768|score={SCORING_VERSION}"
