"""One similarity scale for both episodic backends.

A single config knob -- ``EPISODIC_MEMORY.RETRIEVAL_THRESHOLD`` -- gates recall
for whichever store is configured, so the two backends MUST produce comparable
numbers. They natively do not:

* FAISS ``IndexFlatIP`` returns a raw inner product (unbounded, and only equal
  to cosine when the vectors happen to be unit-norm),
* ChromaDB returns a squared-L2 distance, previously mapped with ``1/(1+d)``.

Both are converted here to **cosine similarity rescaled to [0, 1]** (0.5 means
orthogonal), which makes a threshold like 0.7 mean the same thing on either
backend. Vectors are L2-normalized on write and on query so the conversions
hold.

``SCORING_VERSION`` is folded into each store's embedding fingerprint, so a
store written under the old scoring is reset once through the existing
fingerprint-migration path instead of silently mixing two scales.
"""

import numpy as np

SCORING_VERSION = "cos01"


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, leaving zero rows untouched.

    Args:
        vectors: A 1-D vector or 2-D array of row vectors.

    Returns:
        A float32 array of the same shape with unit-norm rows.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        return array if norm == 0.0 else (array / norm).astype(np.float32)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # Zero-norm rows would divide by zero; leave them as-is (they score 0.5,
    # i.e. "no signal", rather than producing NaN and poisoning the ranking).
    norms[norms == 0.0] = 1.0
    return (array / norms).astype(np.float32)


def cosine_to_unit(cosine: float) -> float:
    """Map a cosine similarity in [-1, 1] to [0, 1]."""
    return _clamp01((float(cosine) + 1.0) / 2.0)


def squared_l2_to_unit(distance: float) -> float:
    """Map a squared-L2 distance between unit vectors to [0, 1].

    For unit vectors ``d = 2 - 2*cos``, so ``cos = 1 - d/2`` and the rescaled
    score is ``1 - d/4``.
    """
    return _clamp01(1.0 - (float(distance) / 4.0))


def _clamp01(value: float) -> float:
    """Guard against float drift (and non-unit legacy vectors) leaving [0, 1]."""
    return max(0.0, min(1.0, value))
