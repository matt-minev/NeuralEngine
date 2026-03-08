import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.universal_recognizer_web.core.preprocess_contract import load_contract
from apps.universal_recognizer_web.core.canonical_preprocessor import CanonicalPreprocessorV2, apply_transform, TRANSFORMS


def _payload(offset_x=0, offset_y=0):
    points = []
    for i in range(80):
        points.append({'x': 130 + offset_x + (i % 3), 'y': 70 + offset_y + i * 2, 't': i})
    return {
        'canvas': {'width': 280, 'height': 280},
        'strokes': [{'points': points}],
    }


def test_contract_load_and_checksum():
    c = load_contract()
    assert c.version in {'v2', 'v3'}
    assert len(c.checksum) == 64


def test_transform_registry_and_shape():
    x = np.zeros((28, 28), dtype=np.float32)
    x[10:18, 12:16] = 1.0
    for t in TRANSFORMS:
        y = apply_transform(x, t)
        assert y.shape == (28, 28)


def test_contract_preprocess_deterministic_output():
    pp = CanonicalPreprocessorV2(load_contract())
    out1, _, _ = pp.preprocess(_payload())
    out2, _, _ = pp.preprocess(_payload())
    np.testing.assert_allclose(out1, out2, atol=1e-7)
    assert out1.shape == (1, 784)


def test_translation_invariance_near_center():
    pp = CanonicalPreprocessorV2(load_contract())
    a, _, _ = pp.preprocess(_payload(0, 0))
    b, _, _ = pp.preprocess(_payload(8, 10))
    assert np.mean(np.abs(a - b)) < 0.12
