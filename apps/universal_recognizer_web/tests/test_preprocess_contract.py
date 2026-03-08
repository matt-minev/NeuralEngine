import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.universal_recognizer_web.core.preprocess_contract import load_contract
from apps.universal_recognizer_web.core.canonical_preprocessor import CanonicalPreprocessorV2, apply_transform, TRANSFORMS


def test_contract_load_and_checksum():
    c = load_contract()
    assert c.version == 'v2'
    assert len(c.checksum) == 64


def test_transform_registry_and_shape():
    x = np.zeros((28, 28), dtype=np.float32)
    x[10:18, 12:16] = 1.0
    for t in TRANSFORMS:
        y = apply_transform(x, t)
        assert y.shape == (28, 28)


def test_contract_preprocess_deterministic_output():
    pp = CanonicalPreprocessorV2(load_contract())
    x = np.zeros((280, 280), dtype=np.float32)
    x[60:220, 120:160] = 1.0
    out1, _, _ = pp.preprocess(x)
    out2, _, _ = pp.preprocess(x)
    np.testing.assert_allclose(out1, out2, atol=1e-7)
    assert out1.shape == (1, 784)
