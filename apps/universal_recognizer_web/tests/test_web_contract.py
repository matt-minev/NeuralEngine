import base64
import io
import numpy as np
import os
import sys
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.universal_recognizer_web.app import app
from apps.web_tests.harness import FlaskHarness
from apps.universal_recognizer_web.core.canonical_preprocessor import CanonicalPreprocessorV2


def _canvas_data_url():
    img = np.zeros((280, 280), dtype=np.uint8)
    img[80:220, 130:150] = 255
    pil = Image.fromarray(img, mode='L')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')


def _strict_payload(offset_x=0, offset_y=0):
    points = []
    for i in range(60):
        points.append({'x': 140 + offset_x, 'y': 90 + i * 2 + offset_y, 't': i})
    return {
        'input': {
            'canvas': {'width': 280, 'height': 280},
            'strokes': [{'points': points}],
            'raster': _canvas_data_url(),
        }
    }


def test_preprocess_shape_and_range():
    pp = CanonicalPreprocessorV2()
    out, metrics, debug = pp.preprocess(_strict_payload()['input'], return_metrics=True, return_debug=True)
    assert out.shape == (1, 784)
    assert np.isfinite(out).all()
    assert 'contract_version' in debug
    assert metrics is not None


def test_predict_schema_v3():
    h = FlaskHarness(app)
    resp = h.client.post('/predict', json=_strict_payload())
    data = h.assert_json_ok(resp)
    for key in ['predicted_character', 'predicted_index', 'confidence', 'calibrated_confidence', 'top_k', 'model_version', 'contract_version', 'contract_checksum']:
        assert key in data


def test_accessibility_is_advisory_only():
    h = FlaskHarness(app)
    resp = h.client.post('/predict/accessibility', json=_strict_payload())
    data = h.assert_json_ok(resp)
    assert 'prediction' in data
    assert 'advisory' in data
    assert 'mirror_candidate' in data['advisory']
    assert 'confusion_candidates' in data['advisory']


def test_debug_preprocess_endpoint():
    h = FlaskHarness(app)
    resp = h.client.post('/api/debug/preprocess', json=_strict_payload())
    data = h.assert_json_ok(resp)
    assert 'contract_version' in data
    assert 'contract_checksum' in data
    assert 'transform_id' in data


def test_strict_mode_rejects_raster_only():
    h = FlaskHarness(app)
    resp = h.client.post('/predict', json={'image': _canvas_data_url()})
    assert resp.status_code >= 400
