import importlib


def test_universal_import_smoke():
    m = importlib.import_module('apps.universal_recognizer_web.app')
    assert hasattr(m, 'app')


def test_digit_import_smoke():
    m = importlib.import_module('apps.digit_recognizer_web.app')
    assert hasattr(m, 'app')


def test_quadratic_import_smoke():
    m = importlib.import_module('apps.quadratic_web.app')
    assert hasattr(m, 'app')
