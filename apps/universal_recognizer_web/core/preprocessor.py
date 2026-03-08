"""Compatibility facade for canonical preprocessing v2."""

from .canonical_preprocessor import (
    CanonicalPreprocessorV2 as AdvancedPreprocessor,
    preprocess_for_prediction,
    preprocess_with_metrics,
)

__all__ = [
    "AdvancedPreprocessor",
    "preprocess_for_prediction",
    "preprocess_with_metrics",
]
