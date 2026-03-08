"""Canonical preprocessing contract v2 for Universal Recognizer Web."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


DEFAULT_CONTRACT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "contracts",
    "preprocess_contract_v2.json",
)


@dataclass
class PreprocessContractV2:
    data: Dict[str, Any]
    source_path: str

    @property
    def version(self) -> str:
        return str(self.data.get("version", "v2"))

    @property
    def checksum(self) -> str:
        payload = json.dumps(self.data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def transform_id(self) -> str:
        return str(self.data.get("transform_id", "identity"))

    @property
    def target_size(self) -> tuple[int, int]:
        resize = self.data.get("resize", {})
        return int(resize.get("height", 28)), int(resize.get("width", 28))

    @property
    def invert_threshold(self) -> float:
        polarity = self.data.get("polarity", {})
        return float(polarity.get("auto_invert_if_mean_above", 0.65))

    def get_stats(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, float]:
        norm = self.data.get("normalization", {})
        mean_raw = norm.get("mean")
        std_raw = norm.get("std")
        mean = np.asarray(mean_raw, dtype=np.float32) if isinstance(mean_raw, list) else None
        std = np.asarray(std_raw, dtype=np.float32) if isinstance(std_raw, list) else None
        clip_min = float(norm.get("clip_min", -5.0))
        clip_max = float(norm.get("clip_max", 5.0))
        eps = float(norm.get("epsilon", 1e-8))
        return mean, std, clip_min, clip_max, eps


def load_contract(path: Optional[str] = None) -> PreprocessContractV2:
    contract_path = path or os.getenv("UNIVERSAL_PREPROCESS_CONTRACT", DEFAULT_CONTRACT_PATH)
    with open(contract_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PreprocessContractV2(data=data, source_path=contract_path)


def save_contract(contract: PreprocessContractV2, path: Optional[str] = None) -> str:
    output_path = path or contract.source_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contract.data, f, indent=2)
    return output_path


def with_stats(contract: PreprocessContractV2, mean: np.ndarray, std: np.ndarray) -> PreprocessContractV2:
    updated = copy.deepcopy(contract.data)
    updated.setdefault("normalization", {})["mean"] = mean.astype(np.float32).tolist()
    updated.setdefault("normalization", {})["std"] = std.astype(np.float32).tolist()
    return PreprocessContractV2(updated, contract.source_path)
