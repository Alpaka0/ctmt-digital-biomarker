from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import base64
import io
import json
import pandas as pd
import joblib

ROOT = Path(__file__).parent

# GitHub connector uploads text files only, so the small deployment model is
# stored as five base64 text parts and reconstructed in memory at startup.
_MODEL_PARTS = sorted((ROOT / "model_parts").glob("part*.txt"))
if not _MODEL_PARTS:
    raise FileNotFoundError("Deployment model parts were not found.")
_model_b64 = "".join(p.read_text(encoding="ascii").strip() for p in _MODEL_PARTS)
MODEL = joblib.load(io.BytesIO(base64.b64decode(_model_b64)))

FEATURE_COLUMNS = json.loads(
    (ROOT / "model_feature_columns.json").read_text(encoding="utf-8")
)


def predict_research_score(features: Dict[str, float]) -> Dict[str, Any]:
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing model features: {missing[:10]}")

    row = pd.DataFrame(
        [[float(features[c]) for c in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    proba = float(MODEL.predict_proba(row)[0, 1])
    pred = int(MODEL.predict(row)[0])

    return {
        "research_probability_mci_pattern": proba,
        "model_class_at_default_threshold": pred,
        "default_threshold": 0.5,
        "interpretation": (
            "연구용 디지털 행동 패턴 점수입니다. "
            "MCI 진단 또는 임상적 확률로 해석하면 안 됩니다."
        ),
    }
