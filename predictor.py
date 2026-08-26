from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).parent

MODEL = joblib.load(ROOT / "ctmt_svc_deployment.joblib")
FEATURE_COLUMNS = json.loads(
    (ROOT / "model_feature_columns.json").read_text(encoding="utf-8")
)

def predict_research_score(features: Dict[str, float]) -> Dict[str, Any]:
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing model features: {missing[:10]}")

    row = pd.DataFrame(
        [[float(features[c]) for c in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    proba = float(MODEL.predict_proba(row)[0, 1])
    pred = int(MODEL.predict(row)[0])

    # Do not translate this into a clinical diagnosis.
    return {
        "research_probability_mci_pattern": proba,
        "model_class_at_default_threshold": pred,
        "default_threshold": 0.5,
        "interpretation": (
            "연구용 디지털 행동 패턴 점수입니다. "
            "MCI 진단 또는 임상적 확률로 해석하면 안 됩니다."
        ),
    }
