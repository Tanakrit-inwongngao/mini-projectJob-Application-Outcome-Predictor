from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    model: Any
    feature_names: Optional[List[str]]
    classes: Optional[List[Any]]


class Predictor:
    """
    โครงสร้างที่รองรับ:
    1) default_model_path = app/best_model_job_app.joblib
    2) artifacts_dir:
        app/model_artifacts/<model_key>/model.joblib
        app/model_artifacts/<model_key>/metrics.json (optional)  # (ตอนนี้ UI ไม่ใช้แล้ว แต่เก็บโครงสร้างไว้ได้)
    """

    def __init__(self, default_model_path: str, artifacts_dir: str = "model_artifacts"):
        self.default_model_path = default_model_path
        self.artifacts_dir = artifacts_dir

        # model_key -> (model_path, metrics_path)
        self.registry: Dict[str, Tuple[str, Optional[str]]] = {}
        self.cache: Dict[str, ModelBundle] = {}

    def discover_models(self) -> None:
        self.registry = {}

        # ---- default model ----
        if os.path.isfile(self.default_model_path):
            default_dir = os.path.join(os.path.dirname(self.default_model_path), "model_artifacts", "best_model_job_app")
            default_metrics = os.path.join(default_dir, "metrics.json")
            default_metrics = default_metrics if os.path.isfile(default_metrics) else None
            self.registry["best_model_job_app"] = (self.default_model_path, default_metrics)

        # ---- artifacts models ----
        if os.path.isdir(self.artifacts_dir):
            for name in os.listdir(self.artifacts_dir):
                folder = os.path.join(self.artifacts_dir, name)
                if not os.path.isdir(folder):
                    continue

                model_path = os.path.join(folder, "model.joblib")
                metrics_path = os.path.join(folder, "metrics.json")

                if os.path.isfile(model_path):
                    self.registry[name] = (model_path, metrics_path if os.path.isfile(metrics_path) else None)

        if not self.registry:
            raise FileNotFoundError(
                f"ไม่พบโมเดล: {self.default_model_path} หรือ {self.artifacts_dir}/<name>/model.joblib\n"
                f"ตรวจสอบว่าไฟล์โมเดลอยู่จริงไหม:\n"
                f"- {self.default_model_path}\n"
                f"- {self.artifacts_dir}\\<model_key>\\model.joblib"
            )

    def list_models(self) -> List[str]:
        return sorted(self.registry.keys())

    def load(self) -> None:
        self.discover_models()
        first = self.list_models()[0]
        self._get_bundle(first)

    def _get_bundle(self, model_key: str) -> ModelBundle:
        if model_key in self.cache:
            return self.cache[model_key]
        if model_key not in self.registry:
            raise ValueError(f"Unknown model: {model_key}")

        model_path, _ = self.registry[model_key]
        obj = joblib.load(model_path)

        feature_names = None
        classes = None

        if hasattr(obj, "feature_names_in_"):
            feature_names = list(getattr(obj, "feature_names_in_"))
        if hasattr(obj, "classes_"):
            classes = list(getattr(obj, "classes_"))

        bundle = ModelBundle(model=obj, feature_names=feature_names, classes=classes)
        self.cache[model_key] = bundle
        return bundle

    def _to_float(self, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _auto_fill_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        years = self._to_float(features.get("years_experience"))
        salary = self._to_float(features.get("expected_salary"))
        tech = self._to_float(features.get("tech_skill"))
        soft = self._to_float(features.get("soft_skill"))
        port = self._to_float(features.get("portfolio_score"))

        def is_missing(v: Any) -> bool:
            return v is None or (isinstance(v, str) and v.strip() == "")

        if is_missing(features.get("salary_per_exp")):
            if years is not None and years > 0 and salary is not None:
                features["salary_per_exp"] = round(salary / years, 4)
            else:
                features["salary_per_exp"] = None

        if is_missing(features.get("skill_total")):
            if tech is not None and soft is not None and port is not None:
                features["skill_total"] = round(tech + soft + port, 4)
            else:
                features["skill_total"] = None

        if is_missing(features.get("eng_x_interview")):
            eng = self._to_float(features.get("english_score"))
            interview = self._to_float(features.get("interview_score"))
            if eng is not None and interview is not None:
                features["eng_x_interview"] = round(eng * interview, 4)
            else:
                features["eng_x_interview"] = None

        return features

    def _prepare_dataframe(self, bundle: ModelBundle, input_data: Dict[str, Any]) -> pd.DataFrame:
        data = dict(input_data or {})
        data = self._auto_fill_derived_features(data)
        df = pd.DataFrame([data])

        if bundle.feature_names:
            for col in bundle.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[bundle.feature_names]

        return df

    def predict_one(self, model_key: str, input_data: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """คืนผลให้เข้ากับหน้า index.html: predicted_class + probabilities"""
        bundle = self._get_bundle(model_key)
        df = self._prepare_dataframe(bundle, input_data)
        model = bundle.model

        pred = model.predict(df)[0]
        predicted_class = str(pred)

        probabilities: List[Dict[str, Any]] = []
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            classes = list(getattr(model, "classes_", []))

            pairs = [(str(c), float(p)) for c, p in zip(classes, proba)]
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[: max(1, int(top_k))]

            probabilities = [{"label": lbl, "probability": prob} for lbl, prob in pairs]

        used_features = df.iloc[0].to_dict()
        for k, v in list(used_features.items()):
            if isinstance(v, np.generic):
                used_features[k] = float(v)

        return {
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "used_features": used_features,
        }
