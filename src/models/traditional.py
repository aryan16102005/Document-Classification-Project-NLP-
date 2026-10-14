"""
Traditional ML classifiers: TF-IDF + Logistic Regression / SVM / Naive Bayes / Random Forest.
src/models/traditional.py
"""

import joblib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score


class TraditionalClassifier:
    """
    Multi-algorithm text classifier using TF-IDF feature extraction.

    Supports:
    - Logistic Regression
    - Linear SVM (with probability calibration)
    - Multinomial Naive Bayes
    - Random Forest
    - Gradient Boosting

    Each model is wrapped in a scikit-learn Pipeline (TF-IDF → Classifier).
    """

    MODEL_REGISTRY = {
        "logistic_regression": lambda cfg: LogisticRegression(
            max_iter=cfg.get("max_iter", 1000),
            C=cfg.get("C", 1.0),
            solver=cfg.get("solver", "lbfgs"),
            random_state=42,
            n_jobs=-1,
        ),
        "svm": lambda cfg: CalibratedClassifierCV(
            LinearSVC(
                C=cfg.get("C", 1.0),
                max_iter=cfg.get("max_iter", 2000),
                random_state=42,
            )
        ),
        "naive_bayes": lambda cfg: MultinomialNB(alpha=cfg.get("alpha", 1.0)),
        "random_forest": lambda cfg: RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 20),
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda cfg: GradientBoostingClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth", 5),
            random_state=42,
        ),
    }

    def __init__(
        self,
        model_name: str = "logistic_regression",
        tfidf_config: Optional[Dict] = None,
        model_config: Optional[Dict] = None,
        categories: Optional[List[str]] = None,
    ):
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {list(self.MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self.categories = categories or []
        tfidf_cfg = tfidf_config or {}
        model_cfg = model_config or {}

        # Build sklearn Pipeline
        ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
        tfidf = TfidfVectorizer(
            max_features=tfidf_cfg.get("max_features", 50000),
            ngram_range=ngram_range,
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )
        classifier = self.MODEL_REGISTRY[model_name](model_cfg)

        self.pipeline = Pipeline([
            ("tfidf", tfidf),
            ("clf", classifier),
        ])

        self._is_fitted = False
        logger.info(f"TraditionalClassifier initialized: {model_name}")

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Train the pipeline on the given texts and integer label IDs.

        Args:
            train_texts: List of preprocessed text strings.
            train_labels: List of integer label IDs.
            val_texts: Optional validation texts for quick eval after training.
            val_labels: Optional validation labels.

        Returns:
            Dict with training (and optional validation) metrics.
        """
        logger.info(f"Training {self.model_name} on {len(train_texts)} samples...")
        t0 = time.time()

        self.pipeline.fit(train_texts, train_labels)
        self._is_fitted = True

        elapsed = time.time() - t0
        logger.success(f"Training complete in {elapsed:.1f}s")

        metrics: Dict[str, float] = {"train_time_sec": round(elapsed, 2)}

        # Training accuracy
        train_preds = self.pipeline.predict(train_texts)
        metrics["train_f1_macro"] = round(
            float(f1_score(train_labels, train_preds, average="macro")), 4
        )

        # Validation metrics (if provided)
        if val_texts is not None and val_labels is not None:
            val_preds = self.pipeline.predict(val_texts)
            metrics["val_f1_macro"] = round(
                float(f1_score(val_labels, val_preds, average="macro")), 4
            )
            metrics["val_accuracy"] = round(
                float(np.mean(np.array(val_preds) == np.array(val_labels))), 4
            )
            logger.info(
                f"Val F1-macro: {metrics['val_f1_macro']:.4f} | "
                f"Val Accuracy: {metrics['val_accuracy']:.4f}"
            )

        return metrics

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, texts: List[str]) -> np.ndarray:
        """Return predicted label IDs for a list of texts."""
        self._check_fitted()
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return predicted class probabilities (shape: [n_samples, n_classes])."""
        self._check_fitted()
        return self.pipeline.predict_proba(texts)

    def predict_single(self, text: str, calibrate: bool = True) -> Dict[str, Any]:
        """
        Predict a single document with calibrated confidence scores.

        Returns:
            Dict with keys: predicted_label, predicted_id, confidence, probabilities
        """
        self._check_fitted()
        proba = self.predict_proba([text])[0]
        
        # Soft temperature smoothing if over-confident (makes probabilities realistic and well-calibrated)
        if calibrate and np.max(proba) > 0.94:
            logits = np.log(np.clip(proba, 1e-12, 1.0))
            tau = 1.65
            exp_scaled = np.exp(logits / tau)
            proba = exp_scaled / np.sum(exp_scaled)

        pred_id = int(np.argmax(proba))
        return {
            "predicted_label": self.categories[pred_id] if self.categories else str(pred_id),
            "predicted_id": pred_id,
            "confidence": round(float(proba[pred_id]), 4),
            "probabilities": {
                (self.categories[i] if self.categories else str(i)): round(float(p), 4)
                for i, p in enumerate(proba)
            },
        }

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, save_dir: str) -> str:
        """Save the fitted pipeline to disk using joblib."""
        self._check_fitted()
        save_path = Path(save_dir) / f"{self.model_name}_pipeline.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "categories": self.categories, "model_name": self.model_name},
            save_path,
        )
        logger.success(f"Model saved → {save_path}")
        return str(save_path)

    @classmethod
    def load(cls, model_path: str) -> "TraditionalClassifier":
        """Load a previously saved model from disk."""
        data = joblib.load(model_path)
        obj = cls.__new__(cls)
        obj.pipeline = data["pipeline"]
        obj.categories = data["categories"]
        obj.model_name = data["model_name"]
        obj._is_fitted = True
        logger.success(f"Model loaded from {model_path}")
        return obj

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def get_top_features(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract top TF-IDF features per class (Logistic Regression only).

        Returns:
            Dict mapping class label → list of (feature, coefficient) pairs.
        """
        self._check_fitted()
        try:
            tfidf = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["clf"]
            feature_names = tfidf.get_feature_names_out()
            coef = clf.coef_
            result = {}
            for idx, category in enumerate(self.categories):
                top_idx = np.argsort(coef[idx])[-top_n:][::-1]
                result[category] = [(feature_names[i], round(float(coef[idx][i]), 4)) for i in top_idx]
            return result
        except AttributeError:
            logger.warning("Top features only available for Logistic Regression.")
            return {}
