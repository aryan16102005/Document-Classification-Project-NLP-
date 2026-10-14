"""
Model evaluation utilities: metrics, confusion matrix, comparison reports.
src/models/evaluator.py
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """
    Centralized evaluation utility for any classifier.
    Computes metrics, generates reports, and saves results to disk.
    """

    def __init__(self, categories: List[str]):
        self.categories = categories
        self.results: Dict[str, Dict] = {}

    def evaluate(
        self,
        model_name: str,
        y_true: List[int],
        y_pred: List[int],
        y_proba: Optional[np.ndarray] = None,
        inference_time_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute a full set of evaluation metrics for a model.

        Args:
            model_name: Identifier for this model (stored internally).
            y_true: Ground-truth label IDs.
            y_pred: Predicted label IDs.
            y_proba: Class probability matrix [n_samples, n_classes] (optional).
            inference_time_ms: Average inference time per sample in ms (optional).

        Returns:
            Dict of evaluation metrics.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = {
            "model": model_name,
            "n_samples": int(len(y_true)),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
            "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        }

        if inference_time_ms is not None:
            metrics["inference_ms_per_sample"] = round(inference_time_ms, 3)

        # Per-class metrics
        report = classification_report(
            y_true, y_pred,
            target_names=self.categories,
            output_dict=True,
            zero_division=0,
        )
        metrics["per_class"] = {
            cat: {
                "precision": round(report[cat]["precision"], 4),
                "recall": round(report[cat]["recall"], 4),
                "f1": round(report[cat]["f1-score"], 4),
                "support": int(report[cat]["support"]),
            }
            for cat in self.categories
            if cat in report
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # ROC AUC (if probabilities provided)
        if y_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                metrics["roc_auc_macro"] = round(float(auc), 4)
            except Exception:
                pass

        self.results[model_name] = metrics
        logger.info(
            f"[{model_name}] Accuracy={metrics['accuracy']:.4f} | "
            f"F1-macro={metrics['f1_macro']:.4f} | "
            f"F1-weighted={metrics['f1_weighted']:.4f}"
        )
        return metrics

    def timed_predict(self, model, texts: List[str]) -> tuple:
        """
        Run predict + predict_proba and measure per-sample latency.

        Returns:
            (predictions, probabilities, ms_per_sample)
        """
        t0 = time.time()
        preds = model.predict(texts)
        elapsed_ms = (time.time() - t0) * 1000
        ms_per_sample = elapsed_ms / max(len(texts), 1)

        try:
            proba = model.predict_proba(texts)
        except Exception:
            proba = None

        return preds, proba, ms_per_sample

    def compare(self) -> pd.DataFrame:
        """
        Return a summary DataFrame comparing all evaluated models.

        Returns:
            DataFrame sorted by F1-macro (descending).
        """
        if not self.results:
            raise RuntimeError("No models evaluated yet.")

        rows = []
        for name, metrics in self.results.items():
            row = {
                "model": name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
            }
            if "roc_auc_macro" in metrics:
                row["roc_auc_macro"] = metrics["roc_auc_macro"]
            if "inference_ms_per_sample" in metrics:
                row["inference_ms_per_sample"] = metrics["inference_ms_per_sample"]
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
        return df

    def print_comparison(self):
        """Print a formatted comparison table."""
        df = self.compare()
        logger.info("\n" + df.to_string(index=False))
        return df

    def save_report(self, output_dir: str, filename: str = "evaluation_report.json") -> str:
        """Save all evaluation results to a JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / filename
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.success(f"Evaluation report saved → {report_path}")
        return str(report_path)

    def get_confusion_matrix_df(self, model_name: str) -> pd.DataFrame:
        """Return the confusion matrix as a labeled DataFrame."""
        if model_name not in self.results:
            raise KeyError(f"Model '{model_name}' not in results.")
        cm = np.array(self.results[model_name]["confusion_matrix"])
        return pd.DataFrame(cm, index=self.categories, columns=self.categories)

    def classification_report_str(self, y_true: List[int], y_pred: List[int]) -> str:
        """Return a formatted classification report string."""
        return classification_report(
            y_true, y_pred,
            target_names=self.categories,
            zero_division=0,
        )
