"""
Tests for traditional ML classifier.
tests/test_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


CATEGORIES = ["technology", "sports", "politics", "entertainment"]
TRAIN_TEXTS = [
    "apple iphone release new technology chip",
    "google deepmind neural network artificial intelligence",
    "microsoft azure cloud computing software",
    "football match score goal player team",
    "basketball nba finals championship victory",
    "tennis grand slam wimbledon player serve",
    "government parliament bill vote policy law",
    "election president senator campaign vote",
    "senate congress legislation debate political",
    "movie film actor director award cinema",
    "music album concert singer celebrity tour",
    "streaming netflix series episode season show",
] * 10
TRAIN_LABELS = ([0] * 30 + [1] * 30 + [2] * 30 + [3] * 30)[:120]

TEST_TEXTS = [
    "new smartphone technology processor chip",
    "soccer football game score championship",
    "government policy vote senator election",
    "movie director award ceremony actor",
]
TEST_LABELS = [0, 1, 2, 3]


class TestTraditionalClassifier:

    @pytest.fixture
    def fitted_lr(self):
        from src.models.traditional import TraditionalClassifier
        clf = TraditionalClassifier(
            model_name="logistic_regression",
            categories=CATEGORIES,
            tfidf_config={"max_features": 1000, "ngram_range": [1, 1]},
        )
        clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
        return clf

    def test_predict_returns_array(self, fitted_lr):
        preds = fitted_lr.predict(TEST_TEXTS)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(TEST_TEXTS)

    def test_predict_proba_shape(self, fitted_lr):
        proba = fitted_lr.predict_proba(TEST_TEXTS)
        assert proba.shape == (len(TEST_TEXTS), len(CATEGORIES))
        # Probabilities sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_single_keys(self, fitted_lr):
        result = fitted_lr.predict_single("technology AI machine learning")
        assert "predicted_label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["predicted_label"] in CATEGORIES
        assert 0.0 <= result["confidence"] <= 1.0

    def test_save_and_load(self, fitted_lr, tmp_path):
        model_path = fitted_lr.save(save_dir=str(tmp_path))
        from src.models.traditional import TraditionalClassifier
        loaded = TraditionalClassifier.load(model_path)
        preds_original = fitted_lr.predict(TEST_TEXTS)
        preds_loaded = loaded.predict(TEST_TEXTS)
        np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_unfitted_raises(self):
        from src.models.traditional import TraditionalClassifier
        clf = TraditionalClassifier(model_name="logistic_regression", categories=CATEGORIES)
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(["test text"])

    def test_invalid_model_name(self):
        from src.models.traditional import TraditionalClassifier
        with pytest.raises(ValueError, match="Unknown model"):
            TraditionalClassifier(model_name="invalid_model")

    def test_naive_bayes_trains(self):
        from src.models.traditional import TraditionalClassifier
        clf = TraditionalClassifier(
            model_name="naive_bayes",
            categories=CATEGORIES,
            tfidf_config={"max_features": 500},
        )
        clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
        preds = clf.predict(TEST_TEXTS)
        assert len(preds) == len(TEST_TEXTS)

    def test_svm_trains(self):
        from src.models.traditional import TraditionalClassifier
        clf = TraditionalClassifier(
            model_name="svm",
            categories=CATEGORIES,
            tfidf_config={"max_features": 500},
        )
        clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
        preds = clf.predict(TEST_TEXTS)
        assert len(preds) == len(TEST_TEXTS)


class TestModelEvaluator:

    def test_evaluate_returns_metrics(self):
        from src.models.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(categories=CATEGORIES)
        y_true = [0, 1, 2, 3, 0, 1]
        y_pred = [0, 1, 2, 3, 1, 0]
        metrics = evaluator.evaluate("test_model", y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics
        assert "per_class" in metrics

    def test_compare_multiple_models(self):
        from src.models.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(categories=CATEGORIES)
        y_true = [0, 1, 2, 3]
        evaluator.evaluate("model_a", y_true, [0, 1, 2, 3])  # Perfect
        evaluator.evaluate("model_b", y_true, [0, 1, 2, 0])  # Imperfect
        df = evaluator.compare()
        assert df.iloc[0]["model"] == "model_a"  # Best model first
        assert len(df) == 2
