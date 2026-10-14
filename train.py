"""
Main training pipeline — generates data, preprocesses, trains all models, evaluates and saves.
train.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
import pandas as pd

from src.data.generator import SampleDataGenerator
from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.models.traditional import TraditionalClassifier
from src.models.evaluator import ModelEvaluator
from src.utils.config import load_config
from src.utils.visualizer import Visualizer


def run_training(config_path: str = "configs/config.yaml"):
    """
    Full end-to-end training pipeline:
    1. Generate / load data
    2. Preprocess text
    3. Train all traditional ML models
    4. Evaluate and compare
    5. Save best model + reports
    """
    logger.info("=" * 60)
    logger.info("   NLP DOCUMENT CLASSIFICATION SYSTEM — Training Pipeline")
    logger.info("=" * 60)

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    prep_cfg = config.get("preprocessing", {})
    tfidf_cfg = config.get("traditional_ml", {}).get("tfidf", {})
    models_cfg = config.get("traditional_ml", {}).get("models", [])
    categories = config.get("categories", [])
    artifacts_dir = config.get("training", {}).get("artifacts_dir", "artifacts")

    # ------------------------------------------------------------------ #
    # Step 1: Generate sample data                                        #
    # ------------------------------------------------------------------ #
    sample_dir = data_cfg.get("sample_dir", "data/sample")
    train_csv = Path(sample_dir) / "train.csv"

    if not train_csv.exists():
        logger.info("No dataset found — generating sample data...")
        gen = SampleDataGenerator(config=data_cfg)
        df = gen.generate(n_per_category=500)
        gen.save(df, output_dir=sample_dir)
    else:
        logger.info(f"Dataset already exists at {sample_dir}")

    # ------------------------------------------------------------------ #
    # Step 2: Load and preprocess                                         #
    # ------------------------------------------------------------------ #
    loader = DataLoader(random_seed=data_cfg.get("random_seed", 42))
    train_df, val_df, test_df = loader.load_splits(directory=sample_dir)

    logger.info(f"Label map: {loader.label_map}")
    categories = loader.categories

    preprocessor = TextPreprocessor(
        lowercase=prep_cfg.get("lowercase", True),
        remove_punctuation=prep_cfg.get("remove_punctuation", True),
        remove_stopwords=prep_cfg.get("remove_stopwords", True),
        lemmatization=prep_cfg.get("lemmatization", True),
        stemming=prep_cfg.get("stemming", False),
        min_word_length=prep_cfg.get("min_word_length", 2),
    )

    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.fit_transform(val_df)
    test_df = preprocessor.fit_transform(test_df)

    train_texts = train_df["text_clean"].tolist()
    val_texts = val_df["text_clean"].tolist()
    test_texts = test_df["text_clean"].tolist()
    train_labels = train_df["label_id"].tolist()
    val_labels = val_df["label_id"].tolist()
    test_labels = test_df["label_id"].tolist()

    # ------------------------------------------------------------------ #
    # Step 3: Visualize data                                              #
    # ------------------------------------------------------------------ #
    viz = Visualizer(output_dir=f"{artifacts_dir}/plots")
    viz.plot_class_distribution(train_df)
    viz.plot_text_length_distribution(train_df)

    # ------------------------------------------------------------------ #
    # Step 4: Train all traditional ML models                            #
    # ------------------------------------------------------------------ #
    evaluator = ModelEvaluator(categories=categories)
    best_f1 = 0.0
    best_model_path = None
    trained_models = {}

    default_models = [
        {"name": "logistic_regression", "params": {}},
        {"name": "svm", "params": {}},
        {"name": "naive_bayes", "params": {}},
        {"name": "random_forest", "params": {}},
    ]
    model_configs = models_cfg if models_cfg else default_models

    for model_cfg in model_configs:
        model_name = model_cfg.get("name", "logistic_regression")
        model_params = model_cfg.get("params", {})

        logger.info(f"\n{'─'*50}")
        logger.info(f"Training: {model_name}")

        clf = TraditionalClassifier(
            model_name=model_name,
            tfidf_config=tfidf_cfg,
            model_config=model_params,
            categories=categories,
        )

        train_metrics = clf.fit(
            train_texts, train_labels,
            val_texts=val_texts, val_labels=val_labels,
        )
        logger.info(f"Train metrics: {train_metrics}")

        # Evaluate on test set
        preds, proba, ms = evaluator.timed_predict(clf, test_texts)
        metrics = evaluator.evaluate(
            model_name=model_name,
            y_true=test_labels,
            y_pred=preds,
            y_proba=proba,
            inference_time_ms=ms,
        )

        # Confusion matrix plot
        import numpy as np
        cm = np.array(metrics["confusion_matrix"])
        viz.plot_confusion_matrix(cm, categories, model_name=model_name)

        # Save model
        model_path = clf.save(save_dir=f"{artifacts_dir}/models/{model_name}")
        trained_models[model_name] = clf

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model_path = model_path

    # ------------------------------------------------------------------ #
    # Step 5: Report & Comparison                                        #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON (Test Set)")
    logger.info("=" * 60)
    comparison_df = evaluator.print_comparison()
    viz.plot_model_comparison(comparison_df)

    # Top features from Logistic Regression
    if "logistic_regression" in trained_models:
        lr_model = trained_models["logistic_regression"]
        top_features = lr_model.get_top_features(top_n=15)
        if top_features:
            viz.plot_top_features(top_features)

    # Save evaluation report
    evaluator.save_report(output_dir=artifacts_dir)
    comparison_df.to_csv(f"{artifacts_dir}/model_comparison.csv", index=False)

    logger.success(f"\nBest model: F1-macro={best_f1:.4f} → {best_model_path}")
    logger.info(f"All artifacts saved to: {artifacts_dir}/")
    logger.info("Run the API server: python -m src.api.app")
    logger.info("Run the UI dashboard: streamlit run src/ui/app.py")

    return comparison_df, best_model_path


if __name__ == "__main__":
    run_training()
