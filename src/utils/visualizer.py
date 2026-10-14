"""
Visualization utilities: confusion matrix, class distribution, feature importance, training curves.
src/utils/visualizer.py
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class Visualizer:
    """
    Generates and saves all visualization artifacts for the NLP system.
    Uses matplotlib + seaborn for static plots.
    """

    def __init__(self, output_dir: str = "artifacts/plots", style: str = "darkgrid"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import seaborn as sns
            sns.set_style(style)
        except ImportError:
            logger.warning("matplotlib/seaborn not available — plots will be skipped.")

    def _savefig(self, fig, filename: str) -> str:
        """Save a figure and return its path."""
        import matplotlib.pyplot as plt
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved plot → {path}")
        return str(path)

    def plot_class_distribution(
        self, df: pd.DataFrame, label_col: str = "label", title: str = "Class Distribution"
    ) -> str:
        """Bar chart of sample counts per class."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        counts = df[label_col].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        palette = sns.color_palette("husl", len(counts))
        sns.barplot(x=counts.index, y=counts.values, palette=palette, ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(ax.patches, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        return self._savefig(fig, "class_distribution.png")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        categories: List[str],
        model_name: str = "model",
        normalize: bool = True,
    ) -> str:
        """Heatmap of confusion matrix with optional normalization."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if normalize:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
            vmax = 1.0
        else:
            cm_display = cm
            fmt = "d"
            vmax = None

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=categories,
            yticklabels=categories,
            vmin=0,
            vmax=vmax,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
        fname = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        return self._savefig(fig, fname)

    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> str:
        """Grouped bar chart comparing models on multiple metrics."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
        available = [m for m in metrics if m in comparison_df.columns]

        df_melted = comparison_df[["model"] + available].melt(
            id_vars="model", var_name="metric", value_name="score"
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_melted, x="model", y="score", hue="metric", ax=ax)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", title="Metric")
        ax.tick_params(axis="x", rotation=20)
        return self._savefig(fig, "model_comparison.png")

    def plot_training_curve(
        self,
        history: Dict[str, List[float]],
        model_name: str = "transformer",
    ) -> str:
        """Line chart of training loss and validation F1 per epoch."""
        import matplotlib.pyplot as plt

        epochs = range(1, len(history.get("train_loss", [])) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if "train_loss" in history and history["train_loss"]:
            axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
            axes[0].set_title("Training Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()

        if "val_f1_macro" in history and history["val_f1_macro"]:
            axes[1].plot(epochs, history["val_f1_macro"], "g-o", label="Val F1-macro")
            axes[1].set_title("Validation F1-macro")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("F1 Score")
            axes[1].set_ylim(0, 1.05)
            axes[1].legend()

        fig.suptitle(f"Training Curves — {model_name}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fname = f"training_curve_{model_name.replace(' ', '_')}.png"
        return self._savefig(fig, fname)

    def plot_top_features(
        self,
        top_features: Dict[str, List],
        top_n: int = 15,
    ) -> str:
        """Horizontal bar charts of top TF-IDF features per class."""
        import matplotlib.pyplot as plt

        categories = list(top_features.keys())
        n_cats = len(categories)
        cols = min(4, n_cats)
        rows = (n_cats + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if n_cats == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (cat, features) in enumerate(top_features.items()):
            row, col = divmod(idx, cols)
            ax = axes[row][col]
            features = features[:top_n]
            words = [f[0] for f in features]
            scores = [f[1] for f in features]
            ax.barh(words[::-1], scores[::-1], color="steelblue")
            ax.set_title(cat.capitalize(), fontsize=11, fontweight="bold")
            ax.set_xlabel("Coefficient")

        # Hide empty subplots
        for idx in range(n_cats, rows * cols):
            row, col = divmod(idx, cols)
            axes[row][col].set_visible(False)

        fig.suptitle("Top TF-IDF Features per Category", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return self._savefig(fig, "top_features.png")

    def plot_wordcloud(
        self,
        texts: List[str],
        label: str,
        max_words: int = 100,
    ) -> str:
        """Word cloud for a given set of texts."""
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            combined = " ".join(texts)
            wc = WordCloud(
                width=800,
                height=400,
                max_words=max_words,
                background_color="white",
                colormap="viridis",
            ).generate(combined)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Word Cloud — {label}", fontsize=13, fontweight="bold")
            fname = f"wordcloud_{label.replace(' ', '_')}.png"
            return self._savefig(fig, fname)
        except ImportError:
            logger.warning("wordcloud not available — skipping word cloud.")
            return ""

    def plot_text_length_distribution(
        self, df: pd.DataFrame, text_col: str = "text"
    ) -> str:
        """Histogram of document word counts."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        lengths = df[text_col].fillna("").apply(lambda x: len(str(x).split()))
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(lengths, bins=50, kde=True, color="steelblue", ax=ax)
        ax.axvline(lengths.mean(), color="red", linestyle="--", label=f"Mean: {lengths.mean():.0f}")
        ax.axvline(lengths.median(), color="orange", linestyle="--", label=f"Median: {lengths.median():.0f}")
        ax.set_title("Document Length Distribution (words)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.legend()
        return self._savefig(fig, "text_length_distribution.png")
