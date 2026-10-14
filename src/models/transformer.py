"""
Transformer-based (DistilBERT) document classifier with HuggingFace Transformers.
src/models/transformer.py
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class TransformerClassifier:
    """
    Fine-tunes a HuggingFace transformer model for multi-class text classification.

    Default model: distilbert-base-uncased (lightweight, fast, near-BERT accuracy).
    Supports any classification-compatible model on HuggingFace Hub.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 8,
        max_length: int = 256,
        categories: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.categories = categories or []
        self._model = None
        self._tokenizer = None
        self._is_fitted = False

        # Auto-detect device
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(
            f"TransformerClassifier | model={model_name} | "
            f"num_labels={num_labels} | device={self.device}"
        )

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def _load_model_and_tokenizer(self):
        """Lazy-load the model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            label2id = {cat: i for i, cat in enumerate(self.categories)} if self.categories else {}
            id2label = {i: cat for i, cat in enumerate(self.categories)} if self.categories else {}

            logger.info(f"Loading model: {self.model_name}")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            )
            self._model.to(self.device)
        except ImportError as e:
            raise ImportError(
                f"transformers and torch are required for TransformerClassifier. "
                f"Install with: pip install transformers torch\n{e}"
            )

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune the transformer on document classification task.

        Args:
            train_texts: Raw or lightly cleaned texts (transformer handles tokenization).
            train_labels: Integer label IDs.
            val_texts: Validation texts for periodic evaluation.
            val_labels: Validation label IDs.
            num_epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: AdamW learning rate.
            warmup_steps: Scheduler warmup steps.
            weight_decay: L2 regularization.
            save_dir: Directory to save checkpoints.

        Returns:
            Dict with training history metrics.
        """
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            from torch.optim import AdamW
            from transformers import get_linear_schedule_with_warmup
            from sklearn.metrics import f1_score
        except ImportError as e:
            raise ImportError(f"Install torch and transformers: {e}")

        self._load_model_and_tokenizer()
        logger.info(f"Fine-tuning {self.model_name} for {num_epochs} epochs...")

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    "input_ids": self.encodings["input_ids"][idx],
                    "attention_mask": self.encodings["attention_mask"][idx],
                    "labels": self.labels[idx],
                }

        train_dataset = TextDataset(train_texts, train_labels, self._tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_texts and val_labels:
            val_dataset = TextDataset(val_texts, val_labels, self._tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(
            self._model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        history = {"train_loss": [], "val_f1_macro": []}
        best_val_f1 = 0.0
        t0 = time.time()

        for epoch in range(num_epochs):
            self._model.train()
            total_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(round(avg_loss, 4))

            # Validation
            if val_loader:
                val_f1 = self._evaluate_loader(val_loader)
                history["val_f1_macro"].append(round(val_f1, 4))
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}"
                )
                if val_f1 > best_val_f1 and save_dir:
                    best_val_f1 = val_f1
                    self.save(save_dir + "/best")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

        self._is_fitted = True
        elapsed = time.time() - t0
        history["train_time_sec"] = round(elapsed, 2)
        logger.success(f"Training complete in {elapsed:.1f}s")
        return history

    def _evaluate_loader(self, loader) -> float:
        """Evaluate model on a DataLoader and return macro F1."""
        import torch
        from sklearn.metrics import f1_score

        self._model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].cpu().numpy()

                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        return f1_score(all_labels, all_preds, average="macro")

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, texts: List[str]) -> np.ndarray:
        """Return predicted label IDs."""
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return class probability matrix (shape: [n, num_labels])."""
        import torch
        import torch.nn.functional as F

        self._check_fitted()
        self._model.eval()

        encodings = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            proba = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        return proba

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict a single document with confidence scores."""
        proba = self.predict_proba([text])[0]
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
        """Save model, tokenizer, and metadata to a directory."""
        self._check_fitted()
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(save_path))
        self._tokenizer.save_pretrained(str(save_path))

        meta = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "categories": self.categories,
        }
        with open(save_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.success(f"Transformer model saved → {save_path}")
        return str(save_path)

    @classmethod
    def load(cls, model_dir: str) -> "TransformerClassifier":
        """Load a saved transformer classifier from a directory."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_path = Path(model_dir)
        with open(model_path / "meta.json") as f:
            meta = json.load(f)

        obj = cls(
            model_name=str(model_path),
            num_labels=meta["num_labels"],
            max_length=meta["max_length"],
            categories=meta["categories"],
        )
        obj._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        obj._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        obj._model.to(obj.device)
        obj._is_fitted = True
        logger.success(f"Transformer model loaded from {model_path}")
        return obj

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
