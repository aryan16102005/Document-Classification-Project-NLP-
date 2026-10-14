"""
Data loader — reads CSV / JSON / TXT datasets and provides train/val/test splits.
src/data/loader.py
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Flexible data loader that supports multiple file formats and
    returns ready-to-use train / validation / test DataFrames.
    """

    SUPPORTED_FORMATS = {".csv", ".json", ".jsonl", ".txt"}

    def __init__(
        self,
        text_col: str = "text",
        label_col: str = "label",
        random_seed: int = 42,
    ):
        self.text_col = text_col
        self.label_col = label_col
        self.random_seed = random_seed
        self._label_map: Optional[Dict[str, int]] = None
        self._categories: Optional[List[str]] = None


    @property
    def label_map(self) -> Dict[str, int]:
        if self._label_map is None:
            raise RuntimeError("No data loaded yet. Call load() first.")
        return self._label_map

    @property
    def categories(self) -> List[str]:
        if self._categories is None:
            raise RuntimeError("No data loaded yet. Call load() first.")
        return self._categories

    @property
    def num_classes(self) -> int:
        return len(self.categories)

    def _read_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def _read_json(self, path: Path) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _read_jsonl(self, path: Path) -> pd.DataFrame:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def _read_txt(self, path: Path) -> pd.DataFrame:
        """Expects lines in the format: LABEL\tTEXT"""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    records.append({self.label_col: parts[0], self.text_col: parts[1]})
        return pd.DataFrame(records)

    def _read_file(self, path: Union[str, Path]) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        readers = {
            ".csv": self._read_csv,
            ".json": self._read_json,
            ".jsonl": self._read_jsonl,
            ".txt": self._read_txt,
        }
        if suffix not in readers:
            raise ValueError(f"Unsupported format: {suffix}. Supported: {self.SUPPORTED_FORMATS}")
        logger.info(f"Reading {suffix} file: {path}")
        return readers[suffix](path)

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns exist and drop empty rows."""
        for col in [self.text_col, self.label_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset. Found: {list(df.columns)}")
        before = len(df)
        df = df.dropna(subset=[self.text_col, self.label_col])
        df = df[df[self.text_col].astype(str).str.strip() != ""]
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with empty text/label.")
        return df.reset_index(drop=True)

    def _build_label_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode string labels to integer IDs."""
        categories = sorted(df[self.label_col].unique().tolist())
        self._categories = categories
        self._label_map = {cat: idx for idx, cat in enumerate(categories)}
        df = df.copy()
        df["label_id"] = df[self.label_col].map(self._label_map)
        logger.info(f"Categories ({len(categories)}): {categories}")
        return df

    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a single file into a validated DataFrame with label IDs.

        Args:
            path: Path to the dataset file.

        Returns:
            DataFrame with columns: [text, label, label_id]
        """
        df = self._read_file(path)
        df = self._validate(df)
        df = self._build_label_map(df)
        logger.success(f"Loaded {len(df)} samples from {path}")
        return df

    def load_splits(
        self,
        directory: Union[str, Path],
        train_file: str = "train.csv",
        val_file: str = "val.csv",
        test_file: str = "test.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load pre-split train / val / test files from a directory.

        Returns:
            (train_df, val_df, test_df)
        """
        directory = Path(directory)
        train_df = self.load(directory / train_file)
        # Re-use label map from training set for val/test
        val_df = self._read_file(directory / val_file)
        test_df = self._read_file(directory / test_file)
        for df in [val_df, test_df]:
            df = self._validate(df)
            df["label_id"] = df[self.label_col].map(self._label_map)

        val_df = self._validate(val_df)
        test_df = self._validate(test_df)
        val_df["label_id"] = val_df[self.label_col].map(self._label_map)
        test_df["label_id"] = test_df[self.label_col].map(self._label_map)

        logger.success(
            f"Loaded splits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
        )
        return train_df, val_df, test_df

    def split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a single DataFrame into train / val / test subsets.

        Args:
            df: Full dataset.
            train_ratio: Fraction for training (default 0.70).
            val_ratio: Fraction for validation (default 0.15); test = remainder.
            stratify: Whether to preserve class distribution across splits.

        Returns:
            (train_df, val_df, test_df)
        """
        stratify_col = df[self.label_col] if stratify else None
        test_ratio = 1.0 - train_ratio - val_ratio

        train, temp = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=stratify_col,
            random_state=self.random_seed,
        )
        val_frac = val_ratio / (val_ratio + test_ratio)
        stratify_temp = temp[self.label_col] if stratify else None
        val, test = train_test_split(
            temp,
            test_size=(1.0 - val_frac),
            stratify=stratify_temp,
            random_state=self.random_seed,
        )
        logger.info(f"Split → train={len(train)}, val={len(val)}, test={len(test)}")
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def get_class_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with class counts and percentages."""
        counts = df[self.label_col].value_counts().reset_index()
        counts.columns = ["label", "count"]
        counts["percentage"] = (counts["count"] / len(df) * 100).round(2)
        return counts
