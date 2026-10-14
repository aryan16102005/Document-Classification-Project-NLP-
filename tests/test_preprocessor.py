"""
Tests for data preprocessing pipeline.
tests/test_preprocessor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.data.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatization=False,   # Skip for speed
        stemming=False,
        min_word_length=2,
    )


class TestTextPreprocessor:

    def test_lowercase(self, preprocessor):
        result = preprocessor.clean_text("HELLO WORLD")
        assert result == result.lower()

    def test_url_removal(self, preprocessor):
        result = preprocessor.clean_text("Visit https://example.com for more info")
        assert "https" not in result
        assert "example" not in result

    def test_html_removal(self, preprocessor):
        result = preprocessor.clean_text("<p>Hello <b>World</b></p>")
        assert "<" not in result
        assert ">" not in result

    def test_punctuation_removal(self, preprocessor):
        result = preprocessor.clean_text("Hello, World! How are you?")
        assert "," not in result
        assert "!" not in result

    def test_empty_string(self, preprocessor):
        assert preprocessor.preprocess("") == ""

    def test_whitespace_normalization(self, preprocessor):
        result = preprocessor.clean_text("Hello    World   ")
        assert "  " not in result

    def test_stop_word_removal(self, preprocessor):
        result = preprocessor.preprocess("The quick brown fox jumps over the lazy dog")
        tokens = result.split()
        assert "the" not in tokens
        assert "over" not in tokens

    def test_min_word_length(self):
        proc = TextPreprocessor(min_word_length=4, remove_stopwords=False, lemmatization=False)
        result = proc.preprocess("a to is quick brown fox")
        tokens = result.split()
        assert all(len(t) >= 4 for t in tokens)

    def test_batch_preprocessing(self, preprocessor):
        texts = ["Hello World", "Machine Learning is great", "Natural Language Processing"]
        results = preprocessor.preprocess_batch(texts, show_progress=False)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_email_removal(self, preprocessor):
        result = preprocessor.clean_text("Contact us at hello@example.com for support")
        assert "@" not in result


class TestDataIntegration:

    def test_generator_creates_dataframe(self):
        from src.data.generator import SampleDataGenerator
        gen = SampleDataGenerator()
        df = gen.generate(n_per_category=10)
        assert len(df) == 80  # 8 categories × 10 samples
        assert "text" in df.columns
        assert "label" in df.columns
        assert "label_id" in df.columns
        assert df["label"].nunique() == 8

    def test_loader_validates_columns(self, tmp_path):
        from src.data.loader import DataLoader
        import pandas as pd
        df = pd.DataFrame({"content": ["Hello"], "category": ["tech"]})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        loader = DataLoader(text_col="text", label_col="label")
        with pytest.raises(ValueError, match="'text' not found"):
            loader.load(str(csv_path))

    def test_loader_splits(self, tmp_path):
        from src.data.generator import SampleDataGenerator
        from src.data.loader import DataLoader
        gen = SampleDataGenerator()
        df = gen.generate(n_per_category=50)
        loader = DataLoader()
        # Manually set label map
        loader._categories = sorted(df["label"].unique().tolist())
        loader._label_map = {cat: i for i, cat in enumerate(loader._categories)}
        df["label_id"] = df["label"].map(loader._label_map)
        train, val, test = loader.split(df, train_ratio=0.70, val_ratio=0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(df)
        assert len(train) > len(val)
        assert len(train) > len(test)
