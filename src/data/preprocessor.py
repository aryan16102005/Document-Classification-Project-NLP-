"""
Text preprocessing pipeline: cleaning, tokenization, stemming/lemmatization.
src/data/preprocessor.py
"""

import re
import string
from typing import List, Optional, Union
import pandas as pd
from loguru import logger


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline with configurable steps.
    
    Steps applied in order:
    1. Lowercase conversion
    2. URL / HTML removal
    3. Punctuation removal
    4. Digit removal (optional)
    5. Whitespace normalization
    6. Tokenization
    7. Stop-word removal
    8. Lemmatization or Stemming
    9. Short-word filtering
    """

    DEFAULT_STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through", "during",
        "before", "after", "above", "below", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "must", "can", "shall", "it", "its", "itself", "they", "them", "their",
        "what", "which", "who", "whom", "when", "where", "why", "how", "all",
        "each", "every", "both", "few", "more", "most", "other", "some", "such",
        "no", "not", "only", "same", "so", "than", "too", "very", "just",
        "he", "she", "we", "you", "i", "me", "him", "her", "us",
        "my", "your", "his", "our", "their", "as", "if", "then", "because",
    }

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_digits: bool = False,
        remove_stopwords: bool = True,
        lemmatization: bool = True,
        stemming: bool = False,
        min_word_length: int = 2,
        extra_stopwords: Optional[List[str]] = None,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords
        self.lemmatization = lemmatization
        self.stemming = stemming
        self.min_word_length = min_word_length

        # Build stop-word set
        self.stopwords = self.DEFAULT_STOPWORDS.copy()
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)

        # Lazy-load heavy NLP tools
        self._lemmatizer = None
        self._stemmer = None

        logger.debug(
            f"TextPreprocessor initialized | lowercase={lowercase} | "
            f"lemmatize={lemmatization} | stem={stemming}"
        )


    def _get_lemmatizer(self):
        """Lazy-load NLTK WordNet lemmatizer."""
        if self._lemmatizer is None:
            try:
                from nltk.stem import WordNetLemmatizer
                import nltk
                for resource in ["wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
                    try:
                        nltk.data.find(f"corpora/{resource}")
                    except LookupError:
                        nltk.download(resource, quiet=True)
                self._lemmatizer = WordNetLemmatizer()
            except ImportError:
                logger.warning("NLTK not available — skipping lemmatization.")
        return self._lemmatizer

    def _get_stemmer(self):
        """Lazy-load NLTK Porter stemmer."""
        if self._stemmer is None:
            try:
                from nltk.stem import PorterStemmer
                self._stemmer = PorterStemmer()
            except ImportError:
                logger.warning("NLTK not available — skipping stemming.")
        return self._stemmer


    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps to a raw string."""
        if not isinstance(text, str) or not text.strip():
            return ""


        if self.lowercase:
            text = text.lower()


        text = re.sub(r"https?://\S+|www\.\S+", " ", text)


        text = re.sub(r"<[^>]+>", " ", text)


        text = re.sub(r"\S+@\S+", " ", text)


        if self.remove_punctuation:
            text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

        if self.remove_digits:
            text = re.sub(r"\d+", " ", text)


        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Split cleaned text into tokens."""
        return text.split()

    def process_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stop-word removal, lemmatization/stemming, length filter."""

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]


        if self.lemmatization:
            lemmatizer = self._get_lemmatizer()
            if lemmatizer:
                tokens = [lemmatizer.lemmatize(t) for t in tokens]


        if self.stemming:
            stemmer = self._get_stemmer()
            if stemmer:
                tokens = [stemmer.stem(t) for t in tokens]


        tokens = [t for t in tokens if len(t) >= self.min_word_length]

        return tokens

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline → returns a clean string."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.process_tokens(tokens)
        return " ".join(tokens)

    def preprocess_batch(
        self,
        texts: Union[List[str], pd.Series],
        show_progress: bool = True,
    ) -> List[str]:
        """Preprocess a batch of texts with optional progress bar."""
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Preprocessing", disable=not show_progress)
        except ImportError:
            iterator = texts

        return [self.preprocess(t) for t in iterator]

    def fit_transform(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Add a 'text_clean' column to the DataFrame in-place.
        
        Args:
            df: Input DataFrame.
            text_col: Column name containing raw text.

        Returns:
            DataFrame with added 'text_clean' column.
        """
        logger.info(f"Preprocessing {len(df)} documents...")
        df = df.copy()
        df["text_clean"] = self.preprocess_batch(df[text_col].fillna(""))
        empty = (df["text_clean"].str.strip() == "").sum()
        if empty:
            logger.warning(f"{empty} documents became empty after preprocessing.")
        logger.success("Preprocessing complete.")
        return df
