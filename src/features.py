"""
features.py

Feature engineering and preprocessing builders for the ML pipeline.

Design principles:
- Keep preprocessing INSIDE the sklearn pipeline for reproducibility and safe inference.
- Handle numeric, categorical, and text data appropriately and separately.
- Use spaCy for robust tokenization + lemmatization.
- Cache the spaCy model to avoid re-loading it for every text.
"""

from __future__ import annotations

import re
from typing import List, Optional

import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---- Column configuration (explicit is better than implicit) ----
NUMERIC_COLS: List[str] = ["Age", "Positive Feedback Count", "Clothing ID"]
CATEGORICAL_COLS: List[str] = ["Division Name", "Department Name", "Class Name"]
TITLE_COL: str = "Title"
REVIEW_TEXT_COL: str = "Review Text"


# spaCy model cache: Loading spaCy inside the tokenizer repeatedly can make training extremely slow.
_NLP: ["spacy.language.Language"] = None


def _get_nlp() -> "spacy.language.Language":
    """
    Lazy-load and cache the spaCy model.
    """
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _NLP


def spacy_tokenizer(text: str) -> List[str]:
    """
    Tokenizer that performs normalization + lemmatization using spaCy.

    Returns:
        List[str]: cleaned tokens
    """
    if text is None:
        return []

    # Normalize and guard against non-string values
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    nlp = _get_nlp()
    doc = nlp(text)

    tokens: List[str] = []
    for token in doc:
        if token.is_space or token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_.strip()
        # Tiny-noise filter
        if not lemma or len(lemma) < 2:
            continue

        tokens.append(lemma)

    return tokens


# Custom transformer: concat Title + Review
# The title often summarizes the sentiment and the review text provides detailed context.
# Combined, they provide richer semantic signal for the model.
class TextConcatenator(BaseEstimator, TransformerMixin):
    """
    Concatenate Title and Review Text into a single text feature.

    Output shape:
    - Returns a 1D array-like of strings (one per row), suitable for TfidfVectorizer.
    """

    def __init__(self, title_col: str = TITLE_COL, text_col: str = REVIEW_TEXT_COL):
        self.title_col = title_col
        self.text_col = text_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a DataFrame with the two columns available
        title = X[self.title_col].fillna("").astype(str)
        text = X[self.text_col].fillna("").astype(str)

        combined = (title + " " + text).str.strip()
        return combined


def build_preprocessor(
    use_title: bool = True,
    max_features: int = 30000,
    ngram_range: tuple = (1, 2),
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - imputes numeric missing values
    - imputes + one-hot encodes categorical columns
    - concatenates title + review text
    - vectorizes text with TF-IDF using spaCy tokenizer

    Args:
        use_title: If True, concatenate Title + Review Text. If False, only Review Text is used.
        max_features: TF-IDF vocabulary cap
        ngram_range: TF-IDF n-grams

    Returns:
        ColumnTransformer ready to be used inside an sklearn Pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    tfidf = TfidfVectorizer(
        tokenizer=spacy_tokenizer,
        preprocessor=None,
        token_pattern=None,  # must be None when using a custom tokenizer
        max_features=max_features,
        ngram_range=ngram_range,
    )

    if use_title:
        text_pipeline = Pipeline(
            steps=[
                ("concat", TextConcatenator(title_col=TITLE_COL, text_col=REVIEW_TEXT_COL)),
                ("tfidf", tfidf),
            ]
        )
        text_selector = [TITLE_COL, REVIEW_TEXT_COL]
    else:
        text_pipeline = Pipeline(
            steps=[
                # For single-column text, TF-IDF can be fed directly.
                ("tfidf", tfidf),
            ]
        )
        text_selector = REVIEW_TEXT_COL

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
            ("txt", text_pipeline, text_selector),
        ],
        remainder="drop",
    )

    return preprocessor