"""
Text preprocessing module for spam classification
"""

import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .config import Config


def download_nltk_resources():
    """Download required NLTK resources if not already present"""
    for resource in Config.NLTK_RESOURCES:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


def clean_text(text: str) -> str:
    """
    Clean and preprocess text data

    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove stopwords
    4. Lemmatize words

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Ensure NLTK resources are available
    download_nltk_resources()

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned_words)


def preprocess_texts(texts: List[str], verbose: bool = True) -> List[str]:
    """
    Preprocess a list of texts

    Args:
        texts: List of raw text strings
        verbose: Print progress

    Returns:
        List of cleaned text strings
    """
    if verbose:
        print(f"Preprocessing {len(texts)} texts...")

    cleaned_texts = [clean_text(text) for text in texts]

    if verbose:
        print("✓ Preprocessing complete")
        print(
            f"  Average length before: {sum(len(t.split()) for t in texts) / len(texts):.1f} words"
        )
        print(
            f"  Average length after: {sum(len(t.split()) for t in cleaned_texts) / len(cleaned_texts):.1f} words"
        )

    return cleaned_texts


def get_text_stats(texts: List[str]) -> dict:
    """
    Get statistics about text data

    Args:
        texts: List of text strings

    Returns:
        Dictionary with statistics
    """
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]

    return {
        "total_texts": len(texts),
        "avg_words": sum(word_counts) / len(word_counts),
        "max_words": max(word_counts),
        "min_words": min(word_counts),
        "avg_chars": sum(char_counts) / len(char_counts),
        "max_chars": max(char_counts),
        "min_chars": min(char_counts),
    }
