"""
Configuration module for Spam Classifier
"""
import os
from pathlib import Path


class Config:
    """Central configuration for the spam classifier system"""

    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "model"
    DOCS_DIR = BASE_DIR / "docs"
    PLOTS_DIR = DOCS_DIR / "plots"

    # Dataset
    DATASET_PATH = DATA_DIR / "SMSSpamCollection.tsv"
    DATASET_COLUMNS = ["label", "body_text"]

    # Model parameters
    MODEL_NAME = "spam_classifier_nb"
    VECTORIZER_NAME = "tfidf_vectorizer"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Text preprocessing
    MAX_FEATURES = 3000
    MIN_DF = 2
    MAX_DF = 0.8
    NGRAM_RANGE = (1, 2)

    # NLTK resources
    NLTK_RESOURCES = ["stopwords", "punkt", "wordnet", "omw-1.4"]

    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_RELOAD = os.getenv("API_RELOAD", "False").lower() == "true"

    # Prediction thresholds
    SPAM_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.6

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_path(cls, model_name: str = None) -> Path:
        """Get full path for model file"""
        name = model_name or cls.MODEL_NAME
        return cls.MODEL_DIR / f"{name}.pkl"

    @classmethod
    def get_vectorizer_path(cls) -> Path:
        """Get full path for vectorizer file"""
        return cls.MODEL_DIR / f"{cls.VECTORIZER_NAME}.pkl"
