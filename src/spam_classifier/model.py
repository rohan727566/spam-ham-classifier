"""
Model management module for spam classification
"""
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from .config import Config
from .preprocess import clean_text


class SpamClassifier:
    """Spam/Ham classifier using Naive Bayes and TF-IDF"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = ["ham", "spam"]

    def create_model(self) -> MultinomialNB:
        """Create a new Multinomial Naive Bayes model"""
        return MultinomialNB()

    def create_vectorizer(self) -> TfidfVectorizer:
        """Create a new TF-IDF vectorizer with configured parameters"""
        return TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            min_df=Config.MIN_DF,
            max_df=Config.MAX_DF,
            ngram_range=Config.NGRAM_RANGE,
        )

    def train(self, X_train: List[str], y_train: np.ndarray):
        """
        Train the model on preprocessed text data

        Args:
            X_train: List of cleaned text strings
            y_train: Array of labels (0=ham, 1=spam)
        """
        print("Training classifier...")

        # Create and fit vectorizer
        self.vectorizer = self.create_vectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  Feature matrix shape: {X_train_vec.shape}")

        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train_vec, y_train)

        print("✓ Training complete")

    def predict(self, text: str) -> Dict:
        """
        Predict spam/ham for a single text

        Args:
            text: Raw text string

        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError(
                "Model not trained or loaded. Call train() or load() first."
            )

        # Clean and vectorize
        cleaned = clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])

        # Predict
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]

        # Get class probabilities
        class_probs = {
            self.classes[i]: float(prob) for i, prob in enumerate(probabilities)
        }

        return {
            "label": self.classes[prediction],
            "confidence": float(max(probabilities)),
            "probabilities": class_probs,
            "is_spam": prediction == 1,
            "cleaned_text": cleaned,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict spam/ham for multiple texts

        Args:
            texts: List of raw text strings

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]

    def save(self, model_path: Path = None, vectorizer_path: Path = None):
        """
        Save model and vectorizer to disk

        Args:
            model_path: Path to save model (default from Config)
            vectorizer_path: Path to save vectorizer (default from Config)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("No model to save. Train first.")

        model_path = model_path or Config.get_model_path()
        vectorizer_path = vectorizer_path or Config.get_vectorizer_path()

        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Vectorizer saved to: {vectorizer_path}")

    def load(self, model_path: Path = None, vectorizer_path: Path = None):
        """
        Load model and vectorizer from disk

        Args:
            model_path: Path to model file (default from Config)
            vectorizer_path: Path to vectorizer file (default from Config)
        """
        model_path = model_path or Config.get_model_path()
        vectorizer_path = vectorizer_path or Config.get_vectorizer_path()

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

        # Load
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Vectorizer loaded from: {vectorizer_path}")

    def get_feature_importance(
        self, top_n: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features (words) for each class

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary with top features for ham and spam
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded.")

        feature_names = self.vectorizer.get_feature_names_out()

        result = {}
        for idx, class_name in enumerate(self.classes):
            # Get log probabilities for this class
            log_probs = self.model.feature_log_prob_[idx]

            # Get top N features
            top_indices = np.argsort(log_probs)[-top_n:][::-1]
            top_features = [(feature_names[i], log_probs[i]) for i in top_indices]

            result[class_name] = top_features

        return result
