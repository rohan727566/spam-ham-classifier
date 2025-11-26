"""
Unit tests for spam classifier model
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.spam_classifier.config import Config
from src.spam_classifier.model import SpamClassifier


class TestSpamClassifier:
    """Test suite for SpamClassifier class"""

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance"""
        return SpamClassifier()

    @pytest.fixture
    def sample_data(self):
        """Sample training data"""
        X_train = [
            "great product love it",
            "amazing service very happy",
            "free money click now",
            "win prize call immediately",
        ]
        y_train = np.array([0, 0, 1, 1])  # 0=ham, 1=spam

        X_test = ["thanks for the update", "congratulations you won cash"]
        y_test = np.array([0, 1])

        return X_train, X_test, y_train, y_test

    def test_classifier_initialization(self, classifier):
        """Test classifier is initialized correctly"""
        assert classifier.model is None
        assert classifier.vectorizer is None
        assert classifier.classes == ["ham", "spam"]

    def test_create_model(self, classifier):
        """Test model creation"""
        model = classifier.create_model()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_vectorizer(self, classifier):
        """Test vectorizer creation"""
        vectorizer = classifier.create_vectorizer()
        assert vectorizer is not None
        assert hasattr(vectorizer, "fit_transform")
        assert vectorizer.max_features == Config.MAX_FEATURES

    def test_training(self, classifier, sample_data):
        """Test model training"""
        X_train, _, y_train, _ = sample_data

        classifier.train(X_train, y_train)

        assert classifier.model is not None
        assert classifier.vectorizer is not None
        assert len(classifier.vectorizer.vocabulary_) > 0

    def test_predict_single_ham(self, classifier, sample_data):
        """Test single prediction for ham message"""
        X_train, _, y_train, _ = sample_data
        classifier.train(X_train, y_train)

        result = classifier.predict("thank you for helping")

        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "is_spam" in result
        assert result["label"] in ["ham", "spam"]
        assert 0 <= result["confidence"] <= 1

    def test_predict_single_spam(self, classifier, sample_data):
        """Test single prediction for spam message"""
        X_train, _, y_train, _ = sample_data
        classifier.train(X_train, y_train)

        result = classifier.predict("free cash prize click now")

        assert result["label"] == "spam"
        assert result["is_spam"] is True
        assert result["confidence"] > 0.5

    def test_predict_batch(self, classifier, sample_data):
        """Test batch prediction"""
        X_train, _, y_train, _ = sample_data
        classifier.train(X_train, y_train)

        texts = ["thanks for help", "free money now", "good service"]
        results = classifier.predict_batch(texts)

        assert len(results) == 3
        assert all("label" in r for r in results)
        assert all("confidence" in r for r in results)

    def test_save_and_load(self, classifier, sample_data):
        """Test model save and load"""
        X_train, _, y_train, _ = sample_data
        classifier.train(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            vec_path = Path(tmpdir) / "test_vec.pkl"

            # Save
            classifier.save(model_path, vec_path)
            assert model_path.exists()
            assert vec_path.exists()

            # Load into new classifier
            new_classifier = SpamClassifier()
            new_classifier.load(model_path, vec_path)

            # Test loaded model
            result = new_classifier.predict("thank you")
            assert result["label"] in ["ham", "spam"]

    def test_load_nonexistent_file(self, classifier):
        """Test loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            classifier.load(Path("nonexistent_model.pkl"), Path("nonexistent_vec.pkl"))

    def test_predict_without_training(self, classifier):
        """Test prediction without training raises error"""
        with pytest.raises(ValueError):
            classifier.predict("test text")

    def test_probabilities_sum_to_one(self, classifier, sample_data):
        """Test probabilities sum to 1.0"""
        X_train, _, y_train, _ = sample_data
        classifier.train(X_train, y_train)

        result = classifier.predict("test message")
        probs = result["probabilities"]

        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01  # Allow small floating point error

    def test_feature_importance(self, classifier, sample_data):
        """Test feature importance extraction"""
        X_train, _, y_train, _ = sample_data
        classifier.train(X_train, y_train)

        features = classifier.get_feature_importance(top_n=5)

        assert "ham" in features
        assert "spam" in features
        assert len(features["ham"]) <= 5
        assert len(features["spam"]) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
