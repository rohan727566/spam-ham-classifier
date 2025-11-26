"""
API tests for FastAPI spam classifier server
"""
import pytest
from fastapi.testclient import TestClient

from src.spam_classifier.model import SpamClassifier
from src.spam_classifier.server import app


@pytest.fixture(scope="session", autouse=True)
def ensure_model_loaded():
    """
    Ensure a trained model is available for API tests.
    Uses the already trained model files in the 'model' directory.
    """
    global classifier
    if classifier is None or classifier.model is None:
        clf = SpamClassifier()
        clf.load()
        # Assign to global used by server
        globals()["classifier"] = clf
    return classifier


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app"""
    return TestClient(app)


def test_health_check(client):
    """Health endpoint should return healthy status and model_loaded True"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_predict_ham(client):
    """Predict endpoint should classify a ham message correctly"""
    payload = {"text": "Hey, are we still meeting for lunch tomorrow?"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["label"] in ["ham", "spam"]
    assert "confidence" in data
    assert "probabilities" in data
    assert "is_spam" in data
    assert "cleaned_text" in data
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["probabilities"], dict)


def test_predict_spam(client):
    """Predict endpoint should classify a spam message with high spam probability"""
    payload = {"text": "WINNER! You have won $1000 cash prize. Call now to claim FREE!"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["label"] in ["ham", "spam"]
    assert 0 <= data["confidence"] <= 1
    # For typical spam message, spam probability should be higher
    assert data["probabilities"]["spam"] >= data["probabilities"]["ham"]


def test_predict_validation_error(client):
    """Empty text should result in 422 validation error"""
    payload = {"text": ""}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422  # FastAPI validation


def test_predict_batch(client):
    """Batch prediction should return list of predictions"""
    payload = {
        "texts": [
            "Thanks for your help today.",
            "FREE prize waiting for you, click now!",
        ]
    }
    response = client.post("/api/predict/batch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert "label" in item
        assert "confidence" in item
        assert "probabilities" in item
        assert "is_spam" in item


def test_predict_batch_limit(client):
    """Batch prediction should enforce max 100 texts"""
    payload = {"texts": ["test"] * 101}
    response = client.post("/api/predict/batch", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "Maximum 100 texts" in data["detail"]


def test_predict_file_wrong_extension(client, tmp_path):
    """File upload should reject non-txt files"""
    file_path = tmp_path / "emails.csv"
    file_path.write_text("test message")

    with file_path.open("rb") as f:
        files = {"file": ("emails.csv", f, "text/csv")}
        response = client.post("/api/predict/file", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "Only .txt files are supported" in data["detail"]
