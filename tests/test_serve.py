import os
import sys
import pytest
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient
from unittest.mock import patch


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


@pytest.fixture
def mock_model_file(tmp_path):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    rng = np.random.default_rng(0)
    X = rng.random((100, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=100)
    model.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, str(model_path))
    return model_path


@pytest.fixture
def serve_app(mock_model_file, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("HOME", str(mock_model_file.parent))
    model_path = os.path.expanduser("~/models/model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(joblib.load(str(mock_model_file)), model_path)

    with patch("google.cloud.storage.Client"):
        import importlib
        import src.serve
        importlib.reload(src.serve)
        yield src.serve.app


@pytest.fixture
def client(serve_app):
    return TestClient(serve_app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestPredict:
    def test_predict_with_valid_features(self, client):
        features = [0.5] * 12
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "label" in data
        assert data["label"] in ("thap", "trung_binh", "cao")

    def test_predict_wrong_feature_count(self, client):
        features = [0.5] * 11
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "12 features" in resp.json()["detail"]

    def test_predict_empty_features(self, client):
        resp = client.post("/predict", json={"features": []})
        assert resp.status_code == 400

    def test_predict_missing_features_field(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


class TestPredictWithoutModel:
    @pytest.fixture
    def serve_app_no_model(self, mock_model_file, monkeypatch):
        monkeypatch.setenv("GCS_BUCKET", "test-bucket")
        monkeypatch.setenv("HOME", str(mock_model_file.parent))
        with patch("google.cloud.storage.Client"):
            import importlib
            with patch("os.path.exists", return_value=False):
                import src.serve
                importlib.reload(src.serve)
                yield src.serve.app

    @pytest.fixture
    def client(self, serve_app_no_model):
        return TestClient(serve_app_no_model)

    def test_predict_returns_503_when_no_model(self, client):
        features = [0.5] * 12
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()
