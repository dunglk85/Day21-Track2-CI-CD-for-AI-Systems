import json
import os
import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import generate_report


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


@pytest.fixture
def model_and_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, str(model_path))

    eval_path = tmp_path / "eval.csv"
    df.to_csv(str(eval_path), index=False)

    metrics_dir = tmp_path / "outputs"
    metrics_dir.mkdir(exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": 0.85, "f1_score": 0.80}, f)

    return model_path, eval_path, metrics_path


class TestFormatConfusionMatrix:
    def test_header_format(self):
        cm = [[5, 1, 0], [0, 6, 1], [1, 0, 7]]
        result = generate_report.format_confusion_matrix(cm, [0, 1, 2])
        assert "Pred_0" in result
        assert "Pred_1" in result
        assert "Pred_2" in result
        assert "True_0" in result
        assert "True_1" in result
        assert "True_2" in result

    def test_values_appear(self):
        cm = [[5, 1, 0], [0, 6, 1], [1, 0, 7]]
        result = generate_report.format_confusion_matrix(cm, [0, 1, 2])
        for row in cm:
            for val in row:
                assert str(val) in result

    def test_single_class(self):
        cm = [[10]]
        result = generate_report.format_confusion_matrix(cm, [0])
        assert "Pred_0" in result
        assert "True_0" in result
        assert "10" in result

    def test_returns_multiline_string(self):
        cm = [[5, 0], [0, 5]]
        result = generate_report.format_confusion_matrix(cm, [0, 1])
        assert "\n" in result


class TestLoadExistingMetrics:
    def test_returns_empty_dict_when_no_file(self, monkeypatch):
        monkeypatch.setattr(generate_report, "METRICS_PATH", "/nonexistent/path.json")
        result = generate_report.load_existing_metrics()
        assert result == {}

    def test_returns_content_when_file_exists(self, tmp_path):
        metrics_file = tmp_path / "metrics.json"
        data = {"accuracy": 0.9, "f1_score": 0.88}
        with open(metrics_file, "w") as f:
            json.dump(data, f)
        original = generate_report.METRICS_PATH
        generate_report.METRICS_PATH = str(metrics_file)
        try:
            result = generate_report.load_existing_metrics()
            assert result == data
        finally:
            generate_report.METRICS_PATH = original


class TestLoadDistribution:
    def test_returns_empty_dict_when_no_file(self, monkeypatch):
        monkeypatch.setattr(generate_report, "DISTRIBUTION_PATH", "/nonexistent/path.json")
        result = generate_report.load_distribution()
        assert result == {}

    def test_returns_content_when_file_exists(self, tmp_path):
        dist_file = tmp_path / "class_distribution.json"
        data = {"class_distribution": {"0": {"count": 50, "ratio": 0.5}}, "is_imbalanced": False}
        with open(dist_file, "w") as f:
            json.dump(data, f)
        original = generate_report.DISTRIBUTION_PATH
        generate_report.DISTRIBUTION_PATH = str(dist_file)
        try:
            result = generate_report.load_distribution()
            assert result == data
        finally:
            generate_report.DISTRIBUTION_PATH = original


class TestMain:
    def test_creates_report_and_updates_metrics(self, model_and_data, monkeypatch):
        model_path, eval_path, metrics_path = model_and_data
        output_dir = metrics_path.parent
        report_path = output_dir / "report.txt"
        dist_file = output_dir / "class_distribution.json"
        dist_file.write_text(json.dumps({}))

        monkeypatch.setattr(generate_report, "MODEL_PATH", str(model_path))
        monkeypatch.setattr(generate_report, "TEST_DATA_PATH", str(eval_path))
        monkeypatch.setattr(generate_report, "METRICS_PATH", str(metrics_path))
        monkeypatch.setattr(generate_report, "REPORT_PATH", str(report_path))
        monkeypatch.setattr(generate_report, "DISTRIBUTION_PATH", str(dist_file))

        generate_report.main()

        assert report_path.exists()
        report_text = report_path.read_text()
        assert "BAO CAO HIEU SUAT MO HINH" in report_text
        assert "Accuracy" in report_text
        assert "F1-score" in report_text
        assert "Confusion Matrix" in report_text

        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert "report_accuracy" in metrics
        assert "report_f1_score" in metrics
        assert "precision_recall_per_class" in metrics
        assert "accuracy" in metrics
        assert "f1_score" in metrics

    def test_main_with_class_distribution(self, model_and_data, monkeypatch):
        model_path, eval_path, metrics_path = model_and_data
        output_dir = metrics_path.parent
        report_path = output_dir / "report.txt"
        dist_file = output_dir / "class_distribution.json"
        dist_data = {
            "class_distribution": {
                "0": {"count": 20, "ratio": 0.4},
                "1": {"count": 15, "ratio": 0.3},
                "2": {"count": 15, "ratio": 0.3},
            },
            "is_imbalanced": False,
        }
        with open(dist_file, "w") as f:
            json.dump(dist_data, f)

        monkeypatch.setattr(generate_report, "MODEL_PATH", str(model_path))
        monkeypatch.setattr(generate_report, "TEST_DATA_PATH", str(eval_path))
        monkeypatch.setattr(generate_report, "METRICS_PATH", str(metrics_path))
        monkeypatch.setattr(generate_report, "REPORT_PATH", str(report_path))
        monkeypatch.setattr(generate_report, "DISTRIBUTION_PATH", str(dist_file))

        generate_report.main()

        assert report_path.exists()
        report_text = report_path.read_text()
        assert "Phan phoi nhan" in report_text
        assert "0.40%" in report_text or "40" in report_text

        metrics = json.loads(metrics_path.read_text())
        assert "class_distribution" in metrics

    def test_main_with_imbalance_warning(self, model_and_data, monkeypatch):
        model_path, eval_path, metrics_path = model_and_data
        output_dir = metrics_path.parent
        report_path = output_dir / "report.txt"
        dist_file = output_dir / "class_distribution.json"
        dist_data = {
            "class_distribution": {
                "0": {"count": 5, "ratio": 0.05},
                "1": {"count": 45, "ratio": 0.45},
                "2": {"count": 50, "ratio": 0.50},
            },
            "is_imbalanced": True,
        }
        with open(dist_file, "w") as f:
            json.dump(dist_data, f)

        monkeypatch.setattr(generate_report, "MODEL_PATH", str(model_path))
        monkeypatch.setattr(generate_report, "TEST_DATA_PATH", str(eval_path))
        monkeypatch.setattr(generate_report, "METRICS_PATH", str(metrics_path))
        monkeypatch.setattr(generate_report, "REPORT_PATH", str(report_path))
        monkeypatch.setattr(generate_report, "DISTRIBUTION_PATH", str(dist_file))

        generate_report.main()

        report_text = report_path.read_text()
        assert "CANH BAO" in report_text
