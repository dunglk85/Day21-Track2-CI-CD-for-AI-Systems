"""
Bonus 3: Bao cao hieu suat tu dong sau khi train.
- Tinh confusion matrix (dang van ban).
- Tinh precision, recall cho tung lop (0, 1, 2).
- Ghi tat ca vao outputs/report.txt.
- Gop them class_distribution (tu Bonus 5) va f1_score vao outputs/metrics.json.

Chay SAU buoc "Train model" (yeu cau: models/model.pkl va data/test.csv da ton tai,
outputs/metrics.json da co it nhat truong 'accuracy' do train.py ghi ra).
"""

import json
import os
import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

MODEL_PATH = "models/model.pkl"     # TODO: doi neu train.py luu model o duong dan khac
TEST_DATA_PATH = "data/eval.csv"    # Da doi thanh duong dan test set that
TARGET_COLUMN = "target"            # TODO: doi thanh ten cot nhan that
CLASSES = [0, 1, 2]

METRICS_PATH = "outputs/metrics.json"
REPORT_PATH = "outputs/report.txt"
DISTRIBUTION_PATH = "outputs/class_distribution.json"


def load_existing_metrics() -> dict:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def load_distribution() -> dict:
    if os.path.exists(DISTRIBUTION_PATH):
        with open(DISTRIBUTION_PATH) as f:
            return json.load(f)
    return {}


def format_confusion_matrix(cm, classes) -> str:
    """Ve confusion matrix dang van ban, khong can matplotlib."""
    header = "        " + "".join(f"Pred_{c:<8}" for c in classes)
    lines = [header]
    for i, row in enumerate(cm):
        row_str = "".join(f"{val:<13}" for val in row)
        lines.append(f"True_{classes[i]:<3} {row_str}")
    return "\n".join(lines)


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    y_pred = model.predict(X_test)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    cm_text = format_confusion_matrix(cm, CLASSES)

    # --- Precision / Recall theo tung lop ---
    precision_per_class = precision_score(
        y_test, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_test, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    f1_macro = f1_score(y_test, y_pred, labels=CLASSES, average="macro", zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    per_class_metrics = {}
    per_class_lines = []
    for i, cls in enumerate(CLASSES):
        per_class_metrics[str(cls)] = {
            "precision": round(float(precision_per_class[i]), 4),
            "recall": round(float(recall_per_class[i]), 4),
        }
        per_class_lines.append(
            f"  Lop {cls}: precision = {precision_per_class[i]:.4f}, "
            f"recall = {recall_per_class[i]:.4f}"
        )

    # --- In ra log pipeline ---
    print("=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(cm_text)
    print()
    print("=" * 50)
    print("PRECISION / RECALL THEO TUNG LOP")
    print("=" * 50)
    for line in per_class_lines:
        print(line)
    print(f"\nAccuracy: {acc:.4f} | F1 (macro): {f1_macro:.4f}")

    # --- Ghi outputs/report.txt ---
    report_lines = [
        "=" * 50,
        "BAO CAO HIEU SUAT MO HINH",
        "=" * 50,
        "",
        f"Accuracy: {acc:.4f}",
        f"F1-score (macro): {f1_macro:.4f}",
        "",
        "Confusion Matrix:",
        cm_text,
        "",
        "Precision / Recall theo tung lop:",
        *per_class_lines,
        "",
    ]

    distribution = load_distribution()
    if distribution:
        report_lines.append("Phan phoi nhan trong tap train:")
        for cls, info in distribution.get("class_distribution", {}).items():
            report_lines.append(
                f"  Lop {cls}: {info['count']} mau ({info['ratio'] * 100:.2f}%)"
            )
        if distribution.get("is_imbalanced"):
            report_lines.append("  CANH BAO: Du lieu mat can bang giua cac lop!")
        report_lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nDa ghi bao cao vao {REPORT_PATH}")

    # --- Gop tat ca vao outputs/metrics.json (khong ghi de metrics goc tu train.py) ---
    metrics = load_existing_metrics()
    metrics["report_accuracy"] = round(float(acc), 4)
    metrics["report_f1_score"] = round(float(f1_macro), 4)
    metrics["precision_recall_per_class"] = per_class_metrics
    if distribution:
        metrics["class_distribution"] = distribution.get("class_distribution", {})
        metrics["is_imbalanced"] = distribution.get("is_imbalanced", False)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Da cap nhat {METRICS_PATH} voi report_accuracy, report_f1_score, precision/recall, class_distribution")


if __name__ == "__main__":
    main()
