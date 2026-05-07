import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.stats import ks_2samp

load_dotenv()  # Tai cac bien moi truong tu file .env

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Tu dong huan luyen nhieu thuat toan va chon ra cai tot nhat.
    """

    # 1. Doc du lieu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 2. Dinh nghia danh sach cac model muon thu nghiem
    models_to_try = {
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100), 
            max_depth=params.get("max_depth", 5), 
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42
        ),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    best_acc = -1
    best_model = None
    best_model_name = ""
    best_metrics = {}

    print("--- Bat dau qua trinh tim kiem model tot nhat ---")

    for name, model in models_to_try.items():
        with mlflow.start_run(run_name=f"Trial: {name}"):
            # Huan luyen
            model.fit(X_train, y_train)
            
            # Danh gia
            preds = model.predict(X_eval)
            acc = accuracy_score(y_eval, preds)
            f1 = f1_score(y_eval, preds, average="weighted")
            
            # Log len MLflow
            mlflow.log_param("model_family", name)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
            
            print(f"Algorithm: {name:20} | Accuracy: {acc:.4f}")

            # Cap nhat model tot nhat
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_model_name = name
                best_metrics = {"accuracy": acc, "f1_score": f1}

    print(f"\n===> KET QUA: '{best_model_name}' la model tot nhat voi Accuracy: {best_acc:.4f}")

    # 3. Luu "Nha vo dich"
    # Luu metrics
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(best_metrics, f)

    # --- MO RONG: Tao bao cao chi tiet (Bonus 3) ---
    best_preds = best_model.predict(X_eval)
    report = classification_report(y_eval, best_preds)
    matrix = confusion_matrix(y_eval, best_preds)
    
    with open("outputs/report.txt", "w") as f:
        f.write("=== KET QUA HUAN LUYEN CHI TIET ===\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy: {best_acc:.4f}\n\n")
        f.write("--- Classification Report ---\n")
        f.write(report)
        f.write("\n--- Confusion Matrix ---\n")
        f.write(str(matrix))
    
    print(f"Da tao bao cao chi tiet tai: outputs/report.txt")
    # ---------------------------------------------

    # Luu model binary
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    
    # --- MO RONG: Kiem tra Data Drift (Bonus 5) ---
    drift_report = []
    # Chia doi tap train de so sanh Phase 1 va Phase 2
    mid_point = len(df_train) // 2
    df_p1 = df_train.iloc[:mid_point]
    df_p2 = df_train.iloc[mid_point:]
    
    for col in X_train.columns:
        stat, p_value = ks_2samp(df_p1[col], df_p2[col])
        if p_value < 0.05:
            drift_report.append(f"Cảnh báo: Phát hiện Drift tại cột '{col}' (p-value: {p_value:.4f})")
    
    with open("outputs/report.txt", "a") as f:
        f.write("\n--- Data Drift Analysis ---\n")
        if not drift_report:
            f.write("Khong phat hien lech lac du lieu dang ke.\n")
        else:
            for line in drift_report:
                f.write(line + "\n")
                print(line)
    # ---------------------------------------------

    # Log model tot nhat vao mot run tong hop
    with mlflow.start_run(run_name=f"Best Model: {best_model_name}"):
        mlflow.log_param("best_algorithm", best_model_name)
        if drift_report:
            mlflow.log_param("data_drift_detected", "True")
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model, "best_model")

    return best_acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
