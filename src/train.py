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

def get_eval_threshold(params):
    return params.get("eval_threshold", 0.70)



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

    # Dat artifact location de tranh loi proxy mlflow-artifacts voi sqlite:///
    client = mlflow.tracking.MlflowClient()
    artifact_root = "file:///" + os.path.abspath("model_artifacts").replace(os.sep, "/")
    exp_name = "wine_quality"
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        client.create_experiment(exp_name, artifact_location=artifact_root)
    mlflow.set_experiment(exp_name)
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 2. Dinh nghia danh sach cac model muon thu nghiem
    rf_params = params.get("random_forest", {})
    gb_params = params.get("gradient_boosting", {})
    lr_params = params.get("logistic_regression", {})

    models_to_try = {
        "random_forest": RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100), 
            max_depth=rf_params.get("max_depth", 10), 
            random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=gb_params.get("n_estimators", 100), 
            max_depth=gb_params.get("max_depth", 5), 
            learning_rate=gb_params.get("learning_rate", 0.1),
            random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=lr_params.get("max_iter", 1000), 
            random_state=42
        )
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
                best_run_id = mlflow.active_run().info.run_id

    print(f"\n===> KET QUA: '{best_model_name}' la model tot nhat voi Accuracy: {best_acc:.4f}")

    # 3. Luu "Nha vo dich"
    # Luu metrics
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(best_metrics, f)

    # Luu model binary
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    
    # Ghi nhan model tot nhat vao dung run cua no
    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("is_best_model", "true")
        mlflow.sklearn.log_model(best_model, "best_model")

    return best_acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
