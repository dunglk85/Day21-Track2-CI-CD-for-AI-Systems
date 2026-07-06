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
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()


def get_eval_threshold(params):
    return params.get("eval_threshold", 0.70)


def setup_mlflow():
    client = mlflow.tracking.MlflowClient()
    artifact_root = "file:///" + os.path.abspath("model_artifacts").replace(os.sep, "/")
    exp_name = "wine_quality"
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        client.create_experiment(exp_name, artifact_location=artifact_root)
    mlflow.set_experiment(exp_name)


def load_data(data_path: str, eval_path: str):
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]
    return X_train, y_train, X_eval, y_eval


def build_models(params: dict) -> dict:
    rf_params = params.get("random_forest", {})
    gb_params = params.get("gradient_boosting", {})
    lr_params = params.get("logistic_regression", {})

    all_models = {
        "random_forest": RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth", 10),
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=gb_params.get("n_estimators", 100),
            max_depth=gb_params.get("max_depth", 5),
            learning_rate=gb_params.get("learning_rate", 0.1),
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=lr_params.get("max_iter", 1000), random_state=42
        ),
    }

    model_type = params.get("model_type")
    if model_type:
        if model_type not in all_models:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choices: {list(all_models.keys())}"
            )
        print(f"--- Huan luyen model theo cau hinh: {model_type} ---")
        return {model_type: all_models[model_type]}

    print("--- Bat dau qua trinh tim kiem model tot nhat (tat ca) ---")
    return all_models


def train_and_select(
    models: dict, X_train, y_train, X_eval, y_eval
):
    best_acc = -1.0
    best_model = None
    best_model_name = ""
    best_metrics = {}
    best_run_id = None

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Trial: {name}"):
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)
            acc = accuracy_score(y_eval, preds)
            f1 = f1_score(y_eval, preds, average="weighted")

            mlflow.log_param("model_family", name)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
            print(f"Algorithm: {name:20} | Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_model_name = name
                best_metrics = {"accuracy": acc, "f1_score": f1}
                best_run_id = mlflow.active_run().info.run_id

    print(
        f"\n===> KET QUA: '{best_model_name}' la model tot nhat voi Accuracy: {best_acc:.4f}"
    )
    return best_model, best_model_name, best_metrics, best_run_id


def save_artifacts(model, metrics: dict, run_id: str):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("is_best_model", "true")
        mlflow.sklearn.log_model(model, "best_model")

    model_uri = f"runs:/{run_id}/best_model"
    reg_result = mlflow.register_model(model_uri, "wine_quality_model")
    print(f"REGISTERED_MODEL_VERSION={reg_result.version}")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    X_train, y_train, X_eval, y_eval = load_data(data_path, eval_path)
    setup_mlflow()
    models = build_models(params)
    best_model, _, best_metrics, best_run_id = train_and_select(
        models, X_train, y_train, X_eval, y_eval
    )
    save_artifacts(best_model, best_metrics, best_run_id)
    return best_metrics["accuracy"]


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
