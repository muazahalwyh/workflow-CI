import pandas as pd
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
import argparse
import joblib
import os

def main(n_estimators: int, max_depth: int, dataset_dir: str):
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load dataset dari folder Dataset/
    X_train = pd.read_csv(os.path.join(dataset_dir, "Dataset/X_train_resampled.csv"))
    y_train = pd.read_csv(os.path.join(dataset_dir, "Dataset/y_train_resampled.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(dataset_dir, "DatasetX_test.csv"))
    y_test = pd.read_csv(os.path.join(dataset_dir, "Datasety_test.csv")).squeeze()

    # Logging MLflow
    mlflow.set_experiment("Customer Churn")

    input_example = X_train.iloc[:5]

    with mlflow.start_run():
        mlflow.autolog()

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)
        joblib.dump(model, "model.pkl")

        print("Training selesai. Model disimpan sebagai model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--dataset_dir", type=str, default="Dataset")
    args = parser.parse_args()

    main(args.n_estimators, args.max_depth, args.dataset_dir)
