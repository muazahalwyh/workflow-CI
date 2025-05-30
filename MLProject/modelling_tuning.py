import pandas as pd
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
import os
import joblib

def main(dataset_dir: str):
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load data
    X_train = pd.read_csv(os.path.join(dataset_dir, "X_train_resampled.csv"))
    y_train = pd.read_csv(os.path.join(dataset_dir, "y_train_resampled.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(dataset_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(dataset_dir, "y_test.csv")).squeeze()

    # Set MLflow experiment
    mlflow.set_experiment("Customer Churn Hyperparameter Tuning")

    # Parameter grid untuk tuning
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluasi di test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log hasil tuning ke MLflow
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)

        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        joblib.dump(best_model, "best_model.pkl")

    print(f"Best params: {best_params}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Model tuning selesai dan disimpan sebagai best_model.pkl")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="Dataset")
    args = parser.parse_args()

    main(args.dataset_dir)
