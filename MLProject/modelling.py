import pandas as pd
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import sys
import argparse

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # file_path = sys.argv[3] if len (sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_bersih_preprocessing.csv")
    # Ambil argumen dari CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=505)
    parser.add_argument("--max_depth", type=int, default=35)
    parser.add_argument("--dataset", type=str, default="data_bersih_preprocessing.csv")
    args = parser.parse_args()

    n_estimators = args.n_estimators
    max_depth = args.max_depth
    file_path = args.dataset
    
    data = pd.read_csv(file_path)
  
    # Split fitur dan target
    X = data.drop("Churn Label", axis=1)
    y = data["Churn Label"]

    # Split data train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ambil contoh input untuk log model (harus DataFrame)
    input_example = X_train.iloc[0:5]

    with mlflow.start_run():
        # # Set parameter model
        # n_estimators = 505
        # max_depth = 37
        
        # Log parameter manual
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # model Random Forest
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        # Train model
        model.fit(X_train, y_train)

        # Prediksi untuk evaluasi
        y_pred = model.predict(X_test)

        # Hitungan metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrik manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Simpan model dengan input_example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print("Model training dan logging selesai.")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")

# with mlflow.start_run():
#     # Log parameters
#     n_estimators = 505
#     max_depth = 37
#     mlflow.autolog()
#     # Train model
#     model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#     mlflow.sklearn.log_model(
#         sk_model=model,
#         artifact_path="model",
#         input_example=input_example
#     )
#     model.fit(X_train, y_train)
#     # Log metrics
#     accuracy = model.score(X_test, y_test)
#     mlflow.log_metric("accuracy", accuracy)
    
# print(f"Training selesai dengan akurasi: {accuracy:.4f}")