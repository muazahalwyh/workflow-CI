import pandas as pd
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

mlflow.set_tracking_uri("https://dagshub.com/muazahalwyh/my-first-repo.mlflow")
mlflow.set_experiment("Experiment Customer Churn")

# Load data
data = pd.read_csv("Dataset/data_bersih_preprocessing.csv")
X = data.drop("Churn Label", axis=1)
y = data["Churn Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_example = X_train.iloc[0:5]

# Range hyperparameter
n_estimators_range = np.linspace(100, 1000, 5, dtype=int)
max_depth_range = np.linspace(5, 50, 5, dtype=int)

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"Tuning_{n_estimators}_{max_depth}"):
            mlflow.autolog()  # otomatis log param, metric, model

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            # log metrik akurasi tambahan
            mlflow.log_metric("accuracy_manual", acc)

            # Simpan model terbaik
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="best_model",
                    input_example=input_example
                )

print("Tuning selesai.")
print(f"Model terbaik: {best_params}, Akurasi: {best_accuracy:.4f}")
