"""
modelling.py (Workflow-CI / MLProject)

Script training yang dijalankan otomatis oleh MLflow Project di dalam
pipeline CI (GitHub Actions). Menggunakan Random Forest dengan MLflow
autolog, tracking disimpan secara lokal ke folder ./mlruns pada runner
GitHub Actions, kemudian artefaknya disimpan/di-commit oleh workflow CI.
"""

import argparse
import os

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TARGET_COLUMN = "income"


def load_train_test(data_dir: str):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def main(data_dir: str, n_estimators: int, max_depth):
    # Catatan: script ini dijalankan oleh `mlflow run`, sehingga run MLflow
    # sudah otomatis aktif (tidak perlu memanggil mlflow.start_run() lagi).
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_train_test(data_dir)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_roc_auc", auc)

    print(f"[INFO] Test Accuracy: {acc:.4f}")
    print(f"[INFO] Test F1-score: {f1:.4f}")
    print(f"[INFO] Test ROC-AUC : {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="adult_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=16)
    args = parser.parse_args()

    main(args.data_dir, args.n_estimators, args.max_depth)
