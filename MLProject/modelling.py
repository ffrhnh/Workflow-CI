"""
modelling.py
============
Script training model untuk MLflow Project (Workflow CI).
Dataset : titanic_preprocessing/titanic_train.csv & titanic_test.csv
Model   : Random Forest Classifier
Logging : Manual logging (params + metrics + artefak)
"""

import os
import json
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Titanic_Classification_CI"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "titanic_preprocessing", "titanic_train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "titanic_preprocessing", "titanic_test.csv")

# Set experiment SEBELUM start_run
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Load Data ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

TARGET  = "Survived"
X_train = train_df.drop(columns=[TARGET]).astype(float)
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET]).astype(float)
y_test  = test_df[TARGET]

FEATURE_NAMES = list(X_train.columns)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── Hyperparameter Tuning ─────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators"     : [100, 200],
    "max_depth"        : [6, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf" : [1, 2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid = PARAM_GRID,
    scoring    = "f1",
    cv         = cv,
    n_jobs     = -1,
    verbose    = 1,
    refit      = True,
    return_train_score=True,
)

print("Memulai GridSearchCV ...")
grid_search.fit(X_train, y_train)
best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best params : {best_params}")
print(f"Best CV F1  : {grid_search.best_score_:.4f}")

# ── Evaluasi ──────────────────────────────────────────────────────────────────
y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test, y_pred, zero_division=0)
f1      = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_prob)
logloss = log_loss(y_test, y_pred_prob)

print(f"\nAccuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1        : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"Log Loss  : {logloss:.4f}")


# ── Helper Plot ───────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Meninggal", "Selamat"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix - Best Model")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[idx], color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title("Feature Importances - Titanic")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_cv_results(gs, save_path):
    results = pd.DataFrame(gs.cv_results_)
    means   = results["mean_test_score"]
    stds    = results["std_test_score"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(range(len(means)), means, yerr=stds, fmt="o-", markersize=4,
                color="darkorange", ecolor="gray", capsize=3)
    ax.set_title("GridSearchCV - CV F1 per Combination")
    ax.set_xlabel("Combination Index")
    ax.set_ylabel("Mean CV F1")
    ax.axhline(gs.best_score_, color="red", linestyle="--",
               label=f"Best: {gs.best_score_:.4f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ── MLflow Logging ────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="RandomForest_CI") as run:

    print(f"\n[MLflow] Run ID     : {run.info.run_id}")
    print(f"[MLflow] Experiment : {EXPERIMENT_NAME}")

    # Log params
    mlflow.log_params(best_params)
    mlflow.log_param("cv_folds",     5)
    mlflow.log_param("scoring",      "f1")
    mlflow.log_param("test_size",    0.20)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("accuracy_score",   acc)
    mlflow.log_metric("precision_score",  prec)
    mlflow.log_metric("recall_score",     rec)
    mlflow.log_metric("f1_score",         f1)
    mlflow.log_metric("roc_auc_score",    roc_auc)
    mlflow.log_metric("log_loss",         logloss)
    mlflow.log_metric("best_cv_f1",       grid_search.best_score_)
    mlflow.log_metric("training_samples", len(X_train))
    mlflow.log_metric("test_samples",     len(X_test))

    # Log artefak
    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        fi_path = os.path.join(tmpdir, "feature_importance.png")
        cv_path = os.path.join(tmpdir, "cv_results.png")
        cr_path = os.path.join(tmpdir, "classification_report.txt")
        bp_path = os.path.join(tmpdir, "best_params.json")

        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        plot_feature_importance(best_model, FEATURE_NAMES, fi_path)
        mlflow.log_artifact(fi_path, artifact_path="plots")

        plot_cv_results(grid_search, cv_path)
        mlflow.log_artifact(cv_path, artifact_path="plots")

        with open(cr_path, "w") as f:
            f.write(classification_report(
                y_test, y_pred,
                target_names=["Meninggal", "Selamat"]
            ))
        mlflow.log_artifact(cr_path, artifact_path="reports")

        with open(bp_path, "w") as f:
            json.dump(best_params, f, indent=2)
        mlflow.log_artifact(bp_path, artifact_path="params")

    # Log model
    signature = infer_signature(X_train, best_model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model      = best_model,
        artifact_path = "model",
        signature     = signature,
        input_example = X_train.head(5),
    )

    print("[MLflow] Semua logging selesai!")
    print(f"[DONE] Training selesai.")
