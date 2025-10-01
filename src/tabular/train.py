import argparse, json, joblib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.tabular.dataset import load_uci_csv
from src.common.paths import TABULAR_MODELS

# --- feature schema (Cleveland) ---
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEG   = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# --- args ---
ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_iter", type=int, default=20, help="RandomizedSearchCV iterations per model")
args = ap.parse_args()

# --- data ---
df = load_uci_csv(args.csv)
X = df.drop(columns=["target"])
y = df["target"].values

# --- preprocessing pipeline (leak-proof) ---
pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore"))]), CATEG),
    ],
    remainder="drop",
)

# --- models & search spaces (LogReg, RF, SVM) ---
models = [
    (
        "logreg",
        LogisticRegression(max_iter=2000),
        {
            "model__C": [0.05, 0.1, 0.5, 1, 2, 5, 10],
            "model__solver": ["lbfgs", "liblinear"],  # liblinear handles small data & L2 well
        },
    ),
    (
        "rf",
        RandomForestClassifier(),
        {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [None, 4, 6, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
    ),
    (
        "svm",
        SVC(probability=True),  # probability=True enables ROC-AUC + calibrated outputs
        {
            "model__C": [0.1, 1, 5, 10],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],  # used when kernel="rbf"
        },
    ),
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

best_est = None
best_auc = -1.0
best_name = None
results = []

for name, model, grid in models:
    pipe = Pipeline([("pre", pre), ("model", model)])
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        n_iter=args.n_iter,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        random_state=args.seed,
        verbose=1,
    )
    search.fit(X, y)
    auc = float(search.best_score_)
    results.append({"model": name, "cv_roc_auc": auc})
    if auc > best_auc:
        best_auc = auc
        best_est = search.best_estimator_
        best_name = name

TABULAR_MODELS.mkdir(parents=True, exist_ok=True)
joblib.dump(best_est, TABULAR_MODELS / "best_model.joblib")

(Path(TABULAR_MODELS / "metrics_tabular.json")).write_text(
    json.dumps({"cv_results": results, "best_model": best_name, "best_auc": best_auc}, indent=2)
)

print("Saved:", TABULAR_MODELS / "best_model.joblib")
print(json.dumps(results, indent=2))
