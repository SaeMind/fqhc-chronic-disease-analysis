#!/usr/bin/env python3
"""
Reproduce the diabetes classifier's test-set metrics from the committed model and feature extract.

    python scripts/report_diabetes_classification.py [--repo PATH]

Loads models/diabetes_classifier.pkl and data/processed/hc2024_model_features.parquet, rebuilds the feature matrix the
way notebooks/03_predictive_models.ipynb does, applies the notebook's split (test_size 0.25, random_state 42), and writes
outputs/diabetes_classification_report.json: recall, precision, F1, AUC-ROC, the confusion matrix, the split parameters
and the sha256 of the model file. Nothing is trained and no data file is modified.

The model file is a pickle: only run this on a copy of the repository you trust. The report contains no timestamp, so
re-running it on the same files and library versions rewrites a byte-identical file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

MODEL = "models/diabetes_classifier.pkl"
DATA = "data/processed/hc2024_model_features.parquet"
OUT = "outputs/diabetes_classification_report.json"
SPLIT = {"test_size": 0.25, "random_state": 42}              # notebooks/03_predictive_models.ipynb
TARGETS = ["diabetes", "hypertension", "mental_health"]       # the notebook splits all three targets together
TARGET = "diabetes"
BASE_FEATURES = ["AGE", "sex_binary", "num_diagnoses", "is_weekend"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_in_git(repo: Path, rel: str):
    try:
        return subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", rel], capture_output=True).returncode == 0
    except OSError:
        return None


def build_features(df):
    """Same construction as the notebook: base columns, race and season dummies (drop_first), missing values -> column mean."""
    import pandas as pd
    race = pd.get_dummies(df["race_simple"], prefix="race", drop_first=True)
    season = pd.get_dummies(df["season"], prefix="season", drop_first=True)
    X = pd.concat([df[BASE_FEATURES], race, season], axis=1)
    return X.fillna(X.mean())


def reproduce(repo: Path) -> dict:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(repo / DATA)
    X = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, df[TARGETS], **SPLIT)
    model = joblib.load(repo / MODEL)
    cols = list(getattr(model, "feature_names_in_", X_test.columns))
    pred = model.predict(X_test[cols])
    proba = model.predict_proba(X_test[cols])[:, 1]
    y = y_test[TARGET]
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "report": "diabetes classifier, test split of the committed NAMCS HC 2024 feature extract",
        "target": TARGET,
        "model": {"path": MODEL, "sha256": sha256(repo / MODEL), "tracked_in_git": tracked_in_git(repo, MODEL),
                  "class": type(model).__name__, "class_weight": str(getattr(model, "class_weight", None)), "feature_names_in": cols},
        "data": {"path": DATA, "sha256": sha256(repo / DATA), "tracked_in_git": tracked_in_git(repo, DATA), "rows": int(len(df))},
        "split": {**SPLIT, "stratify": None, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                  "source": "notebooks/03_predictive_models.ipynb"},
        "decision_threshold": "model.predict (probability 0.5)",
        "metrics": {"recall": float(recall_score(y, pred)), "precision": float(precision_score(y, pred)),
                    "f1": float(f1_score(y, pred)), "auc_roc": float(roc_auc_score(y, proba))},
        "confusion_matrix": {"labels": [0, 1], "layout": "rows = actual, columns = predicted",
                             "matrix": [[tn, fp], [fn, tp]], "tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "positives_in_test": int(y.sum()),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent), help="repository root")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    report = reproduce(repo)
    out = repo / OUT
    out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    m = report["metrics"]
    print(f"recall={m['recall']:.4f} precision={m['precision']:.4f} f1={m['f1']:.4f} auc_roc={m['auc_roc']:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
