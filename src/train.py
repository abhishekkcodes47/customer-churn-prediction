"""Training: stratified split, SMOTE inside the pipeline, grid search, evaluation."""
import time
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocess import add_features, build_matrix, clean_data

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_churn.csv"
OUT = ROOT / "outputs"


def load_prepared(path=RAW):
    """CSV -> cleaned data frame with features (process 1.0 / 2.0)."""
    return add_features(clean_data(pd.read_csv(path)))


def split_data(X, y, ids):
    """Stratified 80/20 split with fixed seed (process 3.1)."""
    return train_test_split(X, y, ids, test_size=0.2, stratify=y, random_state=SEED)


def apply_smote(X, y):
    """SMOTE on a training set only (process 3.2)."""
    return SMOTE(random_state=SEED).fit_resample(X, y)


def build_pipeline(model, scale=False):
    """SMOTE sits inside the pipeline so it only touches training folds."""
    steps = [("scale", StandardScaler())] if scale else []
    return Pipeline(steps + [("smote", SMOTE(random_state=SEED)), ("clf", model)])


def get_models():
    return {
        "Logistic Regression": (
            build_pipeline(LogisticRegression(max_iter=2000, random_state=SEED), scale=True),
            {"clf__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (
            build_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=1)),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [6, 10]}),
        "XGBoost": (
            build_pipeline(XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1)),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.05, 0.1]}),
    }


def evaluate(y_true, prob, threshold=0.5):
    """Accuracy, precision, recall, F1, ROC-AUC and confusion-matrix counts."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, prob),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def tune_models(X_tr, y_tr):
    """Grid search with 5-fold stratified CV (process 3.3)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fitted = {}
    for name, (pipe, grid) in get_models().items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        scores = cross_validate(clone(gs.best_estimator_), X_tr, y_tr, cv=cv, n_jobs=-1,
                                scoring={"f1": "f1", "roc_auc": "roc_auc", "recall": "recall"})
        fitted[name] = {"search": gs, "time": time.time() - t0, "cv": scores}
    return fitted


def select_best(rows):
    """Best model = highest cross-validated ROC-AUC (training data only, no test peeking)."""
    return max(rows, key=lambda r: r["cv_roc_auc"] if r["model"] != "Majority-class baseline" else -1)["model"]


def main():
    OUT.mkdir(exist_ok=True)
    df = load_prepared()
    X, y, ids = build_matrix(df)
    X_tr, X_te, y_tr, y_te, id_tr, id_te = split_data(X, y, ids)
    print(f"train {X_tr.shape}  test {X_te.shape}  churn rate {y_te.mean():.4f}\n")

    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    rows = [{"model": "Majority-class baseline",
             **evaluate(y_te, base.predict_proba(X_te)[:, 1] * 0)}]
    fitted = tune_models(X_tr, y_tr)
    models = {}
    for name, res in fitted.items():
        best = res["search"].best_estimator_
        models[name] = best
        m = evaluate(y_te, best.predict_proba(X_te)[:, 1])
        cv = res["cv"]
        row = {"model": name, **m,
               "cv_f1": cv["test_f1"].mean(), "cv_f1_std": cv["test_f1"].std(),
               "cv_roc_auc": cv["test_roc_auc"].mean(), "cv_roc_auc_std": cv["test_roc_auc"].std(),
               "cv_recall": cv["test_recall"].mean(), "cv_recall_std": cv["test_recall"].std(),
               "best_params": str(res["search"].best_params_), "time_s": round(res["time"], 1)}
        rows.append(row)
        print(f"{name}\n  accuracy={m['accuracy']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f}\n  f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f} "
              f"cv_f1={row['cv_f1']:.4f}\n  best_params={row['best_params']} time={row['time_s']}s")
    best_name = select_best(rows)
    for r in rows:
        r["selected"] = r["model"] == best_name
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    joblib.dump({"models": models, "best": best_name, "feature_names": list(X.columns),
                 "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
                 "id_te": id_te}, OUT / "models.joblib")
    print(f"\nBEST (highest cross-validated ROC-AUC): {best_name}")


if __name__ == "__main__":
    main()
