"""25 automated tests: unit (tc01-tc10), module (tc11-tc12), integration (it01-it07),
system (st01-st06). Self-contained: they train a fixed XGBoost model in memory."""
import time

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from analyze import build_predictions, risk_groups, score_customers, shap_values_for
from preprocess import add_features, assign_risk, build_matrix, clean_data, validate_data
from train import RAW, apply_smote, build_pipeline, evaluate, split_data

PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
              random_state=42, eval_metric="logloss", n_jobs=1)


def mini():  # TD4: blank TotalCharges, zero tenure, duplicated customerID 'B'
    return pd.DataFrame({"customerID": ["A", "B", "B"], "tenure": [0, 10, 10],
                         "TotalCharges": [" ", "500.5", "500.5"]})


def train_model(X_tr, y_tr):
    return build_pipeline(XGBClassifier(**PARAMS)).fit(X_tr, y_tr)


@pytest.fixture(scope="session")
def raw():
    return pd.read_csv(RAW)


@pytest.fixture(scope="session")
def data(raw):
    df = add_features(clean_data(raw))
    X, y, ids = build_matrix(df)
    X_tr, X_te, y_tr, y_te, id_tr, id_te = split_data(X, y, ids)
    return dict(df=df, X=X, y=y, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te)


@pytest.fixture(scope="session")
def model(data):
    return train_model(data["X_tr"], data["y_tr"])


@pytest.fixture(scope="session")
def preds(data, model):
    X_te = data["X_te"]
    sv = shap_values_for(model, X_te)
    return build_predictions(model, X_te, data["df"].loc[X_te.index], sv)


# ---------------- unit tests ----------------
def test_tc01_blank_total_charges_filled_with_zero():
    assert clean_data(mini()).loc[0, "TotalCharges"] == 0


def test_tc02_total_charges_is_numeric():
    assert pd.api.types.is_numeric_dtype(clean_data(mini())["TotalCharges"])


def test_tc03_duplicates_removed():
    assert len(clean_data(mini())) == 2


def test_tc04_avg_charge_zero_tenure_no_division_error():
    out = add_features(clean_data(mini()))
    assert out["avg_charge_per_month"].notna().all()


def test_tc05_avg_charge_value():
    df = pd.DataFrame({"TotalCharges": [500.5], "tenure": [10]})
    assert add_features(df).loc[0, "avg_charge_per_month"] == pytest.approx(50.05)


def test_tc06_risk_low():
    assert assign_risk(0.10) == "Low"


def test_tc07_risk_medium():
    assert assign_risk(0.45) == "Medium"


def test_tc08_risk_high():
    assert assign_risk(0.85) == "High"


def test_tc09_risk_boundaries():
    assert assign_risk(0.30) == "Medium" and assign_risk(0.60) == "High"
    assert assign_risk(0.29) == "Low" and assign_risk(0.59) == "Medium"
    assert assign_risk(0.0) == "Low" and assign_risk(1.0) == "High"


def test_tc10_invalid_probability_rejected():
    for bad in (1.2, -0.01, 1.01):
        with pytest.raises(ValueError):
            assign_risk(bad)


# ---------------- module tests ----------------
def test_tc11_real_dataset_passes_validation(raw):
    df = clean_data(raw)
    assert validate_data(df).sum() == 0
    assert validate_data(df.drop(columns="Churn")).sum() == 0  # new file without label


def test_tc12_invalid_contract_is_flagged(raw):
    df = clean_data(raw)
    df.loc[0, "Contract"] = "Weekly"
    assert validate_data(df).sum() == 1


# ---------------- integration tests ----------------
def test_it01_pipeline_produces_model_ready_matrix(data):
    X = data["X"]
    assert X.shape == (7043, 31) and not X.isna().any().any()


def test_it02_split_is_stratified(data):
    assert abs(data["y_tr"].mean() - data["y_te"].mean()) < 0.01


def test_it03_no_customer_in_both_sets(data):
    assert set(data["X_tr"].index).isdisjoint(data["X_te"].index)


def test_it04_smote_balances_training_only(data):
    _, y_bal = apply_smote(data["X_tr"], data["y_tr"])
    assert y_bal.value_counts().nunique() == 1
    assert len(data["X_te"]) == 1409


def test_it05_probabilities_between_0_and_1(data, model):
    p = score_customers(model, data["X_te"])
    assert ((p >= 0) & (p <= 1)).all()


def test_it06_risk_groups_valid(preds):
    assert set(preds["risk_group"]) <= {"Low", "Medium", "High"}


def test_it07_prediction_file_has_required_columns(preds):
    need = {"customer_id", "churn_probability", "risk_group", "main_reasons", "revenue_at_risk"}
    assert need <= set(preds.columns)


# ---------------- system tests ----------------
def test_st01_results_reproducible_with_fixed_seed(data, model):
    again = train_model(data["X_tr"], data["y_tr"])
    assert np.allclose(score_customers(model, data["X_te"]), score_customers(again, data["X_te"]))


def test_st02_full_dataset_scored_within_5_seconds(data, model):
    t0 = time.time()
    score_customers(model, data["X"])
    assert time.time() - t0 < 5


def test_st03_roc_auc_at_least_0_80(data, model):
    assert evaluate(data["y_te"], score_customers(model, data["X_te"]))["roc_auc"] >= 0.80


def test_st04_recall_beats_majority_baseline(data, model):
    base = evaluate(data["y_te"], np.zeros(len(data["y_te"])))["recall"]
    assert evaluate(data["y_te"], score_customers(model, data["X_te"]))["recall"] > base


def test_st05_risk_groups_rank_churn_correctly(preds):
    rate = preds.groupby("risk_group")["actual"].mean()
    assert rate["Low"] < rate["Medium"] < rate["High"]


def test_st06_risk_assignment_matches_probability(preds):
    assert risk_groups(preds["churn_probability"]) == list(preds["risk_group"])
