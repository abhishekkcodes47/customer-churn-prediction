"""Insight layer: scoring, risk groups, SHAP explanations, predictions.csv."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from preprocess import assign_risk

from train import OUT, RAW, load_prepared

ADDON_TEXT = {"OnlineSecurity": "online security", "OnlineBackup": "online backup",
              "DeviceProtection": "device protection", "TechSupport": "tech support"}


def score_customers(model, X):
    return model.predict_proba(X)[:, 1]


def risk_groups(probs):
    return [assign_risk(float(p)) for p in probs]


def explainer_model(models, best):
    """TreeSHAP needs a tree model; if the best model is linear, explain XGBoost instead."""
    pipe = models[best]
    if isinstance(pipe.named_steps["clf"], LogisticRegression):
        pipe = models["XGBoost"]
    return pipe


def shap_values_for(pipe, X):
    return shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(X)


def reason_text(feature, value, shap_value=None):
    """Translate a raw feature (e.g. 'Contract_Two year' = 0) into plain words."""
    if feature == "tenure":
        return f"tenure only {int(value)} months" if (shap_value or 1) > 0 else f"tenure {int(value)} months"
    if feature == "avg_charge_per_month":
        return f"avg charge ${value:.0f}/month"
    if feature in ("MonthlyCharges", "TotalCharges"):
        return f"{feature} = {value:.0f}"
    if feature == "InternetService_Fiber optic":
        return "Fiber optic internet" if value else "not Fiber optic internet"
    if feature == "InternetService_No":
        return "no internet service" if value else "has internet service"
    if feature == "PaymentMethod_Electronic check":
        return "Electronic check payment" if value else "not Electronic check payment"
    if feature.startswith("Contract_"):
        label = feature.split("_", 1)[1] + " contract"
        return label if value else f"not {label}"
    if feature == "PaperlessBilling_Yes":
        return "Paperless billing" if value else "no paperless billing"
    base, _, cat = feature.partition("_")
    if base in ADDON_TEXT and cat == "Yes":
        return ADDON_TEXT[base] if value else f"no {ADDON_TEXT[base]}"
    if cat == "Yes":
        return f"{base} = Yes" if value else f"{base} = No"
    return f"{feature} = {value:g}"


def shap_reasons(sv, X, top=2):
    """The `top` features that push each customer's score towards churn, in plain text."""
    out = []
    cols = np.array(X.columns)
    for i in range(len(X)):
        idx = np.argsort(-sv[i])[:top]
        out.append("; ".join(reason_text(cols[j], X.iloc[i, j], sv[i, j]) for j in idx))
    return out


def build_predictions(model, X_te, df_te, sv):
    """predictions.csv content (Table A.10), plus the fields the dashboard filters on."""
    probs = score_customers(model, X_te)
    return pd.DataFrame({
        "customer_id": df_te["customerID"].values,
        "churn_probability": probs.round(4),
        "risk_group": risk_groups(probs),
        "main_reasons": shap_reasons(sv, X_te),
        "revenue_at_risk": df_te["MonthlyCharges"].values,
        "Contract": df_te["Contract"].values,
        "tenure": df_te["tenure"].values,
        "InternetService": df_te["InternetService"].values,
        "PaymentMethod": df_te["PaymentMethod"].values,
        "MonthlyCharges": df_te["MonthlyCharges"].values,
        "actual": (df_te["Churn"] == "Yes").astype(int).values})


def main():
    import joblib
    art = joblib.load(OUT / "models.joblib")
    models, best = art["models"], art["best"]
    X_te, y_te = art["X_te"], art["y_te"]
    df = load_prepared(RAW).set_index(pd.RangeIndex(len(load_prepared(RAW))))
    df_te = df.loc[X_te.index]

    sv = shap_values_for(explainer_model(models, best), X_te)
    pred = build_predictions(models[best], X_te, df_te, sv)
    pred.sort_values("churn_probability", ascending=False).to_csv(OUT / "predictions.csv", index=False)
    np.save(OUT / "shap_values.npy", sv)
    (pd.DataFrame({"feature": X_te.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
       .sort_values("mean_abs_shap", ascending=False).to_csv(OUT / "shap_importance.csv", index=False))

    # Table 4.10 - validation of the risk groups
    g = pred.groupby("risk_group").agg(customers=("actual", "size"), churners=("actual", "sum"))
    g["churn_rate_%"] = (100 * g["churners"] / g["customers"]).round(1)
    g = g.reindex(["Low", "Medium", "High"])
    g.to_csv(OUT / "risk_validation.csv")
    print("Risk groups (test set):\n", g, "\n")

    # Table 4.9 - precision / recall at different thresholds
    p, rows = pred["churn_probability"].values, []
    for t in (0.3, 0.4, 0.5, 0.6):
        flag = p >= t
        tp = int((flag & (pred["actual"] == 1)).sum())
        rows.append({"threshold": t, "precision_%": round(100 * tp / max(flag.sum(), 1), 1),
                     "recall_%": round(100 * tp / pred["actual"].sum(), 1)})
    th = pd.DataFrame(rows)
    th.to_csv(OUT / "threshold_table.csv", index=False)
    print("Threshold trade-off:\n", th, "\n")
    high = pred[pred["risk_group"] == "High"]
    print(f"High-risk customers: {len(high)}  revenue at risk: ${high['revenue_at_risk'].sum():,.0f}/month")
    print(f"Saved {len(pred)} predictions to {OUT / 'predictions.csv'}")


if __name__ == "__main__":
    main()
