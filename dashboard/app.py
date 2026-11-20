"""Customer Churn Retention Dashboard (Streamlit).
Run from the project folder:  streamlit run dashboard/app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parents[1] / "outputs"
NAVY, TEAL, RED = "#1f3a5f", "#2a7f8e", "#c0392b"
BANDS = ["0-12", "13-24", "25-48", "49-72"]

st.set_page_config(page_title="Customer Churn Retention Dashboard", layout="wide")
st.markdown("""
<style>
.kpi{background:#fff;border:1px solid #e3e7ee;border-radius:10px;padding:14px 8px;text-align:center}
.kpi .lbl{color:#7a869a;font-size:14px}
.kpi .val{font-size:32px;font-weight:700;color:#1f3a5f}
.kpi .val.red{color:#c0392b}
h1,h2,h3{color:#1f3a5f}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    pred = pd.read_csv(OUT / "predictions.csv")
    pred["tenure_band"] = pd.cut(pred["tenure"], [-1, 12, 24, 48, 72], labels=BANDS)
    return (pred, pd.read_csv(OUT / "shap_importance.csv"), pd.read_csv(OUT / "metrics.csv"))


if not (OUT / "predictions.csv").exists():
    st.error("predictions.csv not found. Run train.py and analyze.py first (see README).")
    st.stop()
pred, shap_imp, metrics = load()

# ---------------- sidebar filters (apply to every screen) ----------------
st.sidebar.header("Filters")
contract = st.sidebar.multiselect("Contract", sorted(pred["Contract"].unique()),
                                  default=sorted(pred["Contract"].unique()))
t_lo, t_hi = st.sidebar.slider("Tenure range (months)", 0, 72, (0, 72))
internet = st.sidebar.multiselect("Internet service", sorted(pred["InternetService"].unique()),
                                  default=sorted(pred["InternetService"].unique()))
payment = st.sidebar.multiselect("Payment method", sorted(pred["PaymentMethod"].unique()),
                                 default=sorted(pred["PaymentMethod"].unique()))
risk = st.sidebar.multiselect("Risk group", ["Low", "Medium", "High"],
                              default=["Low", "Medium", "High"])

f = pred[pred["Contract"].isin(contract) & pred["tenure"].between(t_lo, t_hi)
         & pred["InternetService"].isin(internet) & pred["PaymentMethod"].isin(payment)
         & pred["risk_group"].isin(risk)]

st.title("Customer Churn Retention Dashboard")
applied = []
if len(contract) < pred["Contract"].nunique():
    applied.append("Contract = " + ", ".join(contract))
if len(internet) < pred["InternetService"].nunique():
    applied.append("Internet = " + ", ".join(internet))
st.caption(("Filter applied: " + " | ".join(applied) if applied else "All customers (test set)")
           + f"  (n={len(f):,})")
if f.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# ---------------- KPI cards ----------------
high = f[f["risk_group"] == "High"]
kpis = [("Churn Rate", f"{100 * f['actual'].mean():.1f}%", ""),
        ("Customers", f"{len(f):,}", ""),
        ("High-Risk", f"{len(high):,}", "red"),
        ("Revenue at Risk", f"${high['revenue_at_risk'].sum():,.0f}/mo", "red")]
for col, (label, value, cls) in zip(st.columns(4), kpis):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div>'
                 f'<div class="val {cls}">{value}</div></div>', unsafe_allow_html=True)
st.write("")


def style(ax, title):
    ax.set_title(title, fontsize=10, fontweight="bold", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


# ---------------- charts ----------------
c1, c2, c3 = st.columns(3)
with c1:
    by_c = (100 * f.groupby("Contract")["actual"].mean()).reindex(
        [c for c in ["Month-to-month", "One year", "Two year"] if c in f["Contract"].unique()])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(by_c.index, by_c.values, color=NAVY)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8)
    style(ax, "Churn by Contract (%)")
    st.pyplot(fig)
with c2:
    by_t = 100 * f.groupby("tenure_band", observed=True)["actual"].mean()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(by_t.index.astype(str), by_t.values, marker="o", color=TEAL)
    style(ax, "Churn by Tenure (months, %)")
    st.pyplot(fig)
with c3:
    top = shap_imp.head(6).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.barh(top["feature"], top["mean_abs_shap"], color=RED)
    style(ax, "Top Churn Drivers (SHAP)")
    st.pyplot(fig)

# ---------------- at-risk list ----------------
st.subheader("At-Risk Customer List (top 8 by churn probability)")
cols = ["customer_id", "churn_probability", "risk_group", "Contract", "tenure",
        "MonthlyCharges", "main_reasons"]
ranked = f.sort_values("churn_probability", ascending=False)
st.dataframe(ranked[cols].head(8).rename(columns={
    "customer_id": "Customer ID", "churn_probability": "Churn prob.", "risk_group": "Risk",
    "tenure": "Tenure", "MonthlyCharges": "Monthly $", "main_reasons": "Main reasons"}),
    hide_index=True, width="stretch")

d1, d2 = st.columns(2)
d1.download_button("Export High-risk list (CSV)", high.sort_values("churn_probability",
                   ascending=False)[cols].to_csv(index=False), "high_risk_customers.csv", "text/csv")
d2.download_button("Export full filtered list (CSV)", ranked[cols].to_csv(index=False),
                   "at_risk_customers.csv", "text/csv")

# ---------------- extra screens (Annex A-1 menu flow) ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Full list", "Risk groups", "Model comparison", "Recommendations"])
with tab1:
    st.dataframe(ranked[cols], hide_index=True, width="stretch", height=380)
with tab2:
    g = f.groupby("risk_group").agg(customers=("actual", "size"), churners=("actual", "sum"))
    g["actual churn %"] = (100 * g["churners"] / g["customers"]).round(1)
    g = g.reindex([r for r in ["Low", "Medium", "High"] if r in g.index])
    colours = {"Low": "#2e8b57", "Medium": "#e0a030", "High": RED}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].bar(g.index, g["customers"], color=[colours[i] for i in g.index])
    style(axes[0], "Customers per risk group")
    axes[1].bar(g.index, g["actual churn %"], color=[colours[i] for i in g.index])
    style(axes[1], "Actual churn rate (%)")
    st.pyplot(fig)
    st.dataframe(g, width="stretch")
with tab3:
    show = metrics[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(4)
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Test set, threshold 0.5. Accuracy alone is misleading: the majority-class "
               "baseline reaches ~73.5% accuracy but finds none of the churners.")
with tab4:
    st.table(pd.DataFrame([
        ["High (0.60 and above)", "Month-to-month contract", "Priority call with discounted 1- or 2-year contract offer"],
        ["High", "Fibre-optic internet or no technical support", "Service-quality check; free trial of tech support / online security"],
        ["High", "Pays by electronic cheque", "Incentive to switch to automatic bank/card payment"],
        ["Medium (0.30 to 0.59)", "Any", "Low-cost loyalty message or add-on bundle reminder"],
        ["Low (below 0.30)", "Any", "No retention offer; normal communication"]],
        columns=["Risk group", "Customer profile", "Suggested action"]))
