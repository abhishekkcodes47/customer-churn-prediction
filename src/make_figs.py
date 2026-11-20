"""Charts for the report (Figures 4.1-4.6, A.8). Saved in outputs/figures/."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_curve

from train import OUT, RAW, load_prepared
from analyze import explainer_model

FIG = OUT / "figures"
NAVY, TEAL, RED, GREY = "#1f3a5f", "#2a7f8e", "#c0392b", "#7f8c8d"
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def bar_labels(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                fmt.format(b.get_height()), ha="center", fontsize=7)


def eda_charts(df):
    overall = 100 * (df["Churn"] == "Yes").mean()
    bands = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
    pay = df["PaymentMethod"].str.replace(" (automatic)", "\n(auto)", regex=False).str.replace(" ", "\n", n=1)
    panels = [("Churn by contract type", df["Contract"], NAVY),
              ("Churn by tenure band (months)", bands, TEAL),
              ("Churn by payment method", pay, NAVY),
              ("Churn by internet service", df["InternetService"], TEAL)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (title, key, colour) in zip(axes.ravel(), panels):
        rate = 100 * (df["Churn"] == "Yes").groupby(key, observed=True).mean()
        if title.startswith("Churn by contract"):
            rate = rate.reindex(["Month-to-month", "One year", "Two year"])
        bars = ax.bar(rate.index.astype(str), rate.values, color=colour)
        bar_labels(ax, bars)
        ax.axhline(overall, color=RED, ls="--", lw=0.9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel("Churn rate (%)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_1_eda.png")
    plt.close(fig)


def model_charts(art):
    models, X_te, y_te = art["models"], art["X_te"], art["y_te"]
    m = pd.read_csv(OUT / "metrics.csv")
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    colours = [GREY, "#95a5a6", TEAL, RED]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.2
    for i, (_, r) in enumerate(m.iterrows()):
        vals = [r[c] for c in cols]
        bars = ax.bar(np.arange(5) + i * w, vals, w, label=r["model"], color=colours[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(np.arange(5) + 1.5 * w)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
    ax.set_ylim(0, 1.1)
    ax.legend(ncol=4, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_2_model_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, name in zip(axes, models):
        row = m[m["model"] == name].iloc[0]
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", fontweight="bold",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["No churn", "Churn"])
        ax.set_yticks([0, 1], ["No churn", "Churn"])
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_3_confusion.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for (name, pipe), c in zip(models.items(), [GREY, TEAL, RED]):
        p = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        auc = m.loc[m["model"] == name, "roc_auc"].iloc[0]
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Majority baseline (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_4_roc.png")
    plt.close(fig)


def risk_charts():
    g = pd.read_csv(OUT / "risk_validation.csv").set_index("risk_group")
    colours = ["#2e8b57", "#e0a030", RED]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    b = axes[0].bar(g.index, g["customers"], color=colours)
    axes[0].bar_label(b, fontsize=7)
    axes[0].set_title("Customers per risk group (test set)", fontsize=9, fontweight="bold")
    b = axes[1].bar(g.index, g["churn_rate_%"], color=colours)
    axes[1].bar_label(b, fmt="%.1f%%", fontsize=7)
    axes[1].set_title("Actual churn rate per risk group (%)", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_5_risk_groups.png")
    plt.close(fig)


def shap_charts(art):
    X_te = art["X_te"]
    sv = np.load(OUT / "shap_values.npy")
    imp = pd.read_csv(OUT / "shap_importance.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"], imp["mean_abs_shap"], color=NAVY)
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_6_shap_bar.png")
    plt.close(fig)
    shap.summary_plot(sv, X_te, max_display=10, show=False)
    plt.gcf().savefig(FIG / "figA_8_shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close("all")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    art = joblib.load(OUT / "models.joblib")
    eda_charts(load_prepared(RAW))
    model_charts(art)
    risk_charts()
    shap_charts(art)
    print("Figures saved in", FIG)


if __name__ == "__main__":
    main()
