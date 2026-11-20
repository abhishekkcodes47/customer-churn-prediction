"""Pre-processing: cleaning, feature engineering, validation and risk assignment."""
import pandas as pd

YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, fill blank TotalCharges, remove duplicate customers."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.drop_duplicates(subset="customerID")
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add average charge per month of tenure (tenure 0 treated as 1)."""
    df = df.copy()
    df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


def assign_risk(prob, low=0.3, high=0.6):
    """Map churn probability to Low / Medium / High risk group."""
    if not 0 <= prob <= 1:
        raise ValueError("probability must be between 0 and 1")
    return "Low" if prob < low else ("Medium" if prob < high else "High")


def validate_data(df: pd.DataFrame) -> pd.Series:
    """Apply the rules of Table 3.5. Returns a boolean Series: True = record flagged."""
    ids = df["customerID"].astype(str).str.strip()
    bad = df["customerID"].isna() | (ids == "") | df["customerID"].duplicated()
    bad |= ~df["gender"].isin(["Male", "Female"])
    bad |= ~df["SeniorCitizen"].isin([0, 1])
    for col in YES_NO_COLS:
        if col in df.columns:
            bad |= ~df[col].isin(["Yes", "No"])
    for col in ADDON_COLS:
        bad |= ~df[col].isin(["Yes", "No", "No internet service"])
    bad |= ~df["MultipleLines"].isin(["Yes", "No", "No phone service"])
    bad |= ~df["InternetService"].isin(["DSL", "Fiber optic", "No"])
    bad |= ~df["Contract"].isin(CONTRACTS)
    bad |= ~df["PaymentMethod"].isin(PAYMENTS)
    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    bad |= tenure.isna() | (tenure < 0) | (tenure % 1 != 0)
    bad |= ~(pd.to_numeric(df["MonthlyCharges"], errors="coerce") > 0)
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    bad |= (total < 0) | (total.isna() & (tenure != 0))
    return bad


def build_matrix(df: pd.DataFrame):
    """One-hot encode features. Returns X (float), y (0/1) and the customer ids."""
    X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]),
                       drop_first=True).astype(float)
    y = (df["Churn"] == "Yes").astype(int)
    return X, y, df["customerID"]
