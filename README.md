# Customer Churn Prediction and Retention Analytics

BCA Summer Training Project
**Student:** Abhishek Kumar (Enrollment No. 04220602024)
**Guide:** Dr. Arti Bajaj, Associate Professor
**Institute:** Trinity Institute of Professional Studies, GGSIPU, Delhi

## About the project
Subscription businesses lose revenue when customers cancel. This project
predicts which telecom customers are likely to churn, explains why using
SHAP, and shows the results in an interactive dashboard so that managers
can target retention offers only where they are needed.

**Pipeline:** Data → Processing → Model → Insight → Dashboard

## What it does
- Cleans and validates the IBM Telco dataset (7,043 customers)
- Compares 3 models with 5-fold stratified cross-validation
- Handles class imbalance with SMOTE inside the pipeline (no data leakage)
- Explains each prediction with SHAP in plain words
- Segments customers into Low / Medium / High risk
- Dashboard with filters, KPIs, at-risk list and CSV export

## Results
- XGBoost: ROC-AUC 0.844, precision 59%, recall 66%
- Actual churn by risk group: 8.1% (Low), 35.8% (Medium), 66.3% (High)
- 25 automated tests, 100% coverage of preprocess.py

## Live dashboard
(Streamlit link yahan daalna hai, step 9 me)

## How to run
    pip install -r requirements.txt
    python -m pytest tests -v
    python src/train.py
    python src/analyze.py
    python src/make_figs.py
    streamlit run dashboard/app.py

## Folders
- `src/` : code (preprocess, train, analyze, make_figs)
- `tests/` : 25 automated tests
- `dashboard/` : Streamlit app
- `outputs/` : metrics, predictions, figures
- `data/raw/` : dataset