# Student Performance Analyzer
Teachers often learn too late that a student is struggling. This project predicts students at risk of failing
(final grade G3 < 10 of 20) so support can start early, and shows the results in a Streamlit dashboard.

## Data
UCI Student Performance dataset (mathematics), 395 students, 33 columns. See `docs/data_source.md` and `docs/data_dictionary.md`.
Place `student-mat.csv` in `data/raw/`. If absent, a synthetic dataset with the same schema is generated automatically.

## Approach
Cleaning -> EDA + 6 hypothesis tests -> feature engineering (pipeline, no G1/G2) -> baseline (majority class, logistic regression)
-> decision tree -> Random Forest tuned with grid search (CV on train only) -> decision threshold tuned on train CV for recall >= 0.80
-> single evaluation on the held-out test set -> permutation importance + fairness check -> grade regression (MAE).

## How to run
```
pip install -r requirements.txt
python run_pipeline.py          # cleans, trains, evaluates, saves models + reports/
streamlit run app/app.py        # dashboard
```

## Key results
See `reports/metrics.json`, `reports/experiment_table.csv`, charts in `reports/`. **Fill in your own numbers after running on the real data.**

## Limitations and future work
Small dataset, one school system; predictions should guide support conversations, not label students;
add other courses, more fairness checks, SHAP explanations.

## Structure
```
data/raw, data/processed   notebooks/ (optional: copy sections of run_pipeline.py)
src/ (cleaning, features, data_generation)   models/   app/   docs/   reports/
```
