# Project Charter - Student Performance Analyzer
**Problem:** Teachers find out too late that a student is struggling. We predict which students are at risk of failing so support can begin early.
**Target user:** Class teacher / academic coordinator
**Business questions:**
1. Which factors influence final grades the most?
2. Which students are at risk of failing?
3. Do absences and study habits change the outcome?
**Task type:** Binary classification (pass/fail, pass = G3 >= 10) + grade regression (G3)
**Success metrics:** Recall >= 0.80 on the "fail" class; MAE < 2 grade points
**In scope:** Mathematics course data, prediction, Streamlit dashboard
**Out of scope:** Individual counselling advice, other courses
**Risks:** Sensitive attributes (sex, family background) - fairness checked; small dataset; G1/G2 excluded to keep it an *early* warning.
