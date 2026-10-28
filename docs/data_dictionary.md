# Data dictionary (key columns)
| Column | Meaning | Type | Notes |
|---|---|---|---|
| school | GP or MS | Category | |
| sex | F / M | Category | Sensitive - fairness checked |
| age | 15-22 | Integer | |
| Medu / Fedu | Mother's / father's education | Ordinal 0-4 | |
| studytime | Weekly study level | Ordinal 1-4 | 1 lowest |
| failures | Past class failures | Integer 0-3 | |
| schoolsup / famsup / paid | Extra school / family / paid support | yes/no -> 1/0 | |
| goout, Dalc, Walc, health, famrel, freetime | Lifestyle ratings | Ordinal 1-5 | |
| absences | School absences | Integer | Right-skewed; capped by IQR in `absences_capped` |
| G1, G2 | Period grades | 0-20 | **Not used as features** (early-warning) |
| G3 | Final grade | 0-20 | Regression target |
| pass | G3 >= 10 | 0/1 | Classification target (modelling uses fail = 1 - pass) |
| support_index | schoolsup + famsup + paid | Engineered | |
| alcohol_total | Dalc + Walc | Engineered | |
| grade_trend | G2 - G1 | Engineered | EDA only - leakage risk |
