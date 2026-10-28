"""Runs Stages 3-10 end to end:  python run_pipeline.py"""
import json, sys, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             recall_score, precision_score, f1_score, roc_auc_score,
                             mean_absolute_error)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent))
from src.cleaning import clean_students
from src.data_generation import generate
from src.features import FEATURES, add_features, build_preprocessor
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
RAW = ROOT / "data/raw/student-mat.csv"; REP = ROOT / "reports"

# ---- Stage 3/4: load + profile --------------------------------------------
if RAW.exists():
    df = pd.read_csv(RAW, sep=";"); source = "UCI student-mat.csv (real data)"
else:
    df = generate(); df.to_csv(ROOT / "data/raw/synthetic_students.csv", sep=";", index=False)
    source = "SYNTHETIC data (UCI schema) - replace with real student-mat.csv"
print("DATA SOURCE:", source, "| shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()), "| duplicates:", int(df.duplicated().sum()))

# ---- Stage 5: clean --------------------------------------------------------
clean = add_features(clean_students(df))
clean.to_csv(ROOT / "data/processed/students_clean.csv", index=False)
print("Fail share:", round(1 - clean["pass"].mean(), 3))

# ---- Stage 6: EDA (charts + hypothesis tests) -----------------------------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(data=clean, x="studytime", y="pass", ax=ax[0]); ax[0].set_title("Pass rate by study level")
sns.boxplot(data=clean, x="pass", y="absences_capped", ax=ax[1]); ax[1].set_title("Absences: fail (0) vs pass (1)")
sns.barplot(data=clean, x="failures", y="pass", ax=ax[2]); ax[2].set_title("Pass rate by past failures")
plt.tight_layout(); plt.savefig(REP / "eda_overview.png", dpi=120); plt.close()
corr = clean.select_dtypes("number").corr()["G3"].drop(["G3", "pass"]).sort_values()
plt.figure(figsize=(6, 7)); corr.plot.barh(); plt.title("Correlation with final grade G3")
plt.tight_layout(); plt.savefig(REP / "eda_correlation.png", dpi=120); plt.close()

hyp = []
def test(name, a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False); hyp.append((name, round(p, 4), "significant" if p < .05 else "not significant"))
p_, f_ = clean[clean["pass"] == 1], clean[clean["pass"] == 0]
test("H1 Absences differ between pass/fail", p_["absences_capped"], f_["absences_capped"])
test("H2 Study time differs between pass/fail", p_["studytime"], f_["studytime"])
test("H3 Past failures differ between pass/fail", p_["failures"], f_["failures"])
test("H4 Going out differs between pass/fail", p_["goout"], f_["goout"])
test("H5 Final grade differs by internet access", clean[clean.internet == 1]["G3"], clean[clean.internet == 0]["G3"])
test("H6 Final grade differs by sex", clean[clean.sex == "f"]["G3"], clean[clean.sex == "m"]["G3"])
pd.DataFrame(hyp, columns=["hypothesis", "p_value", "result"]).to_csv(REP / "hypothesis_tests.csv", index=False)
print(pd.DataFrame(hyp, columns=["hypothesis", "p", "result"]).to_string(index=False))

# ---- Stage 7: split --------------------------------------------------------
X, y = clean[FEATURES], clean["pass"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
prep = lambda: build_preprocessor(X)

# ---- Stage 8/9: baseline + models + tuning (CV on train only) --------------
models = {
    "Majority class": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision tree": DecisionTreeClassifier(class_weight="balanced", max_depth=4, random_state=42),
}
# Target for modelling: 1 = student FAILS (the class we must catch)
y_fail_tr, y_fail_te = 1 - y_tr, 1 - y_te
rows = []
for name, m in models.items():
    pipe = Pipeline([("prep", prep()), ("model", m)])
    f1 = cross_val_score(pipe, X_tr, y_fail_tr, cv=5, scoring="f1")
    rc = cross_val_score(pipe, X_tr, y_fail_tr, cv=5, scoring="recall")
    rows.append((name, "default", round(f1.mean(), 3), round(rc.mean(), 3)))

rf = Pipeline([("prep", prep()), ("model", RandomForestClassifier(class_weight="balanced", random_state=42))])
grid = {"model__n_estimators": [100, 300], "model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3, 5]}
search = GridSearchCV(rf, grid, cv=5, scoring="f1", n_jobs=-1).fit(X_tr, y_fail_tr)
rows.append(("Random Forest (tuned)", str(search.best_params_), round(search.best_score_, 3),
             round(cross_val_score(search.best_estimator_, X_tr, y_fail_tr, cv=5, scoring="recall").mean(), 3)))
exp = pd.DataFrame(rows, columns=["model", "settings", "cv_f1_fail", "cv_recall_fail"])
exp.to_csv(REP / "experiment_table.csv", index=False); print(exp.to_string(index=False))

# ---- Stage 10: tune threshold on TRAIN CV, then evaluate test ONCE ---------
from sklearn.model_selection import cross_val_predict
best = search.best_estimator_
oof = cross_val_predict(best, X_tr, y_fail_tr, cv=5, method="predict_proba")[:, 1]
thr = 0.5
for t in np.arange(0.5, 0.05, -0.05):          # highest threshold meeting recall >= 0.80
    if recall_score(y_fail_tr, oof >= t) >= 0.80: thr = float(round(t, 2)); break
else: thr = 0.2
best.fit(X_tr, y_fail_tr)
proba = best.predict_proba(X_te)[:, 1]; pred = (proba >= thr).astype(int)
metrics = {"data_source": source, "threshold_fail": thr,
           "recall_fail": round(recall_score(y_fail_te, pred), 3),
           "precision_fail": round(precision_score(y_fail_te, pred), 3),
           "f1_fail": round(f1_score(y_fail_te, pred), 3),
           "roc_auc": round(roc_auc_score(y_fail_te, proba), 3), "n_test": int(len(y_te))}
print(classification_report(y_fail_te, pred, target_names=["pass", "fail"]))
ConfusionMatrixDisplay(confusion_matrix(y_fail_te, pred), display_labels=["pass", "fail"]).plot()
plt.title("Test confusion matrix"); plt.savefig(REP / "confusion_matrix.png", dpi=120); plt.close()

imp = permutation_importance(best, X_te, y_fail_te, n_repeats=10, random_state=42, scoring="recall")
rank = pd.Series(imp.importances_mean, index=X_te.columns).sort_values()
rank.tail(12).plot.barh(figsize=(6, 5), title="Permutation importance (recall on fail class)")
plt.tight_layout(); plt.savefig(REP / "feature_importance.png", dpi=120); plt.close()
metrics["top_factors"] = list(rank.sort_values(ascending=False).head(5).index)

# fairness check: recall by sex and school
fair = {}
for col in ["sex", "school"]:
    for g in X_te[col].unique():
        mk = (X_te[col] == g).values
        if y_fail_te[mk].sum() > 0: fair[f"{col}={g}"] = round(recall_score(y_fail_te[mk], pred[mk]), 3)
metrics["fairness_recall_by_group"] = fair

# Grade regression (second charter goal): MAE
reg = Pipeline([("prep", prep()), ("model", RandomForestRegressor(300, min_samples_leaf=3, random_state=42))])
reg.fit(X_tr, clean.loc[X_tr.index, "G3"])
metrics["grade_mae"] = round(mean_absolute_error(clean.loc[X_te.index, "G3"], reg.predict(X_te)), 2)

joblib.dump({"model": best, "threshold": thr, "features": FEATURES}, ROOT / "models/student_risk_model.joblib")
joblib.dump(reg, ROOT / "models/grade_regressor.joblib")
(REP / "metrics.json").write_text(json.dumps(metrics, indent=2)); print(json.dumps(metrics, indent=2))
