# Run with:  streamlit run app/app.py   (run  python run_pipeline.py  first)
import json
from pathlib import Path
import joblib, pandas as pd, plotly.express as px, streamlit as st

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Student Performance Analyzer", layout="wide")

@st.cache_data
def load():
    return pd.read_csv(ROOT / "data/processed/students_clean.csv"), json.loads((ROOT / "reports/metrics.json").read_text())
data, metrics = load()
bundle = joblib.load(ROOT / "models/student_risk_model.joblib")
model, thr, FEATURES = bundle["model"], bundle["threshold"], bundle["features"]
reg = joblib.load(ROOT / "models/grade_regressor.joblib")

st.title("🎓 Student Performance Analyzer")
st.caption(f"Data: {metrics['data_source']}")
if "SYNTHETIC" in metrics["data_source"]:
    st.warning("Demo uses synthetic data. Add data/raw/student-mat.csv and re-run run_pipeline.py for real results.")

# ---- sidebar filters
st.sidebar.header("Filters")
school = st.sidebar.selectbox("School", ["All"] + sorted(data["school"].unique()))
sex = st.sidebar.selectbox("Sex", ["All"] + sorted(data["sex"].unique()))
view = data.copy()
if school != "All": view = view[view["school"] == school]
if sex != "All": view = view[view["sex"] == sex]

tab1, tab2, tab3 = st.tabs(["Class overview", "At-risk students", "Check one student"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(view))
    c2.metric("Pass rate", f"{view['pass'].mean():.0%}")
    c3.metric("Average final grade (0-20)", f"{view['G3'].mean():.1f}")
    c4.metric("Average absences", f"{view['absences'].mean():.1f}")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(view.groupby("studytime")["pass"].mean().reset_index(), x="studytime", y="pass",
                          title="Pass rate by study level", labels={"studytime": "Study level (1-4)", "pass": "Pass rate"}),
                   use_container_width=True)
    b.plotly_chart(px.box(view.assign(result=view["pass"].map({0: "Fail", 1: "Pass"})), x="result", y="absences_capped",
                          title="Absences: failing vs passing", labels={"absences_capped": "Absences (capped)", "result": ""}),
                   use_container_width=True)
    a.plotly_chart(px.bar(view.groupby("failures")["pass"].mean().reset_index(), x="failures", y="pass",
                          title="Pass rate by past failures", labels={"failures": "Past failures", "pass": "Pass rate"}),
                   use_container_width=True)
    b.plotly_chart(px.histogram(view, x="G3", nbins=21, title="Final grade distribution", labels={"G3": "Final grade (0-20)"}),
                   use_container_width=True)

with tab2:
    risk = view.copy()
    risk["risk_of_failing"] = model.predict_proba(risk[FEATURES])[:, 1]
    risk["flagged"] = risk["risk_of_failing"] >= thr
    st.write(f"Students at or above the alert threshold (**{thr:.2f}**): **{int(risk['flagged'].sum())}** of {len(risk)}")
    show = risk[risk["flagged"]].sort_values("risk_of_failing", ascending=False)
    st.dataframe(show[["school", "sex", "age", "failures", "studytime", "absences", "risk_of_failing"]]
                 .style.format({"risk_of_failing": "{:.0%}"}), use_container_width=True)
    st.download_button("Download list (CSV)", show.to_csv(index=False), "at_risk_students.csv")

with tab3:
    st.subheader("Early-warning check (no exam grades needed)")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 15, 22, 17)
    studytime = c1.slider("Study level (1 low - 4 high)", 1, 4, 2)
    failures = c1.number_input("Past failures", 0, 4, 0)
    absences = c2.number_input("Absences", 0, 100, 4)
    goout = c2.slider("Going out (1-5)", 1, 5, 3)
    health = c2.slider("Health (1-5)", 1, 5, 3)
    medu = c3.slider("Mother's education (0-4)", 0, 4, 2)
    higher = c3.checkbox("Wants higher education", True)
    internet = c3.checkbox("Internet at home", True)
    if st.button("Predict"):
        base = data[FEATURES].median(numeric_only=True)
        row = {f: (base[f] if pd.api.types.is_numeric_dtype(data[f]) else data[f].mode()[0]) for f in FEATURES}
        row.update(age=age, studytime=studytime, failures=failures, absences_capped=min(absences, data["absences_capped"].max()),
                   goout=goout, health=health, Medu=medu, higher=int(higher), internet=int(internet))
        row = pd.DataFrame([row])[FEATURES]
        p_fail = model.predict_proba(row)[0, 1]
        st.metric("Risk of failing", f"{p_fail:.0%}")
        (st.error if p_fail >= thr else st.success)(
            "⚠️ At risk - consider early support." if p_fail >= thr else "✅ Not flagged as at risk.")
        st.write(f"Expected final grade: **{reg.predict(row)[0]:.1f} / 20** (typical error ±{metrics['grade_mae']})")
        st.caption("Use this to start a support conversation, not to label a student.")
    st.divider()
    st.caption(f"Model test results - recall on fail class: {metrics['recall_fail']}, precision: {metrics['precision_fail']}, ROC-AUC: {metrics['roc_auc']}")
