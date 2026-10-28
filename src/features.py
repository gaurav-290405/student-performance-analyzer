"""Stage 7: features + preprocessing pipeline (early-warning: NO G1/G2/G3)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
            "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid",
            "activities", "higher", "internet", "romantic", "famrel", "freetime",
            "goout", "Dalc", "Walc", "health", "absences_capped", "support_index",
            "alcohol_total"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["support_index"] = df["schoolsup"] + df["famsup"] + df["paid"]   # extra support received
    df["alcohol_total"] = df["Dalc"] + df["Walc"]
    df["grade_trend"] = df["G2"] - df["G1"]    # analysis only - NOT a model input (leakage)
    return df

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes("number").columns
    cat = X.select_dtypes(exclude="number").columns
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
