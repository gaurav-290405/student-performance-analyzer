"""Stage 5: repeatable cleaning function."""
import pandas as pd

YES_NO = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]

def clean_students(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    for col in df.select_dtypes(exclude="number"):
        df[col] = df[col].str.strip().str.lower()
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(df[num].median())
    for c in YES_NO:                                   # yes/no -> 1/0
        if c in df: df[c] = (df[c] == "yes").astype(int)
    q1, q3 = df["absences"].quantile([0.25, 0.75])     # IQR cap for absences
    df["absences_capped"] = df["absences"].clip(upper=q3 + 1.5 * (q3 - q1))
    df["pass"] = (df["G3"] >= 10).astype(int)          # target: pass if G3 >= 10
    return df
