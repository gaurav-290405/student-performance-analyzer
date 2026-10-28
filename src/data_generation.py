"""Synthetic data in the SAME schema as UCI student-mat.csv (Stage 3 fallback).
Assumptions are documented in docs/data_source.md. Replace with the real file if you can."""
import numpy as np, pandas as pd

def generate(n=395, seed=42):
    rng = np.random.default_rng(seed)
    yn = lambda p: np.where(rng.random(n) < p, "yes", "no")
    df = pd.DataFrame({
        "school": rng.choice(["GP", "MS"], n, p=[0.88, 0.12]),
        "sex": rng.choice(["F", "M"], n, p=[0.53, 0.47]),
        "age": rng.choice([15, 16, 17, 18, 19, 20, 21, 22], n, p=[.21, .26, .25, .21, .04, .01, .01, .01]),
        "address": rng.choice(["U", "R"], n, p=[0.78, 0.22]),
        "famsize": rng.choice(["GT3", "LE3"], n, p=[0.71, 0.29]),
        "Pstatus": rng.choice(["T", "A"], n, p=[0.9, 0.1]),
        "Medu": rng.integers(0, 5, n), "Fedu": rng.integers(0, 5, n),
        "Mjob": rng.choice(["at_home", "health", "other", "services", "teacher"], n),
        "Fjob": rng.choice(["at_home", "health", "other", "services", "teacher"], n),
        "reason": rng.choice(["course", "home", "other", "reputation"], n),
        "guardian": rng.choice(["mother", "father", "other"], n, p=[.69, .23, .08]),
        "traveltime": rng.choice([1, 2, 3, 4], n, p=[.65, .27, .06, .02]),
        "studytime": rng.choice([1, 2, 3, 4], n, p=[.27, .5, .16, .07]),
        "failures": rng.choice([0, 1, 2, 3], n, p=[.79, .13, .04, .04]),
        "schoolsup": yn(.13), "famsup": yn(.61), "paid": yn(.46), "activities": yn(.51),
        "nursery": yn(.79), "higher": yn(.95), "internet": yn(.83), "romantic": yn(.33),
        "famrel": rng.choice([1, 2, 3, 4, 5], n, p=[.02, .05, .17, .49, .27]),
        "freetime": rng.choice([1, 2, 3, 4, 5], n, p=[.05, .16, .4, .29, .1]),
        "goout": rng.choice([1, 2, 3, 4, 5], n, p=[.06, .26, .33, .22, .13]),
        "Dalc": rng.choice([1, 2, 3, 4, 5], n, p=[.7, .19, .07, .02, .02]),
        "Walc": rng.choice([1, 2, 3, 4, 5], n, p=[.38, .22, .2, .13, .07]),
        "health": rng.choice([1, 2, 3, 4, 5], n),
        "absences": np.minimum(rng.negative_binomial(1.2, 0.17, n), 75),
    })
    ability = rng.normal(0, 1, n)
    base = (10.5 + 1.6 * ability + 0.6 * (df.studytime - 2) - 1.7 * df.failures
            - 0.25 * (df.goout - 3) - 0.05 * df.absences + 0.2 * (df.Medu - 2)
            + 0.8 * (df.higher == "yes") - 0.15 * (df.age - 17) + 0.3 * (df.internet == "yes"))
    g1 = np.clip(np.round(base + rng.normal(0, 1.6, n)), 0, 20)
    g2 = np.clip(np.round(0.5 * base + 0.5 * g1 + rng.normal(0, 1.3, n)), 0, 20)
    g3 = np.clip(np.round(0.4 * base + 0.6 * g2 + rng.normal(0, 1.2, n)), 0, 20)
    g3[rng.random(n) < 0.03] = 0   # a few dropouts scoring 0, as in the real data
    df["G1"], df["G2"], df["G3"] = g1.astype(int), g2.astype(int), g3.astype(int)
    return df
