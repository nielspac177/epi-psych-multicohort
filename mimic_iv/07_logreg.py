"""Logistic regression: any_psych ~ surgical + age + female (full cohort + matched)."""
import pandas as pd, numpy as np, statsmodels.api as sm
from pathlib import Path
ANA = Path("/Volumes/Niels 2/MIMIC/physionet.org/files/mimiciv/3.1/analysis/epilepsy_psych")
OUT = ANA / "psm_results"

df = pd.read_csv(ANA / "epilepsy_patient_cohort_psm.csv")
def fit(d, label):
    X = sm.add_constant(d[["surgical","anchor_age","female"]].astype(float))
    y = d["any_psych"].astype(int)
    m = sm.Logit(y, X).fit(disp=False)
    res = pd.DataFrame({
        "term": ["(Intercept)","Surgery (vs none)","Age (per year)","Female (vs male)"],
        "coef": m.params.values,
        "OR":   np.exp(m.params.values),
        "OR_lo":np.exp(m.conf_int()[0].values),
        "OR_hi":np.exp(m.conf_int()[1].values),
        "p":    m.pvalues.values,
    })
    res.to_csv(OUT / f"logreg_{label}.csv", index=False)
    print(f"\n=== {label}  (n={len(d)}, events={int(y.sum())}) ===")
    print(res.round(4).to_string(index=False))
    return res, m

# Full cohort
fit(df, "full")

# Matched cohort A
pairs = pd.read_csv(OUT / "matched_pairs_A_full.csv")
matched = pd.concat([
    df.set_index("subject_id").loc[pairs["surg_subject"]].reset_index(),
    df.set_index("subject_id").loc[pairs["ctrl_subject"]].reset_index(),
])
fit(matched, "matched_A")
