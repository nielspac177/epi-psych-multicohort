"""Multivariable logistic regression: any_psych ~ all PSM covariates.
Identifies which predictors actually drive psychiatric comorbidity differences."""
import pandas as pd, numpy as np, statsmodels.api as sm
from pathlib import Path
ANA = Path("/Volumes/Niels 2/MIMIC/physionet.org/files/mimiciv/3.1/analysis/epilepsy_psych")
OUT = ANA / "psm_results"

df = pd.read_csv(ANA / "epilepsy_patient_cohort_psm.csv")

PRED = [
    ("surgical",     "Surgery (vs none)"),
    ("anchor_age",   "Age (per year)"),
    ("female",       "Female (vs male)"),
    ("n_asms",       "Unique ASMs (per drug)"),
    ("intractable",  "Drug-resistant code"),
    ("focal",        "Focal epilepsy"),
    ("se",           "Status epilepticus history"),
    ("n_epi_hadm",   "Epilepsy admissions (per +1)"),
    ("ins_Medicaid", "Insurance: Medicaid (vs Medicare)"),
    ("ins_Private",  "Insurance: Private (vs Medicare)"),
    ("ins_Other",    "Insurance: Other (vs Medicare)"),
]
cols = [v for v, _ in PRED]
X = sm.add_constant(df[cols].astype(float))
y = df["any_psych"].astype(int)
m = sm.Logit(y, X).fit(disp=False)

res = pd.DataFrame({
    "term": ["(Intercept)"] + [lab for _, lab in PRED],
    "coef": m.params.values,
    "OR":   np.exp(m.params.values),
    "OR_lo":np.exp(m.conf_int()[0].values),
    "OR_hi":np.exp(m.conf_int()[1].values),
    "p":    m.pvalues.values,
})
res.to_csv(OUT / "logreg_multivariable.csv", index=False)
print(f"n={len(df)}, events={int(y.sum())}, pseudo-R2={m.prsquared:.3f}")
print(res.round(4).to_string(index=False))

# LaTeX table
def fmt_p(p): return "$<$0.001" if p < .001 else f"{p:.3f}"
def fmt_row(r):
    star = "\\textbf{" if r["p"] < 0.05 else ""
    end  = "}" if r["p"] < 0.05 else ""
    return f"\\quad {star}{r['term']}{end} & {star}{r['OR']:.2f}{end} & {star}({r['OR_lo']:.2f}--{r['OR_hi']:.2f}){end} & {star}{fmt_p(r['p'])}{end} \\\\"

lines = [
    r"\begin{table}[H]\centering",
    r"\caption{Multivariable logistic regression for any psychiatric comorbidity in the full epilepsy cohort. All propensity-score covariates are included as simultaneous predictors to identify the variables actually driving psychiatric comorbidity prevalence after mutual adjustment.}",
    r"\label{tab:logreg_multi}",
    r"\small",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"\textbf{Predictor} & \textbf{aOR} & \textbf{95\% CI} & \textbf{$p$} \\",
    r"\midrule",
    rf"\multicolumn{{4}}{{l}}{{\textit{{n = {len(df):,}; events (any psych) = {int(y.sum()):,}; pseudo-$R^2$ = {m.prsquared:.3f}}}}}\\",
    r"\addlinespace",
]
for _, r in res.iterrows():
    if r["term"] == "(Intercept)": continue
    lines.append(fmt_row(r))
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\vspace{2pt}\par\footnotesize\textit{Note}: Adjusted odds ratios (aOR) from a single multivariable logistic regression with all listed predictors entered simultaneously. Reference categories: Medicare (insurance), White (race), male (sex). Bold rows indicate $p<0.05$. \textit{Surgery} is the variable of interest from the matched analysis; the other rows identify the covariates that, after mutual adjustment, are independently associated with the presence of any psychiatric comorbidity in MIMIC-IV epilepsy patients.",
    r"\end{table}",
]
(ANA/"overleaf"/"sections_psm"/"logreg_multi.tex").write_text("\n".join(lines))
print("\nwrote logreg_multi.tex")
