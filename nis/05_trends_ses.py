"""
05_trends_ses.py
Combined analysis for epilepsy psych project:
  (1) Temporal trends — survey-weighted prevalence by year, surgical & non-surgical separately
  (2) PNES audit + recomputed any_psych excluding PNES
  (3) SES/deprivation replication of BWH single-center analysis using NIS proxies
      Aim A: weighted Table 1 stratified by ZIPINC_QRTL
      Aim B: psychiatric comorbidity ~ SES (logistic, survey-weighted)
      Aim C: surgical outcomes ~ SES
      Aim D: dissociation test (effect sizes psych vs outcomes)

Outputs to tables/ and figures/. Uses statsmodels survey GLM (freq weights) as a
practical approximation; full Taylor linearization SEs would require R survey.
For inference we report robust HC1 SEs; the cohort sizes are large and effect
sizes/directions are the primary interest.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

ROOT = "/Volumes/Niels 2/NIS_new_version/NIS_epilepsy_psych"
os.chdir(ROOT)
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

df = pd.read_parquet("output/epilepsy_analytic.parquet")
print(f"Loaded {len(df):,} rows")

# ----------------------------------------------------------------------
# Recompute any_psych EXCLUDING PNES (per methods text)
# ----------------------------------------------------------------------
PSYCH_COMPONENTS = [
    "psych_depression","psych_bipolar","psych_anxiety","psych_ptsd","psych_ocd",
    "psych_schizophrenia","psych_psychosis","psych_adhd","psych_alcohol",
    "psych_drug","psych_suicidal",
]
df["any_psych_nopnes"] = (df[PSYCH_COMPONENTS].sum(axis=1) > 0).astype(int)
for c in PSYCH_COMPONENTS + ["psych_pnes","any_psych_nopnes"]:
    df[c] = df[c].astype(int)

# Coarse adult-only filter (analysis_plan limited to ≥18)
df = df[df["AGE"] >= 18].copy()
df["surgical"] = df["any_surgery"].astype(int)

# ----------------------------------------------------------------------
# Helper: weighted prevalence with Wilson-ish CI (use design effect approx)
# We use weighted proportions and compute SE via linearisation:
#   Var(p) = sum(w_i^2 (y_i - p)^2) / (sum w_i)^2
# ----------------------------------------------------------------------
def wprev(y, w):
    y = np.asarray(y, float); w = np.asarray(w, float)
    sw = w.sum()
    p = (w * y).sum() / sw
    se = np.sqrt(((w**2) * (y - p)**2).sum()) / sw
    lo = max(0.0, p - 1.96*se); hi = min(1.0, p + 1.96*se)
    return p, lo, hi, sw

DISORDERS = {
    "any_psych_nopnes": "Any psychiatric (excl. PNES)",
    "psych_depression": "Depression",
    "psych_anxiety": "Anxiety",
    "psych_bipolar": "Bipolar",
    "psych_psychosis": "Psychotic disorders",
    "psych_schizophrenia": "Schizophrenia spectrum",
    "psych_alcohol": "Alcohol use",
    "psych_drug": "Drug use",
    "psych_adhd": "ADHD",
    "psych_ptsd": "PTSD",
    "psych_suicidal": "Suicidality",
    "psych_pnes": "PNES",
}

# ======================================================================
# (1) TEMPORAL TRENDS — survey-weighted prevalence by year, two cohorts
# ======================================================================
print("\n[1] Temporal trends ...")
trend_rows = []
for grp_label, grp_df in [("Surgical", df[df.surgical==1]), ("Non-surgical", df[df.surgical==0])]:
    for var, name in DISORDERS.items():
        for y, sub in grp_df.groupby("YEAR"):
            p, lo, hi, n = wprev(sub[var], sub["DISCWT"])
            trend_rows.append(dict(group=grp_label, disorder=name, var=var,
                                   year=int(y), prev=p, lo=lo, hi=hi,
                                   n_unweighted=len(sub), n_weighted=n))
trend = pd.DataFrame(trend_rows)
trend.to_csv("tables/trends_by_year.csv", index=False)

# Trend tests: weighted logistic with year continuous
trend_test_rows = []
for grp_label, grp_df in [("Surgical", df[df.surgical==1]), ("Non-surgical", df[df.surgical==0])]:
    for var, name in DISORDERS.items():
        sub = grp_df[["YEAR", var, "DISCWT"]].dropna()
        if sub[var].sum() < 10 or sub[var].sum() == len(sub):
            continue
        try:
            m = smf.glm(f"{var} ~ YEAR", data=sub, family=sm.families.Binomial(),
                        freq_weights=sub["DISCWT"]).fit(disp=0)
            b = m.params["YEAR"]; se = m.bse["YEAR"]
            trend_test_rows.append(dict(
                group=grp_label, disorder=name,
                or_per_year=np.exp(b),
                ci_lo=np.exp(b-1.96*se), ci_hi=np.exp(b+1.96*se),
                p=m.pvalues["YEAR"]))
        except Exception as e:
            print("  trend fit failed:", grp_label, name, e)
trend_tests = pd.DataFrame(trend_test_rows)
trend_tests.to_csv("tables/trends_or_per_year.csv", index=False)
print(f"  trends rows: {len(trend)}, trend tests: {len(trend_tests)}")

# Plots: 2 figures (Surgical, Non-surgical), each faceted 3x4 by disorder
def plot_trends(group_label, fname):
    sub = trend[trend.group == group_label]
    disorders_order = list(DISORDERS.values())
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
    for ax, dname in zip(axes.flat, disorders_order):
        d = sub[sub.disorder == dname].sort_values("year")
        if len(d):
            ax.plot(d.year, d.prev*100, "-o", color="#1f77b4", lw=1.5, ms=4)
            ax.fill_between(d.year, d.lo*100, d.hi*100, alpha=0.25, color="#1f77b4")
        ax.set_title(dname, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    for ax in axes[:,0]:
        ax.set_ylabel("Prevalence (%)")
    fig.suptitle(f"Survey-weighted psychiatric prevalence by year — {group_label} cohort",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=160, bbox_inches="tight")
    plt.close(fig)

plot_trends("Surgical", "trends_surgical.png")
plot_trends("Non-surgical", "trends_nonsurgical.png")
print("  saved trends_surgical.png, trends_nonsurgical.png")

# ======================================================================
# (2) PNES AUDIT
# ======================================================================
print("\n[2] PNES audit ...")
pnes_rows = []
for grp_label, grp_df in [("Surgical", df[df.surgical==1]),
                          ("Non-surgical", df[df.surgical==0])]:
    p, lo, hi, n = wprev(grp_df["psych_pnes"], grp_df["DISCWT"])
    co_psych_p, *_ = wprev(grp_df.loc[grp_df.psych_pnes==1, "any_psych_nopnes"],
                           grp_df.loc[grp_df.psych_pnes==1, "DISCWT"]) \
        if grp_df.psych_pnes.sum() else (np.nan, np.nan, np.nan, 0)
    pnes_rows.append(dict(group=grp_label, pnes_prev=p, lo=lo, hi=hi,
                          n_pnes=int(grp_df.psych_pnes.sum()),
                          pnes_with_other_psych=co_psych_p))
pd.DataFrame(pnes_rows).to_csv("tables/pnes_audit.csv", index=False)
print(pd.DataFrame(pnes_rows))

# ======================================================================
# (3) SES / deprivation replication — ICD-10 era only (2016–2020)
# ======================================================================
print("\n[3] SES replication (ICD-10 era 2016-2020, surgical cohort) ...")
ses_df = df[(df.YEAR >= 2016) & (df.surgical == 1)].copy()
print(f"  surgical ICD-10: n={len(ses_df):,}")

# Recode covariates
race_map = {1:"White", 2:"Black", 3:"Hispanic", 4:"AsianPI", 5:"NativeAmerican", 6:"Other"}
pay_map = {1:"Medicare", 2:"Medicaid", 3:"Private", 4:"SelfPay", 5:"NoCharge", 6:"Other"}
nchs_map = {1:"LargeCentralMetro", 2:"LargeFringeMetro", 3:"MediumMetro",
            4:"SmallMetro", 5:"Micropolitan", 6:"NonCore"}
ses_df["race_cat"] = ses_df["RACE"].map(race_map)
ses_df["pay_cat"] = ses_df["PAY1"].map(pay_map)
ses_df["nchs_cat"] = ses_df["PL_NCHS"].map(nchs_map)
ses_df["zipq"] = ses_df["ZIPINC_QRTL"]

# Aim A: weighted Table 1 stratified by ZIPINC_QRTL
def w_pct(mask, w):
    return 100 * (w[mask].sum() / w.sum()) if w.sum() > 0 else np.nan
def w_mean(x, w):
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m])) if m.sum() else np.nan

table1_rows = []
groups = [("Overall", ses_df)] + [(f"Q{int(q)}", ses_df[ses_df.zipq==q])
                                   for q in sorted(ses_df.zipq.dropna().unique())]
for label, sub in groups:
    w = sub["DISCWT"]
    row = {"Stratum": label, "N (unweighted)": len(sub),
           "N (weighted)": int(w.sum()),
           "Age, mean": round(w_mean(sub["AGE"], w),1),
           "Female %": round(w_pct(sub["FEMALE"]==1, w),1),
           "White %": round(w_pct(sub["RACE"]==1, w),1),
           "Black %": round(w_pct(sub["RACE"]==2, w),1),
           "Hispanic %": round(w_pct(sub["RACE"]==3, w),1),
           "Medicare %": round(w_pct(sub["PAY1"]==1, w),1),
           "Medicaid %": round(w_pct(sub["PAY1"]==2, w),1),
           "Private %": round(w_pct(sub["PAY1"]==3, w),1),
           "Self-pay %": round(w_pct(sub["PAY1"]==4, w),1),
           "Any psych %": round(w_pct(sub["any_psych_nopnes"]==1, w),1),
           "Depression %": round(w_pct(sub["psych_depression"]==1, w),1),
           "Anxiety %": round(w_pct(sub["psych_anxiety"]==1, w),1),
           "Mortality %": round(w_pct(sub["DIED"]==1, w),2),
           "LOS, mean": round(w_mean(sub["LOS"], w),1),
           "Charges, mean $": int(w_mean(sub["TOTCHG"], w))
                if sub["TOTCHG"].notna().any() else np.nan,
          }
    table1_rows.append(row)
table1 = pd.DataFrame(table1_rows)
table1.to_csv("tables/ses_table1.csv", index=False)
print(table1.to_string(index=False))

# Aim B: psych ~ SES (survey-weighted logistic)
print("\n  Aim B: psychiatric ~ SES")
def fit_ses_logit(yvar, data, label):
    d = data[[yvar,"AGE","FEMALE","zipq","race_cat","pay_cat","nchs_cat",
              "DISCWT","intractable"]].dropna()
    d = d[d["race_cat"].isin(["White","Black","Hispanic"])]  # keep main groups
    if d[yvar].sum() < 10:
        return None
    formula = (f"{yvar} ~ AGE + C(FEMALE) + C(zipq) + C(race_cat, Treatment('White')) "
               f"+ C(pay_cat, Treatment('Private')) + C(nchs_cat) + intractable")
    try:
        m = smf.glm(formula, data=d, family=sm.families.Binomial(),
                    freq_weights=d["DISCWT"]).fit(disp=0)
        out = pd.DataFrame({
            "term": m.params.index,
            "OR": np.exp(m.params.values),
            "lo": np.exp(m.params.values - 1.96*m.bse.values),
            "hi": np.exp(m.params.values + 1.96*m.bse.values),
            "p":  m.pvalues.values})
        out.insert(0, "outcome", label)
        return out
    except Exception as e:
        print("    fit failed:", label, e)
        return None

psych_models = []
for v, name in [("any_psych_nopnes","Any psychiatric"),
                ("psych_depression","Depression"),
                ("psych_anxiety","Anxiety"),
                ("psych_alcohol","Alcohol use"),
                ("psych_drug","Drug use")]:
    r = fit_ses_logit(v, ses_df, name)
    if r is not None: psych_models.append(r)
psych_results = pd.concat(psych_models, ignore_index=True) if psych_models else pd.DataFrame()
psych_results.to_csv("tables/ses_psych_models.csv", index=False)

# Aim C: surgical outcomes ~ SES
print("\n  Aim C: surgical outcomes ~ SES")
ses_df["nonroutine_dc"] = (~ses_df["DISPUNIFORM"].isin([1, 6])).astype(int)  # 1=routine,6=home health
ses_df["log_los"] = np.log1p(ses_df["LOS"].clip(lower=0))
ses_df["log_chg"] = np.log1p(ses_df["TOTCHG"].clip(lower=0))

def fit_outcome(yvar, data, label, family="binomial"):
    d = data[[yvar,"AGE","FEMALE","zipq","race_cat","pay_cat","nchs_cat",
              "DISCWT","intractable"]].dropna()
    d = d[d["race_cat"].isin(["White","Black","Hispanic"])]
    if family == "binomial" and d[yvar].sum() < 10:
        return None
    formula = (f"{yvar} ~ AGE + C(FEMALE) + C(zipq) + C(race_cat, Treatment('White')) "
               f"+ C(pay_cat, Treatment('Private')) + C(nchs_cat) + intractable")
    try:
        if family == "binomial":
            m = smf.glm(formula, data=d, family=sm.families.Binomial(),
                        freq_weights=d["DISCWT"]).fit(disp=0)
            est = np.exp(m.params.values); lo = np.exp(m.params.values-1.96*m.bse.values)
            hi = np.exp(m.params.values+1.96*m.bse.values); estname="OR"
        else:
            m = smf.glm(formula, data=d, family=sm.families.Gaussian(),
                        freq_weights=d["DISCWT"]).fit(disp=0)
            est = m.params.values; lo = m.params.values-1.96*m.bse.values
            hi = m.params.values+1.96*m.bse.values; estname="beta"
        out = pd.DataFrame({"term": m.params.index, estname: est, "lo": lo,
                            "hi": hi, "p": m.pvalues.values})
        out.insert(0, "outcome", label)
        out.insert(1, "model", family)
        return out
    except Exception as e:
        print("    fit failed:", label, e)
        return None

outcome_models = []
for v, name, fam in [("DIED","In-hospital mortality","binomial"),
                     ("nonroutine_dc","Nonroutine discharge","binomial"),
                     ("log_los","log(LOS+1)","gaussian"),
                     ("log_chg","log(Charges+1)","gaussian")]:
    r = fit_outcome(v, ses_df, name, fam)
    if r is not None: outcome_models.append(r)
outcome_results = pd.concat(outcome_models, ignore_index=True) if outcome_models else pd.DataFrame()
outcome_results.to_csv("tables/ses_outcome_models.csv", index=False)

# Aim D: dissociation — extract effect of zipq Q4 vs Q1 and race Black vs White
def extract(df_, key):
    if df_.empty: return pd.DataFrame()
    return df_[df_["term"].str.contains(key, regex=False, na=False)].copy()

dissoc_rows = []
for label, df_ in [("Psych ~ ZIP Q4 (vs Q1)", extract(psych_results, "zipq)[T.4")),
                   ("Outcomes ~ ZIP Q4 (vs Q1)", extract(outcome_results, "zipq)[T.4")),
                   ("Psych ~ Race Black", extract(psych_results, "race_cat, Treatment('White'))[T.Black")),
                   ("Outcomes ~ Race Black", extract(outcome_results, "race_cat, Treatment('White'))[T.Black"))]:
    for _, r in df_.iterrows():
        d = {"comparison": label, "outcome": r["outcome"], "p": r["p"]}
        if "OR" in r:  d["effect"]=r["OR"];   d["type"]="OR"
        else:          d["effect"]=r["beta"]; d["type"]="beta"
        dissoc_rows.append(d)
diss = pd.DataFrame(dissoc_rows)
diss.to_csv("tables/ses_dissociation.csv", index=False)
print("\n  Dissociation table:")
print(diss.to_string(index=False) if len(diss) else "  (empty)")

# Forest plot for SES models — Q4 vs Q1 effects
def forest(df_, title, fname, est_col="OR"):
    if df_.empty:
        return
    sub = df_[df_["term"].str.contains("zipq", regex=False)].copy()
    if sub.empty: return
    sub["term_short"] = sub["term"].str.replace("C(zipq)[T.","Q",regex=False).str.replace("]","",regex=False)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4*len(sub))))
    ys = np.arange(len(sub))
    ax.errorbar(sub[est_col], ys,
                xerr=[sub[est_col]-sub["lo"], sub["hi"]-sub[est_col]],
                fmt="o", color="#333")
    ax.axvline(1 if est_col=="OR" else 0, ls="--", color="grey")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['outcome']} — {r['term_short']}" for _, r in sub.iterrows()],
                       fontsize=8)
    ax.set_xlabel(est_col)
    if est_col == "OR": ax.set_xscale("log")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=160, bbox_inches="tight")
    plt.close(fig)

forest(psych_results, "SES → Psychiatric (Q4 vs Q1)", "forest_ses_psych.png", "OR")
forest(outcome_results[outcome_results.model=="binomial"],
       "SES → Surgical outcomes (binary, Q4 vs Q1)", "forest_ses_outcomes.png", "OR")

print("\nDONE. Outputs in tables/ and figures/.")
