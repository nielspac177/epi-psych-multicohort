"""
Analysis C: PSM and multivariable logreg with ONLY demographics + epilepsy severity.
No race, no insurance.

Covariates: age, female, n_asms, intractable, focal, se, n_epi_hadm
"""
import duckdb, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.contingency_tables import mcnemar

ANA = Path("/Volumes/Niels 2/MIMIC/physionet.org/files/mimiciv/3.1/analysis/epilepsy_psych")
OUT = ANA / "psm_results"
df = pd.read_csv(ANA / "epilepsy_patient_cohort_psm.csv")

PS_C = ["anchor_age","female","n_asms","intractable","focal","se","n_epi_hadm"]
PSYCH = ["has_depression","has_bipolar","has_anxiety","has_ptsd","has_ocd",
         "has_psychotic","has_adhd","has_substance_use","has_suicidal_ideation",
         "has_pnes","any_psych"]

# ---------- PSM ----------
treat = df["surgical"].astype(int).values
X = df[PS_C].astype(float).values
Xs = StandardScaler().fit_transform(X)
lr = LogisticRegression(max_iter=2000).fit(Xs, treat)
ps = lr.predict_proba(Xs)[:, 1]
logit = np.log(np.clip(ps,1e-6,1-1e-6) / np.clip(1-ps,1e-6,1-1e-6))
caliper = 0.2 * np.std(logit)
print(f"caliper = {caliper:.4f}")

t_idx = np.where(treat==1)[0]; c_idx = np.where(treat==0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(logit[c_idx].reshape(-1,1))
used=set(); pairs=[]
rng = np.random.default_rng(42)
for ti in rng.permutation(t_idx):
    dists, idxs = nn.kneighbors([[logit[ti]]], n_neighbors=20)
    for d,j in zip(dists[0], idxs[0]):
        ci = c_idx[j]
        if ci in used: continue
        if d > caliper: break
        used.add(ci); pairs.append((ti,ci,d)); break
print(f"matched: {len(pairs)} / {len(t_idx)}")

pairs_df = pd.DataFrame(pairs, columns=["t_idx","c_idx","dist"])
pairs_df["surg_subject"] = df.loc[pairs_df["t_idx"],"subject_id"].values
pairs_df["ctrl_subject"] = df.loc[pairs_df["c_idx"],"subject_id"].values
pairs_df.to_csv(OUT/"matched_pairs_C_demosev.csv", index=False)

# Balance
bal=[]
for f in PS_C:
    a=df.loc[pairs_df["t_idx"],f].astype(float).values
    b=df.loc[pairs_df["c_idx"],f].astype(float).values
    sd=np.sqrt((a.var()+b.var())/2) or 1
    pa=df[df["surgical"]==1][f].astype(float).values
    pb=df[df["surgical"]==0][f].astype(float).values
    psd=np.sqrt((pa.var()+pb.var())/2) or 1
    bal.append({"var":f,
                "surg_pre":pa.mean(),"non_pre":pb.mean(),"smd_pre":(pa.mean()-pb.mean())/psd,
                "surg_post":a.mean(),"non_post":b.mean(),"smd_post":(a.mean()-b.mean())/sd})
bal_df=pd.DataFrame(bal); bal_df.to_csv(OUT/"balance_C_demosev.csv", index=False)
print("\nBalance (Analysis C):")
print(bal_df.round(3).to_string(index=False))

# McNemar
rows=[]
for col in PSYCH:
    a=df.loc[pairs_df["t_idx"],col].astype(int).values
    b=df.loc[pairs_df["c_idx"],col].astype(int).values
    n11=int(((a==1)&(b==1)).sum()); n10=int(((a==1)&(b==0)).sum())
    n01=int(((a==0)&(b==1)).sum()); n00=int(((a==0)&(b==0)).sum())
    disc=n10+n01
    res=mcnemar([[n11,n10],[n01,n00]], exact=(disc<25), correction=True)
    rows.append({"category":col,"n_pairs":len(pairs_df),
                 "surg_n":int(a.sum()),"surg_pct":100*a.mean(),
                 "nonsurg_n":int(b.sum()),"nonsurg_pct":100*b.mean(),
                 "surg_only":n10,"nonsurg_only":n01,
                 "stat":res.statistic,"p_value":res.pvalue})
mc=pd.DataFrame(rows); mc.to_csv(OUT/"mcnemar_C_demosev.csv", index=False)
print("\nMcNemar (Analysis C):")
print(mc[["category","surg_pct","nonsurg_pct","surg_only","nonsurg_only","p_value"]].round(4).to_string(index=False))

# ---------- Multivariable logreg (no race, no insurance) ----------
PRED = [
    ("surgical",     "Surgery (vs none)"),
    ("anchor_age",   "Age (per year)"),
    ("female",       "Female (vs male)"),
    ("n_asms",       "Unique ASMs (per drug)"),
    ("intractable",  "Drug-resistant code"),
    ("focal",        "Focal epilepsy"),
    ("se",           "Status epilepticus history"),
    ("n_epi_hadm",   "Epilepsy admissions (per +1)"),
]
cols=[v for v,_ in PRED]
Xr=sm.add_constant(df[cols].astype(float)); yr=df["any_psych"].astype(int)
m=sm.Logit(yr, Xr).fit(disp=False)
res=pd.DataFrame({
    "term":["(Intercept)"]+[lab for _,lab in PRED],
    "OR":   np.exp(m.params.values),
    "OR_lo":np.exp(m.conf_int()[0].values),
    "OR_hi":np.exp(m.conf_int()[1].values),
    "p":    m.pvalues.values,
})
res.to_csv(OUT/"logreg_C_demosev.csv", index=False)
print(f"\nLogreg C: n={len(df)}, events={int(yr.sum())}, pseudo-R2={m.prsquared:.3f}")
print(res.round(4).to_string(index=False))

# ---------- LaTeX ----------
def fmt_p(p): return "$<$0.001" if p<.001 else f"{p:.3f}"

# Balance + McNemar + logreg in one tex file
LAB = {"anchor_age":"Age, mean","female":"Female","n_asms":"Unique ASMs, mean",
       "intractable":"Drug-resistant code","focal":"Focal epilepsy",
       "se":"Status epilepticus history","n_epi_hadm":"Epilepsy admissions, mean"}
PCT = {"female","intractable","focal","se"}
def fv(v,k): return f"{v*100:.1f}\\%" if k in PCT else f"{v:.2f}"
def hi(s,c): return r"\textbf{"+s+r"}" if c else s

n_t=int(df["surgical"].sum()); n_c=int((df["surgical"]==0).sum()); n_p=len(pairs_df)

lines=[r"% Auto-generated by 10_psm_C_no_insurance.py",
r"\begin{table}[H]\centering",
rf"\caption{{Covariate balance, Analysis C (demographics + epilepsy severity only; no race, no insurance). 1:1 PSM, {n_p} matched pairs.}}",
r"\label{tab:balance_C}",
r"\footnotesize\setlength{\tabcolsep}{4pt}",
r"\begin{tabular}{lrrrrrr}\toprule",
rf" & \multicolumn{{3}}{{c}}{{\textbf{{Before matching}} ({n_t} surg \textbar{{}} {n_c} non-surg)}} & \multicolumn{{3}}{{c}}{{\textbf{{After matching}} ({n_p} pairs)}} \\",
r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
r"\textbf{Variable} & Surg & Non-surg & $|$SMD$|$ & Surg & Non-surg & $|$SMD$|$ \\",
r"\midrule",
]
for _,r in bal_df.iterrows():
    sp=abs(r["smd_pre"]); spo=abs(r["smd_post"])
    lines.append(f"\\quad {LAB[r['var']]} & {fv(r['surg_pre'],r['var'])} & {fv(r['non_pre'],r['var'])} & {hi(f'{sp:.3f}',sp>=0.10)} & {fv(r['surg_post'],r['var'])} & {fv(r['non_post'],r['var'])} & {hi(f'{spo:.3f}',spo>=0.10)} \\\\")
lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}",""]

# McNemar table
LBL_MC = {"any_psych":"Any psychiatric disorder","has_depression":"Depressive disorders",
          "has_substance_use":"Substance use disorders","has_anxiety":"Anxiety disorders",
          "has_bipolar":"Bipolar disorders","has_psychotic":"Psychotic disorders",
          "has_ptsd":"PTSD","has_suicidal_ideation":"Suicidal ideation",
          "has_adhd":"ADHD","has_ocd":"OCD","has_pnes":"PNES (not in composite)"}
mc_ord = mc.set_index("category").loc[list(LBL_MC.keys())].reset_index()
lines += [r"\begin{table}[H]\centering",
rf"\caption{{McNemar test on the {n_p} matched pairs from Analysis C (demographics + epilepsy severity only).}}",
r"\label{tab:mcnemar_C}",
r"\small",
r"\begin{tabular}{lccccc}\toprule",
r"\textbf{Category} & \textbf{Surg \%} & \textbf{Non-surg \%} & \textbf{Surg-only} & \textbf{Nonsurg-only} & \textbf{$p$} \\",
r"\midrule"]
for _,r in mc_ord.iterrows():
    star = r["p_value"] < 0.05
    p_str = fmt_p(r["p_value"])
    line = f"{LBL_MC[r['category']]} & {r['surg_pct']:.1f} & {r['nonsurg_pct']:.1f} & {int(r['surg_only'])} & {int(r['nonsurg_only'])} & {hi(p_str, star)} \\\\"
    lines.append(line)
lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}",""]

# Logreg table
lines += [r"\begin{table}[H]\centering",
r"\caption{Multivariable logistic regression for any psychiatric comorbidity --- Analysis C (no race, no insurance). Predictors: surgery, demographics, epilepsy severity.}",
r"\label{tab:logreg_C}",
r"\small",
r"\begin{tabular}{lccc}\toprule",
r"\textbf{Predictor} & \textbf{aOR} & \textbf{95\% CI} & \textbf{$p$} \\",
r"\midrule",
rf"\multicolumn{{4}}{{l}}{{\textit{{n = {len(df):,}; events (any psych) = {int(yr.sum()):,}; pseudo-$R^2$ = {m.prsquared:.3f}}}}}\\",
r"\addlinespace"]
for _,r in res.iterrows():
    if r["term"]=="(Intercept)": continue
    star = r["p"]<0.05
    or_ = f"{r['OR']:.2f}"; ci=f"({r['OR_lo']:.2f}--{r['OR_hi']:.2f})"; p_=fmt_p(r["p"])
    if star:
        lines.append(f"\\quad \\textbf{{{r['term']}}} & \\textbf{{{or_}}} & \\textbf{{{ci}}} & \\textbf{{{p_}}} \\\\")
    else:
        lines.append(f"\\quad {r['term']} & {or_} & {ci} & {p_} \\\\")
lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]

(ANA/"overleaf"/"sections_psm"/"analysis_C.tex").write_text("\n".join(lines))
print("\nwrote analysis_C.tex")
