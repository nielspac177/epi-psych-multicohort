"""
PSM analysis: surgical vs non-surgical epilepsy patients in MIMIC-IV.

Two analyses:
  (A) Full cohort, PSM 1:1 caliper 0.2*SD(logit PS) on:
      age, sex, insurance, race, n_asms, intractable (G40.x1), focal (G40.1/2),
      status epilepticus (G41.x), n_epilepsy_admissions
  (B) Sensitivity: restrict to intractable patients (G40.x1) in both arms,
      PSM 1:1 on age, sex, insurance, race, n_asms, focal, SE, n_admissions.

Outcome: McNemar test on each psychiatric category over matched pairs.
"""
import duckdb, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path("/Volumes/Niels 2/MIMIC/physionet.org/files/mimiciv/3.1")
ANA = ROOT / "analysis/epilepsy_psych"
OUT = ANA / "psm_results"
OUT.mkdir(exist_ok=True)

con = duckdb.connect()

# ---------- 1. Build augmented patient cohort ----------
print("Loading patient cohort...")
pat = pd.read_csv(ANA / "epilepsy_patient_cohort.csv")
pat = pat.rename(columns={"ever_had_surgery": "surgical"})
print(f"  n={len(pat)}, surgical={pat['surgical'].sum()}")

# Epilepsy hadm_ids per patient (need diagnoses + admissions)
print("Building epilepsy hadm map + ICD subtype flags...")
dx = con.execute(f"""
    SELECT subject_id, hadm_id, icd_code, icd_version
    FROM read_csv_auto('{ROOT}/hosp/diagnoses_icd.csv')
    WHERE subject_id IN (SELECT subject_id FROM pat)
""").df()

def epilepsy_match(c, v):
    c = str(c).strip()
    if v == 9:  return c.startswith("345")
    if v == 10: return c.startswith("G40")
    return False

dx["is_epi"] = [epilepsy_match(c, v) for c, v in zip(dx["icd_code"], dx["icd_version"])]
dx_epi = dx[dx["is_epi"]].copy()

# Intractable: ICD-10 G40.x1 / G40.x9 ; ICD-9 345.x1 (note: in ICD-9, 5th digit 1 = intractable)
def is_intractable(c, v):
    c = str(c).strip()
    if v == 10 and c.startswith("G40") and len(c) >= 5:
        # G40 + letter/digit + digit + intractable digit (1)
        return c[-1] in ("1",) or "intract" in c.lower()
    if v == 9 and c.startswith("345") and len(c) >= 5:
        return c[-1] == "1"
    return False

def is_focal(c, v):
    c = str(c).strip()
    if v == 10:
        # G40.0 localization-related idiopathic, G40.1 simple partial, G40.2 complex partial
        return c.startswith(("G400", "G401", "G402"))
    if v == 9:
        return c.startswith(("3454", "3455"))  # 345.4 partial w/ impair, 345.5 partial w/o
    return False

def is_se(c, v):
    c = str(c).strip()
    if v == 10: return c.startswith("G41")
    if v == 9:  return c.startswith("3453")  # 345.3 grand mal status
    return False

dx_epi["intractable"] = [is_intractable(c, v) for c, v in zip(dx_epi["icd_code"], dx_epi["icd_version"])]
dx_epi["focal"]       = [is_focal(c, v)       for c, v in zip(dx_epi["icd_code"], dx_epi["icd_version"])]
dx_epi["se"]          = [is_se(c, v)          for c, v in zip(dx_epi["icd_code"], dx_epi["icd_version"])]

flags = dx_epi.groupby("subject_id").agg(
    intractable=("intractable", "any"),
    focal=("focal", "any"),
    se=("se", "any"),
    n_epi_hadm=("hadm_id", "nunique"),
).reset_index()
print(f"  intractable={flags['intractable'].sum()}, focal={flags['focal'].sum()}, SE={flags['se'].sum()}")

# ---------- 2. ASM count from prescriptions ----------
print("Counting ASMs per patient...")
ASM_PATTERNS = [
    "levetiracetam", "keppra", "lacosamide", "vimpat", "valproate", "valproic",
    "depakote", "depakene", "phenytoin", "dilantin", "fosphenytoin", "cerebyx",
    "carbamazepine", "tegretol", "oxcarbazepine", "trileptal", "lamotrigine",
    "lamictal", "topiramate", "topamax", "zonisamide", "zonegran", "clobazam",
    "onfi", "perampanel", "fycompa", "brivaracetam", "briviact", "cenobamate",
    "xcopri", "phenobarbital", "primidone", "mysoline", "ethosuximide",
    "zarontin", "gabapentin", "neurontin", "pregabalin", "lyrica", "vigabatrin",
    "sabril", "rufinamide", "banzel", "felbamate", "felbatol", "tiagabine",
    "gabitril", "eslicarbazepine", "aptiom", "clonazepam", "klonopin",
]
like = " OR ".join([f"LOWER(drug) LIKE '%{p}%'" for p in ASM_PATTERNS])
asm = con.execute(f"""
    SELECT subject_id, COUNT(DISTINCT LOWER(TRIM(drug))) AS n_asms_raw,
           COUNT(DISTINCT LOWER(SPLIT_PART(TRIM(drug),' ',1))) AS n_asms
    FROM read_csv_auto('{ROOT}/hosp/prescriptions.csv')
    WHERE subject_id IN (SELECT subject_id FROM pat)
      AND ({like})
    GROUP BY subject_id
""").df()
print(f"  patients with >=1 ASM record: {len(asm)}")

# ---------- 3. Merge ----------
df = pat.merge(flags, on="subject_id", how="left").merge(asm[["subject_id","n_asms"]], on="subject_id", how="left")
for c, v in [("intractable",False),("focal",False),("se",False),("n_epi_hadm",1),("n_asms",0)]:
    df[c] = df[c].fillna(v)
df["intractable"] = df["intractable"].astype(int)
df["focal"] = df["focal"].astype(int)
df["se"] = df["se"].astype(int)
df["female"] = (df["gender"] == "F").astype(int)

# Insurance / race -> dummies
INS = ["Medicare","Medicaid","Private","Other"]
df["insurance"] = df["insurance"].fillna("Other")
df.loc[~df["insurance"].isin(INS), "insurance"] = "Other"
ins_d = pd.get_dummies(df["insurance"], prefix="ins").astype(int)

df["race_grp"] = df["race"].fillna("UNKNOWN").str.upper().str[:5]
race_top = df["race_grp"].value_counts().head(5).index.tolist()
df["race_grp"] = df["race_grp"].where(df["race_grp"].isin(race_top), "OTHER")
race_d = pd.get_dummies(df["race_grp"], prefix="race").astype(int)

df = pd.concat([df, ins_d, race_d], axis=1)
df.to_csv(ANA / "epilepsy_patient_cohort_psm.csv", index=False)
print(f"Saved augmented cohort: {df.shape}")

# ---------- 4. PSM helper ----------
PSYCH = ["has_depression","has_bipolar","has_anxiety","has_ptsd","has_ocd",
         "has_psychotic","has_adhd","has_substance_use","has_suicidal_ideation",
         "has_pnes","any_psych"]

def run_psm(data, label, ps_features):
    d = data.copy().reset_index(drop=True)
    treat = d["surgical"].astype(int).values
    n_t, n_c = treat.sum(), (1-treat).sum()
    print(f"\n=== PSM [{label}] n_surg={n_t}, n_nonsurg={n_c} ===")
    X = d[ps_features].astype(float).values
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000, solver="lbfgs").fit(Xs, treat)
    ps = lr.predict_proba(Xs)[:, 1]
    logit = np.log(np.clip(ps, 1e-6, 1-1e-6) / np.clip(1-ps, 1e-6, 1-1e-6))
    caliper = 0.2 * np.std(logit)
    print(f"  caliper (logit PS) = {caliper:.4f}")

    t_idx = np.where(treat == 1)[0]
    c_idx = np.where(treat == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(logit[c_idx].reshape(-1,1))
    used = set()
    pairs = []
    rng = np.random.default_rng(42)
    order = rng.permutation(t_idx)
    for ti in order:
        # query nearest unused control
        dists, idxs = nn.kneighbors([[logit[ti]]], n_neighbors=min(20, len(c_idx)))
        for dist, j in zip(dists[0], idxs[0]):
            ci = c_idx[j]
            if ci in used: continue
            if dist > caliper: break
            used.add(ci); pairs.append((ti, ci, dist)); break

    print(f"  matched pairs: {len(pairs)} / {n_t} treated")
    if not pairs:
        return None
    pairs_df = pd.DataFrame(pairs, columns=["t_idx","c_idx","dist"])
    pairs_df["surg_subject"] = d.loc[pairs_df["t_idx"],"subject_id"].values
    pairs_df["ctrl_subject"] = d.loc[pairs_df["c_idx"],"subject_id"].values
    pairs_df.to_csv(OUT / f"matched_pairs_{label}.csv", index=False)

    # Balance check (SMD)
    bal = []
    for f in ps_features:
        a = d.loc[pairs_df["t_idx"], f].astype(float).values
        b = d.loc[pairs_df["c_idx"], f].astype(float).values
        sd = np.sqrt((a.var()+b.var())/2) or 1
        bal.append({"var": f, "surg_mean": a.mean(), "ctrl_mean": b.mean(),
                    "smd": (a.mean()-b.mean())/sd})
    bal_df = pd.DataFrame(bal); bal_df.to_csv(OUT / f"balance_{label}.csv", index=False)
    print(bal_df.round(3).to_string(index=False))

    # McNemar per psych category
    rows = []
    for col in PSYCH:
        a = d.loc[pairs_df["t_idx"], col].astype(int).values  # surgical
        b = d.loc[pairs_df["c_idx"], col].astype(int).values  # non-surg
        # 2x2: rows = surg (0/1), cols = nonsurg (0/1)
        n11 = int(((a==1)&(b==1)).sum()); n10 = int(((a==1)&(b==0)).sum())
        n01 = int(((a==0)&(b==1)).sum()); n00 = int(((a==0)&(b==0)).sum())
        table = [[n11,n10],[n01,n00]]
        # McNemar exact if discordant small
        disc = n10 + n01
        try:
            res = mcnemar(table, exact=(disc < 25), correction=True)
            pval = res.pvalue; stat = res.statistic
        except Exception:
            pval = np.nan; stat = np.nan
        rows.append({
            "category": col,
            "n_pairs": len(pairs_df),
            "surg_n": int(a.sum()), "surg_pct": 100*a.mean(),
            "nonsurg_n": int(b.sum()), "nonsurg_pct": 100*b.mean(),
            "concordant_pos": n11, "concordant_neg": n00,
            "surg_only": n10, "nonsurg_only": n01,
            "mcnemar_stat": stat, "p_value": pval,
        })
    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT / f"mcnemar_{label}.csv", index=False)
    print(res_df[["category","surg_pct","nonsurg_pct","surg_only","nonsurg_only","p_value"]].round(4).to_string(index=False))
    return res_df

# ---------- 5. Run analyses ----------
PS_A = ["anchor_age","female","n_asms","intractable","focal","se","n_epi_hadm"] \
       + [c for c in df.columns if c.startswith("ins_")]
res_A = run_psm(df, "A_full", PS_A)

# Sensitivity B: intractable only
df_B = df[df["intractable"] == 1].copy()
print(f"\nIntractable subset: n={len(df_B)}, surgical={df_B['surgical'].sum()}")
PS_B = [v for v in PS_A if v != "intractable"]
res_B = run_psm(df_B, "B_intractable", PS_B)

# ---------- 6. PNES verification ----------
print("\n=== PNES verification ===")
pnes_codes_v10 = ["F445"]; pnes_codes_v9 = ["30011"]
pnes_dx = con.execute(f"""
    SELECT subject_id, COUNT(*) AS n_codes
    FROM read_csv_auto('{ROOT}/hosp/diagnoses_icd.csv')
    WHERE (icd_version=10 AND icd_code LIKE 'F445%')
       OR (icd_version=9  AND icd_code LIKE '30011%')
    GROUP BY subject_id
""").df()
print(f"  Total patients in MIMIC w/ PNES code: {len(pnes_dx)}")
in_cohort = pnes_dx[pnes_dx["subject_id"].isin(df["subject_id"])]
print(f"  In epilepsy cohort: {len(in_cohort)}")
surg_pnes = df[(df["surgical"]==1) & (df["has_pnes"]==1)]
nonsurg_pnes = df[(df["surgical"]==0) & (df["has_pnes"]==1)]
print(f"  Surgical w/ PNES: {len(surg_pnes)} / {df['surgical'].sum()}")
print(f"  Non-surgical w/ PNES: {len(nonsurg_pnes)} / {(df['surgical']==0).sum()}")

print("\nDone. Outputs in:", OUT)
