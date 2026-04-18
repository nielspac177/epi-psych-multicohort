"""Temporal-trend figures: SEPARATE plots for surgical and non-surgical.
Period assignment = period of FIRST epilepsy admission ('new patients').
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

ANA = Path("/Volumes/Niels 2/MIMIC/physionet.org/files/mimiciv/3.1/analysis/epilepsy_psych")
OUT = ANA / "psm_results"

df = pd.read_csv(ANA / "epilepsy_patient_cohort_psm.csv")
# Use existing year_bin or anchor_year_group as period proxy
period_col = "year_bin" if "year_bin" in df.columns else "anchor_year_group"
periods = sorted(df[period_col].dropna().unique())
print("periods:", periods)

CATS = {
    "any_psych": "Any psychiatric",
    "has_depression": "Depression",
    "has_anxiety": "Anxiety",
    "has_substance_use": "Substance use",
    "has_bipolar": "Bipolar",
    "has_psychotic": "Psychotic",
    "has_ptsd": "PTSD",
    "has_suicidal_ideation": "Suicidal ideation",
    "has_adhd": "ADHD",
}

def prevalence_by_period(sub):
    rows = []
    for p in periods:
        s = sub[sub[period_col] == p]
        n = len(s)
        if n == 0: continue
        row = {"period": p, "n": n}
        for c in CATS:
            row[c] = 100 * s[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)

def plot_any(sub, title, color, fname):
    d = prevalence_by_period(sub)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d["period"], d["any_psych"], "o-", color=color, lw=2, ms=8)
    for _, r in d.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["period"], r["any_psych"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("Prevalence of any psychiatric disorder (%)")
    ax.set_xlabel("Study period (de-identified)")
    ax.set_title(title); ax.grid(alpha=.3)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150); plt.close()
    print("saved", fname)

def plot_by_disorder(sub, title, fname):
    d = prevalence_by_period(sub)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for i, (c, lab) in enumerate(CATS.items()):
        if c == "any_psych": continue
        ax.plot(d["period"], d[c], "o-", label=lab, color=cmap(i % 10), lw=1.6, ms=6)
    ax.set_ylim(0, 75); ax.set_ylabel("Prevalence (%)")
    ax.set_xlabel("Study period (de-identified)")
    ax.set_title(title); ax.grid(alpha=.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=9)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150, bbox_inches="tight"); plt.close()
    print("saved", fname)

surg = df[df["surgical"] == 1]
nonsurg = df[df["surgical"] == 0]
print(f"surgical n={len(surg)}, non-surgical n={len(nonsurg)}")

plot_any(surg, "Any psychiatric comorbidity over time — Surgical patients\n(new patients per period)",
         "#d97a3a", "temporal_surg_any.png")
plot_any(nonsurg, "Any psychiatric comorbidity over time — Non-surgical patients\n(new patients per period)",
         "#3a6fd9", "temporal_nonsurg_any.png")
plot_by_disorder(surg, "Psychiatric comorbidities over time — Surgical patients (by disorder)",
                 "temporal_surg_by_disorder.png")
plot_by_disorder(nonsurg, "Psychiatric comorbidities over time — Non-surgical patients (by disorder)",
                 "temporal_nonsurg_by_disorder.png")

# Save underlying tables
prevalence_by_period(surg).to_csv(OUT / "temporal_surg.csv", index=False)
prevalence_by_period(nonsurg).to_csv(OUT / "temporal_nonsurg.csv", index=False)
print("done")
