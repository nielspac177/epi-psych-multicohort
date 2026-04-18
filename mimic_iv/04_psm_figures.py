"""Figures for PSM analysis."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/Volumes/Niels 2/MIMIC/physionet.org/files/mimiciv/3.1/analysis/epilepsy_psych/psm_results")
LABELS = {
    "any_psych": "Any psychiatric disorder",
    "has_depression": "Depression", "has_anxiety": "Anxiety",
    "has_substance_use": "Substance use", "has_bipolar": "Bipolar",
    "has_psychotic": "Psychotic", "has_ptsd": "PTSD",
    "has_suicidal_ideation": "Suicidal ideation", "has_adhd": "ADHD",
    "has_ocd": "OCD", "has_pnes": "PNES",
}

def plot(label, title, fname):
    df = pd.read_csv(OUT / f"mcnemar_{label}.csv")
    df = df.set_index("category").loc[list(LABELS.keys())].reset_index()
    df["lab"] = df["category"].map(LABELS)
    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    w = 0.38
    ax.barh(y + w/2, df["surg_pct"], w, label="Surgical (matched)", color="#d97a3a")
    ax.barh(y - w/2, df["nonsurg_pct"], w, label="Non-surgical (matched)", color="#3a6fd9")
    for i, row in df.iterrows():
        yi = y[i]
        if pd.notna(row["p_value"]):
            star = "***" if row["p_value"] < .001 else "**" if row["p_value"] < .01 else "*" if row["p_value"] < .05 else ""
            txt = f"p={row['p_value']:.3f}{star}"
            ax.text(max(row["surg_pct"], row["nonsurg_pct"]) + 1, yi, txt, va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(df["lab"])
    ax.set_xlabel("Prevalence (%)"); ax.set_title(title)
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / fname, dpi=150); plt.close()
    print("saved", fname)

plot("A_full", f"PSM (Full Cohort): Surgical vs Non-Surgical (1:1, 243 pairs)\nMcNemar test", "psm_A_prevalence.png")
plot("B_intractable", f"PSM Sensitivity (Intractable Only): 60 pairs\nMcNemar test", "psm_B_prevalence.png")

# Balance / love plot
def love(label, fname):
    bal = pd.read_csv(OUT / f"balance_{label}.csv")
    fig, ax = plt.subplots(figsize=(7, max(4, .35*len(bal))))
    ax.scatter(bal["smd"].abs(), range(len(bal)), color="#d97a3a", zorder=3)
    ax.axvline(0.1, ls="--", color="gray"); ax.axvline(0, color="black", lw=.5)
    ax.set_yticks(range(len(bal))); ax.set_yticklabels(bal["var"])
    ax.set_xlabel("|Standardized mean difference|")
    ax.set_title(f"Covariate balance after PSM ({label})")
    ax.grid(axis="x", alpha=.3); plt.tight_layout(); plt.savefig(OUT / fname, dpi=150); plt.close()
    print("saved", fname)

love("A_full", "psm_A_balance.png")
love("B_intractable", "psm_B_balance.png")
