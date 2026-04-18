"""Pre/post love plots from balance_prepost CSVs."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
OUT = Path("/Volumes/Niels 2/MIMIC/physionet.org/files/mimiciv/3.1/analysis/epilepsy_psych/psm_results")

def love(label, title, fname):
    d = pd.read_csv(OUT / f"balance_prepost_{label}.csv")
    d["pre"]  = d["SMD_pre"].abs()
    d["post"] = d["SMD_post"].abs()
    d = d.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, max(4, .35*len(d))))
    for i, r in d.iterrows():
        ax.plot([r["pre"], r["post"]], [y[i], y[i]], "-", color="lightgray", lw=1, zorder=1)
    ax.scatter(d["pre"],  y, s=55, color="#3a6fd9", label="Before matching", zorder=3)
    ax.scatter(d["post"], y, s=55, color="#d97a3a", marker="D", label="After matching", zorder=3)
    ax.axvline(0.10, ls="--", color="gray", lw=1)
    ax.axvline(0,    color="black", lw=.5)
    ax.set_yticks(y); ax.set_yticklabels(d["Variable"])
    ax.set_xlabel("|Standardized mean difference|")
    ax.set_title(title)
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / fname, dpi=150); plt.close()
    print("saved", fname)

love("A_full",        "Covariate balance, Analysis A (full cohort)",          "loveplot_A.png")
love("B_intractable", "Covariate balance, Analysis B (intractable-only)",     "loveplot_B.png")
