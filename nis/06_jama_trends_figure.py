"""
JAMA Neurology-style two-panel temporal trends figure.
Panel A: Surgical cohort. Panel B: Non-surgical cohort.
Three disorders: Any psychiatric (excl. PNES), Depression, Anxiety.
Legend placed at the bottom.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path("/Volumes/Niels 2/NIS_new_version/NIS_epilepsy_psych")
PUB = Path("/Volumes/Niels 2/NIS_new_version/NIS_epy_surg_pub_figures")
PUB.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df = pd.read_csv(ROOT / "tables/trends_by_year.csv")
ors = pd.read_csv(ROOT / "tables/trends_or_per_year.csv")

DISORDERS = [
    ("Any psychiatric (excl. PNES)", "Any psychiatric disorder", "#1F3A68"),
    ("Depression",                    "Depression",               "#B2182B"),
    ("Anxiety",                       "Anxiety",                  "#2E7D32"),
]

MARKERS = {"Any psychiatric disorder": "o",
           "Depression":              "s",
           "Anxiety":                 "^"}

def fmt_p(p):
    if p < 0.001:
        return "P<.001"
    if p < 0.01:
        return f"P=.{int(round(p*1000)):03d}"
    return f"P=.{int(round(p*100)):02d}"

def fmt_or(x):
    return f"{x:.2f}"

def plot_panel(ax, group_name, panel_letter, panel_title):
    sub = df[df["group"] == group_name].copy()
    for disorder, label, color in DISORDERS:
        s = sub[sub["disorder"] == disorder].sort_values("year")
        if s.empty:
            continue
        years = s["year"].values
        prev = s["prev"].values * 100
        lo = s["lo"].values * 100
        hi = s["hi"].values * 100
        ax.fill_between(years, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(years, prev, color=color, linewidth=1.6,
                marker=MARKERS[label], markersize=4.5,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.7, label=label)

    any_or = ors[(ors["group"] == group_name) &
                 (ors["disorder"] == "Any psychiatric (excl. PNES)")]
    if not any_or.empty:
        or_val = any_or["or_per_year"].values[0]
        p_val = any_or["p"].values[0]
        ann = f"OR/yr = {fmt_or(or_val)}\n{fmt_p(p_val)}"
        ax.annotate(
            ann,
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top",
            fontsize=8.5, color="#1F3A68",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white",
                      edgecolor="#1F3A68",
                      linewidth=0.6),
        )

    ax.set_xlabel("Year")
    ax.set_xticks(range(2012, 2021, 2))
    ax.set_xlim(2011.5, 2020.5)
    ax.set_ylim(0, 55)
    ax.set_yticks(range(0, 56, 10))
    ax.tick_params(axis="both", direction="out", length=3.5, pad=2)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.3,
            color="#CCCCCC", alpha=0.7)
    ax.set_axisbelow(True)
    ax.text(-0.13, 1.06, panel_letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")
    ax.set_title(panel_title, fontsize=10.5, fontweight="bold",
                 loc="center", pad=6)


fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6), sharey=True)
plot_panel(axes[0], "Surgical",     "A", "Surgical Patients")
plot_panel(axes[1], "Non-surgical", "B", "Non-Surgical Patients")

axes[0].set_ylabel("Survey-Weighted Prevalence, %")

handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels,
                 loc="lower center",
                 bbox_to_anchor=(0.5, -0.01),
                 ncol=3,
                 frameon=False,
                 handlelength=2.4,
                 columnspacing=2.2,
                 handletextpad=0.7)

fig.suptitle(
    "Temporal Trends in Psychiatric Comorbidity Among Epilepsy Patients, NIS 2012-2020",
    fontsize=11.5, fontweight="bold", y=0.995,
)

fig.tight_layout(rect=[0, 0.06, 1, 0.96])

png_main = ROOT / "figures" / "fig_trends_jama.png"
fig.savefig(png_main, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(ROOT / "figures" / "fig_trends_jama.pdf",
            bbox_inches="tight", facecolor="white")

png_pub = PUB / "fig_trends_jama.png"
fig.savefig(png_pub, dpi=600, bbox_inches="tight", facecolor="white")

tmp_png = PUB / "_tmp_jama.png"
fig.savefig(tmp_png, dpi=600, bbox_inches="tight", facecolor="white")
im = Image.open(tmp_png).convert("RGB")
tiff_pub = PUB / "fig_trends_jama.tif"
im.save(tiff_pub, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
tmp_png.unlink()

plt.close(fig)

print("Saved:")
for p in [png_main, ROOT / "figures" / "fig_trends_jama.pdf",
          png_pub, tiff_pub]:
    print(" ", p)
