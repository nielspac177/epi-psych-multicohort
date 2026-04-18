#!/usr/bin/env python3
"""
Deprivation Indices → Epilepsy Surgery Outcomes
================================================
Tests associations between neighborhood deprivation indices and:
  1. Seizure freedom at last follow-up
  2. Favorable seizure outcome (Engel I-II)
  3. Engel class I vs II-IV
  4. Surgical complications
  5. ASM reduction / discontinuation
  6. Delta seizure frequency
  7. Time to subsequent treatment
  8. Treatment type (resection vs neuromodulation)
  9. Geospatial analysis of seizure outcomes
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import warnings, os, json
warnings.filterwarnings("ignore")

OUTPUT_DIR = "results_epilepsy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv("results/merged_deprivation_psych_data.csv")
df["zip_clean"] = df["zip_clean"].astype(str).str.split(".").str[0].str.zfill(5)
df.loc[df["zip_clean"].str.contains("nan", na=True), "zip_clean"] = np.nan

# Rename columns for formula compatibility
rename = {}
for c in df.columns:
    if "Male=0" in c: rename[c] = "female"
    elif c == "Age at surgery": rename[c] = "age_at_surgery"
    elif "Duration of epilepsy" in c: rename[c] = "epilepsy_duration"
    elif "Preop # AEDs" == c: rename[c] = "preop_asms"
    elif "Seizure frequency" in c and "per month)" in c and "follow" not in c.lower():
        rename[c] = "preop_seizure_freq"
    elif c == "GTCs? 0=no, 1= yes": rename[c] = "gtcs"
    elif c == "Preop MRI abnormal? 0=no, 1=yes": rename[c] = "mri_abnormal"
df = df.rename(columns=rename)
print(f"  {len(df)} patients loaded")

# ============================================================
# CREATE EPILEPSY OUTCOME VARIABLES
# ============================================================
print("\nCreating outcome variables...")

# 1. Seizure freedom
df["seizure_free"] = (df["Seizure-free at last follow-up?"] == "y").astype(int)
df.loc[df["Seizure-free at last follow-up?"].isna(), "seizure_free"] = np.nan

# 2. Favorable seizure outcome
df["favorable_outcome"] = (df["favorable seizure outcomes"] == "yes").astype(int)
df.loc[df["favorable seizure outcomes"].isna(), "favorable_outcome"] = np.nan

# 3. Engel I
df["engel_I"] = (df["Engel score simplified"] == "I").astype(int)
df.loc[df["Engel score simplified"].isna(), "engel_I"] = np.nan

# 4. Engel I-II (good) vs III-IV (poor)
df["engel_good"] = df["Engel score simplified"].isin(["I", "II"]).astype(int)
df.loc[df["Engel score simplified"].isna(), "engel_good"] = np.nan

# 5. Any complication
df["any_complication"] = df["ALL COMPLICATIONS"].astype(int)

# 6. ASM reduced
df["asm_reduced"] = (df["ASM reduced - all"] == "YES").astype(int)
df.loc[df["ASM reduced - all"].isna(), "asm_reduced"] = np.nan

# 7. ASM discontinued
df["asm_discontinued"] = (df["ASM discontinued - all"] == "YES").astype(int)
df.loc[df["ASM discontinued - all"].isna(), "asm_discontinued"] = np.nan

# 8. Delta seizure frequency (continuous, negative = improvement)
df["delta_sz_freq"] = df["Delta Seizure Frequency"]

# 9. Time to treatment (days)
df["time_to_tx"] = df["Time to treatment"]

# 10. Resection vs neuromodulation (among those with subsequent treatment)
df["resection_vs_neuromod"] = np.nan
df.loc[df["Subsequent Resection vs. Neuromod"] == "RESLITT", "resection_vs_neuromod"] = 1
df.loc[df["Subsequent Resection vs. Neuromod"] == "NEUROMOD", "resection_vs_neuromod"] = 0

# 11. Subsequent treatment received
df["subsequent_tx"] = (df["Subsequent treatment? (y/n)"] == "y").astype(int)

# 12. Length of follow-up
df["followup_years"] = df["Length of follow-up "]

# Print summary
outcomes_summary = {
    "Seizure-free": ("seizure_free", "binary"),
    "Favorable outcome": ("favorable_outcome", "binary"),
    "Engel I": ("engel_I", "binary"),
    "Engel I-II": ("engel_good", "binary"),
    "Any complication": ("any_complication", "binary"),
    "ASM reduced": ("asm_reduced", "binary"),
    "ASM discontinued": ("asm_discontinued", "binary"),
    "Delta seizure freq": ("delta_sz_freq", "continuous"),
    "Subsequent treatment": ("subsequent_tx", "binary"),
    "Resection vs neuromod": ("resection_vs_neuromod", "binary"),
}

for label, (var, vtype) in outcomes_summary.items():
    valid = df[var].dropna()
    if vtype == "binary":
        print(f"  {label}: {int(valid.sum())}/{len(valid)} ({valid.mean()*100:.1f}%)")
    else:
        print(f"  {label}: mean={valid.mean():.1f}, median={valid.median():.1f}, n={len(valid)}")

# ============================================================
# MAIN ANALYSIS: DEPRIVATION INDICES → EPILEPSY OUTCOMES
# ============================================================
print("\n" + "=" * 70)
print("DEPRIVATION INDICES → EPILEPSY OUTCOMES")
print("=" * 70)

index_vars = [
    "SDI", "SVI_overall", "SVI_theme1_socioeconomic", "SVI_theme2_hh_composition",
    "SVI_theme3_minority_language", "SVI_theme4_housing_transport",
    "ICE_income", "ICE_race_income", "DCI_proxy",
]

covariates = ["female", "age_at_surgery", "epilepsy_duration"]

binary_outcomes = [
    ("seizure_free", "Seizure-free at last follow-up"),
    ("favorable_outcome", "Favorable seizure outcome (Engel I-II)"),
    ("engel_I", "Engel class I"),
    ("any_complication", "Any complication"),
    ("asm_reduced", "ASM reduced"),
    ("asm_discontinued", "ASM discontinued"),
    ("subsequent_tx", "Subsequent treatment received"),
    ("resection_vs_neuromod", "Resection vs neuromodulation"),
]

all_results = []

for outcome_var, outcome_label in binary_outcomes:
    n_events = df[outcome_var].dropna().sum()
    n_total = df[outcome_var].dropna().shape[0]
    print(f"\n--- {outcome_label} (events={int(n_events)}/{n_total}) ---")

    if n_events < 10 or (n_total - n_events) < 10:
        print("  SKIPPED: insufficient events for logistic regression")
        continue

    for idx_var in index_vars:
        analysis_cols = [outcome_var, idx_var] + covariates
        subset = df[analysis_cols].dropna()
        if len(subset) < 20:
            continue

        subset[f"{idx_var}_z"] = (subset[idx_var] - subset[idx_var].mean()) / subset[idx_var].std()

        # Unadjusted
        try:
            m1 = smf.logit(f"{outcome_var} ~ {idx_var}_z", data=subset).fit(disp=0)
            or1 = np.exp(m1.params[f"{idx_var}_z"])
            ci1 = np.exp(m1.conf_int().loc[f"{idx_var}_z"])
            p1 = m1.pvalues[f"{idx_var}_z"]
        except:
            continue

        # Adjusted
        try:
            formula = f"{outcome_var} ~ {idx_var}_z + " + " + ".join(covariates)
            m2 = smf.logit(formula, data=subset).fit(disp=0)
            or2 = np.exp(m2.params[f"{idx_var}_z"])
            ci2 = np.exp(m2.conf_int().loc[f"{idx_var}_z"])
            p2 = m2.pvalues[f"{idx_var}_z"]
        except:
            or2 = ci2 = p2 = np.nan
            ci2 = pd.Series([np.nan, np.nan])

        all_results.append({
            "Outcome": outcome_label, "Index": idx_var, "N": len(subset),
            "OR_unadj": round(or1, 3),
            "CI95_unadj": f"({ci1.iloc[0]:.3f}-{ci1.iloc[1]:.3f})",
            "p_unadj": round(p1, 4),
            "OR_adj": round(or2, 3) if not np.isnan(or2) else np.nan,
            "CI95_adj": f"({ci2.iloc[0]:.3f}-{ci2.iloc[1]:.3f})" if not any(pd.isna(ci2)) else np.nan,
            "p_adj": round(p2, 4) if not np.isnan(p2) else np.nan,
        })

        if p2 < 0.05:
            sig = " *"
        else:
            sig = ""
        if idx_var in ["SDI", "SVI_overall", "ICE_income", "ICE_race_income", "DCI_proxy"] or p1 < 0.1 or p2 < 0.1:
            print(f"  {idx_var}: OR_adj={or2:.3f} ({ci2.iloc[0]:.3f}-{ci2.iloc[1]:.3f}), p={p2:.4f}{sig}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{OUTPUT_DIR}/epilepsy_association_results.csv", index=False)

# Print significant results
sig = results_df[results_df["p_adj"] < 0.05]
if len(sig) > 0:
    print(f"\n*** {len(sig)} significant associations (p<0.05, adjusted) ***")
    print(sig[["Outcome", "Index", "OR_adj", "CI95_adj", "p_adj"]].to_string(index=False))
else:
    print("\n*** No significant associations at p<0.05 (adjusted) ***")

# Also check borderline
borderline = results_df[(results_df["p_adj"] >= 0.05) & (results_df["p_adj"] < 0.10)]
if len(borderline) > 0:
    print(f"\n*** {len(borderline)} borderline associations (0.05 <= p < 0.10, adjusted) ***")
    print(borderline[["Outcome", "Index", "OR_adj", "CI95_adj", "p_adj"]].to_string(index=False))

# ============================================================
# CONTINUOUS OUTCOMES: LINEAR REGRESSION
# ============================================================
print("\n" + "=" * 70)
print("LINEAR REGRESSION: Deprivation → Continuous Epilepsy Outcomes")
print("=" * 70)

continuous_outcomes = [
    ("delta_sz_freq", "Delta seizure frequency"),
    ("time_to_tx", "Time to treatment (days)"),
    ("followup_years", "Length of follow-up (years)"),
]

linear_results = []
for outcome_var, outcome_label in continuous_outcomes:
    valid_n = df[outcome_var].dropna().shape[0]
    print(f"\n--- {outcome_label} (n={valid_n}) ---")

    for idx_var in ["SDI", "SVI_overall", "ICE_income", "ICE_race_income", "DCI_proxy"]:
        analysis_cols = [outcome_var, idx_var] + covariates
        subset = df[analysis_cols].dropna()
        if len(subset) < 20:
            continue

        subset[f"{idx_var}_z"] = (subset[idx_var] - subset[idx_var].mean()) / subset[idx_var].std()

        try:
            formula = f"{outcome_var} ~ {idx_var}_z + " + " + ".join(covariates)
            m = smf.ols(formula, data=subset).fit()
            coef = m.params[f"{idx_var}_z"]
            ci = m.conf_int().loc[f"{idx_var}_z"]
            p = m.pvalues[f"{idx_var}_z"]

            sig = " *" if p < 0.05 else ""
            print(f"  {idx_var}: beta={coef:.2f} ({ci.iloc[0]:.2f}-{ci.iloc[1]:.2f}), p={p:.4f}{sig}")

            linear_results.append({
                "Outcome": outcome_label, "Index": idx_var, "N": len(subset),
                "Beta": round(coef, 3), "CI95": f"({ci.iloc[0]:.3f}-{ci.iloc[1]:.3f})",
                "p": round(p, 4),
            })
        except:
            pass

pd.DataFrame(linear_results).to_csv(f"{OUTPUT_DIR}/linear_regression_results.csv", index=False)

# ============================================================
# DISTANCE → EPILEPSY OUTCOMES
# ============================================================
print("\n" + "=" * 70)
print("DISTANCE TO BWH → EPILEPSY OUTCOMES")
print("=" * 70)

dist_results = []
for outcome_var, outcome_label in binary_outcomes:
    n_events = df[outcome_var].dropna().sum()
    n_total = df[outcome_var].dropna().shape[0]
    if n_events < 10 or (n_total - n_events) < 10:
        continue

    cols = [outcome_var, "distance_to_BWH_miles"] + covariates
    subset = df[cols].dropna()
    if len(subset) < 20:
        continue

    subset["dist_z"] = (subset["distance_to_BWH_miles"] - subset["distance_to_BWH_miles"].mean()) / subset["distance_to_BWH_miles"].std()

    try:
        m = smf.logit(f"{outcome_var} ~ dist_z + " + " + ".join(covariates), data=subset).fit(disp=0)
        orv = np.exp(m.params["dist_z"])
        ci = np.exp(m.conf_int().loc["dist_z"])
        p = m.pvalues["dist_z"]
        sig = " *" if p < 0.05 else ""
        print(f"  {outcome_label}: OR={orv:.3f} ({ci.iloc[0]:.3f}-{ci.iloc[1]:.3f}), p={p:.4f}{sig}")

        dist_results.append({
            "Outcome": outcome_label, "N": len(subset),
            "OR_adj": round(orv, 3), "CI95": f"({ci.iloc[0]:.3f}-{ci.iloc[1]:.3f})", "p": round(p, 4),
        })
    except:
        pass

pd.DataFrame(dist_results).to_csv(f"{OUTPUT_DIR}/distance_epilepsy_results.csv", index=False)

# ============================================================
# RUCA × EPILEPSY OUTCOMES
# ============================================================
print("\n" + "=" * 70)
print("RUCA × EPILEPSY OUTCOMES")
print("=" * 70)

if "RUCA_category" in df.columns:
    for outcome_var, outcome_label in [
        ("seizure_free", "Seizure-free"),
        ("favorable_outcome", "Favorable outcome"),
        ("any_complication", "Any complication"),
    ]:
        subset = df[["RUCA_category", outcome_var]].dropna()
        if len(subset) < 20:
            continue
        ct = pd.crosstab(subset["RUCA_category"], subset[outcome_var])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            rates = subset.groupby("RUCA_category")[outcome_var].mean()
            print(f"\n  {outcome_label} (chi2={chi2:.2f}, p={p:.4f}):")
            for cat, rate in rates.items():
                n = (subset["RUCA_category"] == cat).sum()
                print(f"    {cat}: {rate:.1%} (n={n})")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n  Creating visualizations...")
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Forest plot: deprivation indices → seizure outcomes
key_outcomes = ["Seizure-free at last follow-up", "Favorable seizure outcome (Engel I-II)",
                "Engel class I", "Any complication", "ASM reduced"]
key_indices = ["SDI", "SVI_overall", "ICE_income", "ICE_race_income", "DCI_proxy"]

subset_results = results_df[
    results_df["Outcome"].isin(key_outcomes) &
    results_df["Index"].isin(key_indices)
].copy()

if len(subset_results) > 0:
    fig, axes = plt.subplots(1, len(key_outcomes), figsize=(5 * len(key_outcomes), 6), sharey=True)
    if len(key_outcomes) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, key_outcomes):
        sub = subset_results[subset_results["Outcome"] == outcome].copy()
        if len(sub) == 0:
            continue

        sub["CI_lo"] = sub["CI95_adj"].str.extract(r"\(([0-9.]+)").astype(float)
        sub["CI_hi"] = sub["CI95_adj"].str.extract(r"-([0-9.]+)\)").astype(float)

        y_pos = range(len(sub))
        ax.errorbar(
            sub["OR_adj"], y_pos,
            xerr=[sub["OR_adj"] - sub["CI_lo"], sub["CI_hi"] - sub["OR_adj"]],
            fmt="o", color="steelblue", capsize=3, markersize=6
        )
        ax.axvline(x=1, color="red", linestyle="--", alpha=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sub["Index"])
        ax.set_xlabel("OR (95% CI)")
        short_title = outcome.replace("at last follow-up", "").replace("(Engel I-II)", "").strip()
        ax.set_title(short_title, fontsize=10)

        for i, (_, row) in enumerate(sub.iterrows()):
            p_str = f"p={row['p_adj']:.3f}" if row['p_adj'] >= 0.001 else "p<.001"
            sig = " *" if row['p_adj'] < 0.05 else ""
            ax.annotate(f"{p_str}{sig}", xy=(max(row["CI_hi"] + 0.02, 1.3), i), fontsize=7, va="center")

    plt.suptitle("Deprivation Indices → Epilepsy Surgery Outcomes\n(OR per 1-SD increase, adjusted)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/forest_plot_epilepsy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/forest_plot_epilepsy.png")

# Seizure freedom by SDI quartile
if "SDI" in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    for ax, (out, label) in zip(axes, [
        ("seizure_free", "Seizure-Free"),
        ("favorable_outcome", "Favorable Outcome"),
        ("any_complication", "Any Complication"),
    ]):
        subset = df[["SDI", out]].dropna()
        subset["SDI_Q"] = pd.qcut(subset["SDI"], 4,
                                   labels=["Q1\n(least\ndeprived)", "Q2", "Q3", "Q4\n(most\ndeprived)"])
        rates = subset.groupby("SDI_Q", observed=True)[out].agg(["mean", "count"])
        rates["se"] = np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["count"])
        ax.bar(range(4), rates["mean"], yerr=1.96 * rates["se"],
               color=colors, edgecolor="white", capsize=5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(rates.index)
        ax.set_ylabel("Proportion")
        ax.set_title(f"{label} by SDI Quartile")
        ax.set_ylim(0, 1)
        for i, (_, row) in enumerate(rates.iterrows()):
            ax.text(i, row["mean"] + 1.96 * row["se"] + 0.03,
                    f"n={int(row['count'])}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/epilepsy_by_sdi_quartile.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/epilepsy_by_sdi_quartile.png")

# Seizure freedom by ICE quartile
if "ICE_race_income" in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_ice = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]

    for ax, (out, label) in zip(axes, [
        ("seizure_free", "Seizure-Free"),
        ("favorable_outcome", "Favorable Outcome"),
        ("any_complication", "Any Complication"),
    ]):
        subset = df[["ICE_race_income", out]].dropna()
        subset["ICE_Q"] = pd.qcut(subset["ICE_race_income"], 4,
                                   labels=["Q1\n(most\ndeprived)", "Q2", "Q3", "Q4\n(most\nprivileged)"])
        rates = subset.groupby("ICE_Q", observed=True)[out].agg(["mean", "count"])
        rates["se"] = np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["count"])
        ax.bar(range(4), rates["mean"], yerr=1.96 * rates["se"],
               color=colors_ice, edgecolor="white", capsize=5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(rates.index)
        ax.set_ylabel("Proportion")
        ax.set_title(f"{label} by ICE Race-Income Quartile")
        ax.set_ylim(0, 1)
        for i, (_, row) in enumerate(rates.iterrows()):
            ax.text(i, row["mean"] + 1.96 * row["se"] + 0.03,
                    f"n={int(row['count'])}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/epilepsy_by_ice_quartile.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/epilepsy_by_ice_quartile.png")

# Distance → seizure outcomes quartile
if "distance_to_BWH_miles" in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    for ax, (out, label) in zip(axes, [
        ("seizure_free", "Seizure-Free"),
        ("favorable_outcome", "Favorable Outcome"),
        ("any_complication", "Any Complication"),
    ]):
        subset = df[["distance_to_BWH_miles", out]].dropna()
        subset["dist_Q"] = pd.qcut(subset["distance_to_BWH_miles"], 4,
                                    labels=["Q1\n(closest)", "Q2", "Q3", "Q4\n(farthest)"])
        rates = subset.groupby("dist_Q", observed=True)[out].agg(["mean", "count"])
        rates["se"] = np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["count"])
        ax.bar(range(4), rates["mean"], yerr=1.96 * rates["se"],
               color=colors, edgecolor="white", capsize=5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(rates.index)
        ax.set_ylabel("Proportion")
        ax.set_title(f"{label} by Distance to BWH")
        ax.set_ylim(0, 1)
        for i, (_, row) in enumerate(rates.iterrows()):
            ax.text(i, row["mean"] + 1.96 * row["se"] + 0.03,
                    f"n={int(row['count'])}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/epilepsy_by_distance_quartile.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/epilepsy_by_distance_quartile.png")

# ============================================================
# GEOSPATIAL: MORAN'S I FOR SEIZURE OUTCOMES
# ============================================================
print("\n" + "=" * 70)
print("GEOSPATIAL: MORAN'S I FOR SEIZURE OUTCOMES")
print("=" * 70)

try:
    import geopandas as gpd
    from libpysal.weights import KNN
    from esda.moran import Moran

    print("  Loading ZCTA shapefiles...")
    zcta_gdf = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2022/ZCTA520/tl_2022_us_zcta520.zip")
    zcta_gdf = zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"}).to_crs(epsg=4326)

    # Aggregate by ZCTA
    epi_agg = df.groupby("zip_clean").agg(
        n_patients=("zip_clean", "count"),
        sz_free_rate=("seizure_free", "mean"),
        favorable_rate=("favorable_outcome", "mean"),
        complication_rate=("any_complication", "mean"),
    ).reset_index()
    epi_agg["zip_clean"] = epi_agg["zip_clean"].astype(str)

    ne_zctas = zcta_gdf[
        (zcta_gdf.geometry.centroid.x.between(-74, -69)) &
        (zcta_gdf.geometry.centroid.y.between(40.5, 43.5))
    ].copy()

    geo = ne_zctas.merge(epi_agg, left_on="ZCTA", right_on="zip_clean", how="inner")
    print(f"  {len(geo)} patient ZCTAs for spatial analysis")

    if len(geo) > 10:
        w = KNN.from_dataframe(geo, k=min(5, len(geo) - 1))
        w.transform = "r"

        for var, label in [
            ("sz_free_rate", "Seizure-free rate"),
            ("favorable_rate", "Favorable outcome rate"),
            ("complication_rate", "Complication rate"),
        ]:
            vals = geo[var].fillna(geo[var].median()).values
            mi = Moran(vals, w, permutations=999)
            sig = " *" if mi.p_sim < 0.05 else ""
            print(f"  {label}: I={mi.I:.4f}, p={mi.p_sim:.4f}{sig}")

    # Geospatial map: seizure freedom
    import matplotlib.patheffects as pe
    BWH_LAT, BWH_LON = 42.3358, -71.1065

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (var, label) in zip(axes, [
        ("sz_free_rate", "Seizure-Free Rate"),
        ("favorable_rate", "Favorable Outcome Rate"),
    ]):
        ne_zctas.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.1)
        scatter = ax.scatter(
            geo.geometry.centroid.x, geo.geometry.centroid.y,
            c=geo[var], s=geo["n_patients"] * 30 + 15,
            cmap="RdYlGn", vmin=0, vmax=1,
            edgecolors="black", linewidths=0.5, alpha=0.8, zorder=5
        )
        ax.plot(BWH_LON, BWH_LAT, marker="*", color="red", markersize=15, zorder=10,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate("BWH", xy=(BWH_LON + 0.08, BWH_LAT + 0.05), fontsize=10, fontweight="bold", color="red",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        plt.colorbar(scatter, ax=ax, shrink=0.6, label=label)
        ax.set_xlim(-73.5, -69.5)
        ax.set_ylim(41.0, 43.2)
        ax.set_title(f"{label} by Patient ZCTA", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/map_seizure_outcomes.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/map_seizure_outcomes.png")

except Exception as e:
    print(f"  Geospatial analysis failed: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# INTERACTION: PSYCH DX × DEPRIVATION → SEIZURE OUTCOMES
# ============================================================
print("\n" + "=" * 70)
print("INTERACTION: Psych Dx x Deprivation → Seizure Outcomes")
print("=" * 70)

for outcome_var, outcome_label in [("seizure_free", "Seizure-free"), ("favorable_outcome", "Favorable outcome")]:
    for idx_var in ["ICE_race_income", "SDI"]:
        cols = [outcome_var, "preop_any_psych_dx", idx_var] + covariates
        subset = df[cols].dropna()
        if len(subset) < 30:
            continue

        subset[f"{idx_var}_z"] = (subset[idx_var] - subset[idx_var].mean()) / subset[idx_var].std()

        # Model with interaction
        try:
            formula = f"{outcome_var} ~ preop_any_psych_dx * {idx_var}_z + " + " + ".join(covariates)
            m = smf.logit(formula, data=subset).fit(disp=0)
            interaction_term = f"preop_any_psych_dx:{idx_var}_z"
            or_int = np.exp(m.params[interaction_term])
            p_int = m.pvalues[interaction_term]
            sig = " *" if p_int < 0.05 else ""
            print(f"  {outcome_label} ~ preop_psych_dx x {idx_var}: interaction OR={or_int:.3f}, p={p_int:.4f}{sig}")
        except Exception as e:
            print(f"  {outcome_label} ~ preop_psych_dx x {idx_var}: failed ({e})")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nOutput files in {OUTPUT_DIR}/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")

# Save a JSON summary for the LaTeX report generator
summary = {
    "n_patients": len(df),
    "seizure_free_n": int(df["seizure_free"].sum()),
    "seizure_free_pct": round(df["seizure_free"].mean() * 100, 1),
    "favorable_n": int(df["favorable_outcome"].dropna().sum()),
    "favorable_pct": round(df["favorable_outcome"].mean() * 100, 1),
    "engel_I_n": int(df["engel_I"].dropna().sum()),
    "engel_I_pct": round(df["engel_I"].mean() * 100, 1),
    "complication_n": int(df["any_complication"].sum()),
    "complication_pct": round(df["any_complication"].mean() * 100, 1),
    "asm_reduced_n": int(df["asm_reduced"].dropna().sum()),
    "asm_reduced_pct": round(df["asm_reduced"].mean() * 100, 1),
    "n_significant": len(sig) if len(sig) > 0 else 0,
}
with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone!")
