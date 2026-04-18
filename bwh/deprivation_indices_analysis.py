#!/usr/bin/env python3
"""
Neighborhood Deprivation Indices & Psychiatric Comorbidity in Epilepsy Surgery
==============================================================================
Computes SDI, SVI-like, ICE, RUCA, and DCI-proxy indices from ZIP/ZCTA-level
ACS data and tests their association with pre- and post-operative psychiatric
diagnoses in an epilepsy surgery cohort.

Requirements:
    pip install pandas numpy openpyxl requests scipy statsmodels matplotlib seaborn geopandas

Census API key: https://api.census.gov/data/key_signup.html
"""

import pandas as pd
import numpy as np
import requests
import warnings
import os
from io import StringIO
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION — EDIT THESE
# ============================================================
CENSUS_API_KEY = "YOUR_KEY_HERE"  # Request at https://api.census.gov/data/key_signup.html
DATA_FILE = "Epilepsy_surgery_rohan_dataset_v1.xlsx"
ACS_YEAR = 2022  # 5-year ACS (2018-2022)
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PART 1: LOAD & PREPARE PATIENT DATA
# ============================================================
def load_patient_data(filepath):
    """Load the ASM analysis sheet and clean ZIP codes + psychiatric outcomes."""
    df = pd.read_excel(filepath, sheet_name="ASM analysis")
    print(f"Loaded {len(df)} patients from ASM analysis sheet")

    # Clean ZIP codes: remove brackets, ensure 5-digit string
    df["zip_clean"] = (
        df["Zipcode"]
        .astype(str)
        .str.strip("[]")
        .str.strip()
        .str.zfill(5)
    )
    df.loc[df["Zipcode"].isna(), "zip_clean"] = np.nan
    print(f"  {df['zip_clean'].notna().sum()} patients with valid ZIP codes")
    print(f"  {df['zip_clean'].nunique()} unique ZIP codes")

    # --- Pre-op psychiatric diagnosis (any F-code in pre-op ICD-10 DSM-5 columns) ---
    pre_dsm_cols = [
        "ICD_10_1 (DSM-5 AXIS)", "ICD_10_2 (DSM-5 AXIS)",
        "ICD_10_3 (DSM-5 AXIS)", "ICD_10_4 (DSM-5 AXIS)",
        "ICD_10_5 (DSM-5 AXIS)", "ICD_10_6 (DSM-5 AXIS)",
        "ICD_10_7 (DSM-5 AXIS)",
    ]
    df["preop_any_psych_dx"] = (
        df[pre_dsm_cols]
        .apply(lambda row: row.dropna().astype(str).str.startswith("F").any(), axis=1)
        .astype(int)
    )

    # Count of pre-op psychiatric diagnoses
    df["preop_psych_count"] = (
        df[pre_dsm_cols]
        .apply(lambda row: row.dropna().astype(str).str.startswith("F").sum(), axis=1)
    )

    # --- Post-op psychiatric diagnosis ---
    post_dsm_cols = [c for c in df.columns if "DSM" in c and "Post" in c]
    df["postop_any_psych_dx"] = (
        df[post_dsm_cols]
        .apply(lambda row: row.dropna().astype(str).str.upper().str.startswith("F").any(), axis=1)
        .astype(int)
    )

    df["postop_psych_count"] = (
        df[post_dsm_cols]
        .apply(lambda row: row.dropna().astype(str).str.upper().str.startswith("F").sum(), axis=1)
    )

    # --- New-onset post-op psychiatric diagnosis (had none pre-op, has one post-op) ---
    df["new_onset_postop_psych"] = (
        (df["preop_any_psych_dx"] == 0) & (df["postop_any_psych_dx"] == 1)
    ).astype(int)

    # --- Diagnostic subcategories (pre-op) ---
    def has_icd_prefix(row, prefixes):
        codes = row.dropna().astype(str)
        return any(
            any(c.startswith(p) for p in prefixes)
            for c in codes
        )

    for label, prefixes in [
        ("depression", ["F32", "F33", "F34"]),
        ("anxiety", ["F40", "F41", "F43"]),
        ("psychosis", ["F06", "F20", "F22", "F23", "F25", "F28", "F29"]),
        ("substance_use", ["F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"]),
        ("neurodevelopmental", ["F80", "F81", "F84", "F90"]),
    ]:
        df[f"preop_{label}"] = df[pre_dsm_cols].apply(
            lambda row: int(has_icd_prefix(row, prefixes)), axis=1
        )
        df[f"postop_{label}"] = df[post_dsm_cols].apply(
            lambda row: int(has_icd_prefix(row, prefixes)), axis=1
        )

    print(f"\n  Pre-op any psych dx: {df['preop_any_psych_dx'].sum()}/{df['preop_any_psych_dx'].notna().sum()}")
    print(f"  Post-op any psych dx: {df['postop_any_psych_dx'].sum()}/{df['postop_any_psych_dx'].notna().sum()}")
    print(f"  New-onset post-op: {df['new_onset_postop_psych'].sum()}")

    for cat in ["depression", "anxiety", "psychosis", "substance_use", "neurodevelopmental"]:
        pre_n = df[f"preop_{cat}"].sum()
        post_n = df[f"postop_{cat}"].sum()
        print(f"  {cat}: pre={pre_n}, post={post_n}")

    return df


# ============================================================
# PART 2: FETCH ACS DATA & COMPUTE INDICES AT ZCTA LEVEL
# ============================================================
def fetch_acs_data(api_key, year=2022):
    """
    Pull ACS 5-year estimates at ZCTA level for computing
    SDI, SVI-like, ICE, and DCI-proxy indices.
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"

    # --- Define all variables needed ---
    # SDI components (Robert Graham Center formula):
    #   1. % poverty (B17001_002 / B17001_001)
    #   2. % < HS education among 25+ (B15003_002..016 / B15003_001)
    #   3. % single-parent HH with children (B11003_010+B11003_016 / B11003_001)
    #   4. % renter-occupied (B25003_003 / B25003_001)
    #   5. % overcrowded (>1 person/room) (B25014_005+B25014_006+B25014_007+B25014_011+B25014_012+B25014_013 / B25014_001)
    #   6. % HH without vehicle (B08201_002 / B08201_001)
    #   7. % non-employed adults 16-64 (1 - B23025_004/B23025_003) ... approximated

    # SVI-like themes:
    #   Theme 1 (Socioeconomic): poverty, unemployment, per capita income, no HS diploma
    #   Theme 2 (HH Composition/Disability): 65+, <17, disability, single parent
    #   Theme 3 (Minority/Language): minority, limited English
    #   Theme 4 (Housing/Transport): multi-unit, mobile homes, crowding, no vehicle, group quarters

    # ICE: (affluent white - poor Black) / total
    #   B19001A (white alone income), B19001B (Black alone income), B19001_001 (total HH)

    # DCI-proxy: poverty, median income, education, housing vacancy, employment change
    #   B17001, B19013, B15003, B25002 (vacancy), B23025

    variables = [
        # Total population
        "B01003_001E",
        # Poverty
        "B17001_001E", "B17001_002E",
        # Education (25+) - total and less than HS
        "B15003_001E",
        "B15003_002E", "B15003_003E", "B15003_004E", "B15003_005E",
        "B15003_006E", "B15003_007E", "B15003_008E", "B15003_009E",
        "B15003_010E", "B15003_011E", "B15003_012E", "B15003_013E",
        "B15003_014E", "B15003_015E", "B15003_016E",
        # Single parent HH
        "B11003_001E", "B11003_010E", "B11003_016E",
        # Tenure (owner/renter)
        "B25003_001E", "B25003_003E",
        # Overcrowding (occupants per room)
        "B25014_001E",
        "B25014_005E", "B25014_006E", "B25014_007E",   # owner >1.00
        "B25014_011E", "B25014_012E", "B25014_013E",  # renter >1.00
        # Vehicles
        "B08201_001E", "B08201_002E",
        # Employment
        "B23025_001E", "B23025_003E", "B23025_004E", "B23025_005E",
        # Per capita income
        "B19301_001E",
        # Median HH income
        "B19013_001E",
        # Age groups (65+, <18)
        "B01001_001E",
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E",
        "B01001_024E", "B01001_025E",  # Male 65+
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E",
        "B01001_048E", "B01001_049E",  # Female 65+
        # Under 18 (male)
        "B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E",
        # Under 18 (female)
        "B01001_027E", "B01001_028E", "B01001_029E", "B01001_030E",
        # Disability (civilian non-institutionalized with disability)
        "B18101_001E", "B18101_004E", "B18101_007E", "B18101_010E",
        "B18101_013E", "B18101_016E", "B18101_019E",
        # Race/ethnicity for minority %
        "B03002_001E", "B03002_003E",  # total, NH white alone
        # Limited English HH
        "B16002_001E", "B16002_004E", "B16002_007E", "B16002_010E", "B16002_013E",
        # Housing: multi-unit structures
        "B25024_001E", "B25024_007E", "B25024_008E", "B25024_009E",  # 10-19, 20-49, 50+
        # Mobile homes
        "B25024_010E",
        # Group quarters
        "B26001_001E",
        # Housing vacancy
        "B25002_001E", "B25002_003E",
        # ICE: income by race
        # White alone HH income - top brackets (>=125k)
        "B19001A_001E",  # total white HH
        "B19001A_014E", "B19001A_015E", "B19001A_016E", "B19001A_017E",
        # Black alone HH income - bottom brackets (<25k)
        "B19001B_001E",  # total Black HH
        "B19001B_002E", "B19001B_003E", "B19001B_004E", "B19001B_005E",
        # Total HH income
        "B19001_001E",
        "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E",  # >=125k (affluent)
        "B19001_002E", "B19001_003E",  # <15k (poor)
        # Gini index
        "B19083_001E",
    ]

    # Census API has a limit of ~50 variables per call, so batch them
    results = {}
    batch_size = 48
    for i in range(0, len(variables), batch_size):
        batch = variables[i:i + batch_size]
        var_str = ",".join(batch)
        url = f"{base_url}?get=NAME,{var_str}&for=zip%20code%20tabulation%20area:*&key={api_key}"
        print(f"  Fetching ACS batch {i // batch_size + 1}...")
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"  ERROR: {resp.status_code} — {resp.text[:200]}")
            return None
        data = resp.json()
        header = data[0]
        rows = data[1:]
        batch_df = pd.DataFrame(rows, columns=header)
        if not isinstance(results, pd.DataFrame):
            results = batch_df
        else:
            # Merge on ZCTA
            zcta_col = [c for c in batch_df.columns if "zip" in c.lower() or "zcta" in c.lower()]
            merge_col = zcta_col[0] if zcta_col else batch_df.columns[-1]
            batch_df = batch_df.drop(columns=["NAME"], errors="ignore")
            results = results.merge(batch_df, on=merge_col, how="outer", suffixes=("", "_dup"))
            # Drop duplicate columns
            results = results[[c for c in results.columns if not c.endswith("_dup")]]

    # Identify ZCTA column
    zcta_col = [c for c in results.columns if "zip" in c.lower() or "zcta" in c.lower() or "tabulation" in c.lower()]
    zcta_col_name = zcta_col[0] if zcta_col else results.columns[-1]
    results = results.rename(columns={zcta_col_name: "ZCTA"})

    # Convert numeric columns
    for col in results.columns:
        if col not in ["NAME", "ZCTA"]:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    print(f"  Retrieved ACS data for {len(results)} ZCTAs")
    return results


def compute_sdi(acs):
    """Compute Social Deprivation Index (Robert Graham Center method)."""
    df = acs.copy()

    # 1. % poverty
    df["pct_poverty"] = df["B17001_002E"] / df["B17001_001E"]

    # 2. % < HS education (25+): sum of B15003_002E through B15003_016E / total
    less_hs_cols = [f"B15003_{str(i).zfill(3)}E" for i in range(2, 17)]
    df["pct_no_hs"] = df[less_hs_cols].sum(axis=1) / df["B15003_001E"]

    # 3. % single parent HH with children
    df["pct_single_parent"] = (df["B11003_010E"] + df["B11003_016E"]) / df["B11003_001E"]

    # 4. % renter-occupied
    df["pct_renter"] = df["B25003_003E"] / df["B25003_001E"]

    # 5. % overcrowded (>1 person/room)
    overcrowd_cols = ["B25014_005E", "B25014_006E", "B25014_007E",
                      "B25014_011E", "B25014_012E", "B25014_013E"]
    df["pct_overcrowded"] = df[overcrowd_cols].sum(axis=1) / df["B25014_001E"]

    # 6. % HH without vehicle
    df["pct_no_vehicle"] = df["B08201_002E"] / df["B08201_001E"]

    # 7. % non-employed (unemployed / civilian labor force)
    df["pct_unemployed"] = df["B23025_005E"] / df["B23025_003E"]

    # Composite SDI: percentile rank of sum of percentile ranks
    sdi_components = [
        "pct_poverty", "pct_no_hs", "pct_single_parent",
        "pct_renter", "pct_overcrowded", "pct_no_vehicle", "pct_unemployed"
    ]

    for comp in sdi_components:
        df[f"{comp}_pctile"] = df[comp].rank(pct=True, na_option="keep")

    df["SDI_raw"] = df[[f"{c}_pctile" for c in sdi_components]].mean(axis=1)
    df["SDI"] = df["SDI_raw"].rank(pct=True, na_option="keep") * 100

    return df[["ZCTA", "SDI"] + sdi_components]


def compute_svi_like(acs):
    """Compute SVI-like index from ACS data (4 themes)."""
    df = acs.copy()

    # Theme 1: Socioeconomic
    df["pct_poverty"] = df["B17001_002E"] / df["B17001_001E"]
    df["pct_unemployed"] = df["B23025_005E"] / df["B23025_003E"]
    df["per_capita_income"] = df["B19301_001E"]
    less_hs_cols = [f"B15003_{str(i).zfill(3)}E" for i in range(2, 17)]
    df["pct_no_hs"] = df[less_hs_cols].sum(axis=1) / df["B15003_001E"]

    # Theme 2: HH Composition & Disability
    male_65 = ["B01001_020E", "B01001_021E", "B01001_022E",
               "B01001_023E", "B01001_024E", "B01001_025E"]
    female_65 = ["B01001_044E", "B01001_045E", "B01001_046E",
                 "B01001_047E", "B01001_048E", "B01001_049E"]
    df["pct_65plus"] = (df[male_65].sum(axis=1) + df[female_65].sum(axis=1)) / df["B01001_001E"]

    male_under18 = ["B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E"]
    female_under18 = ["B01001_027E", "B01001_028E", "B01001_029E", "B01001_030E"]
    df["pct_under18"] = (df[male_under18].sum(axis=1) + df[female_under18].sum(axis=1)) / df["B01001_001E"]

    # Disability: with disability / total civilian noninstitutionalized
    disability_cols = ["B18101_004E", "B18101_007E", "B18101_010E",
                       "B18101_013E", "B18101_016E", "B18101_019E"]
    df["pct_disability"] = df[disability_cols].sum(axis=1) / df["B18101_001E"]

    df["pct_single_parent"] = (df["B11003_010E"] + df["B11003_016E"]) / df["B11003_001E"]

    # Theme 3: Minority Status & Language
    df["pct_minority"] = 1 - (df["B03002_003E"] / df["B03002_001E"])
    # Limited English speaking HH
    lep_cols = ["B16002_004E", "B16002_007E", "B16002_010E", "B16002_013E"]
    df["pct_limited_english"] = df[lep_cols].sum(axis=1) / df["B16002_001E"]

    # Theme 4: Housing Type & Transportation
    multi_unit_cols = ["B25024_007E", "B25024_008E", "B25024_009E"]
    df["pct_multi_unit"] = df[multi_unit_cols].sum(axis=1) / df["B25024_001E"]
    df["pct_mobile_home"] = df["B25024_010E"] / df["B25024_001E"]

    overcrowd_cols = ["B25014_005E", "B25014_006E", "B25014_007E",
                      "B25014_011E", "B25014_012E", "B25014_013E"]
    df["pct_overcrowded"] = overcrowd_cols and df[overcrowd_cols].sum(axis=1) / df["B25014_001E"]
    df["pct_no_vehicle"] = df["B08201_002E"] / df["B08201_001E"]
    df["pct_group_quarters"] = df["B26001_001E"] / df["B01003_001E"]

    # Compute theme scores as average percentile rank
    theme1_vars = ["pct_poverty", "pct_unemployed", "pct_no_hs"]
    # Per capita income is inverse (higher = less vulnerable)
    df["per_capita_income_inv"] = -df["per_capita_income"]
    theme1_vars_rank = theme1_vars + ["per_capita_income_inv"]

    theme2_vars = ["pct_65plus", "pct_under18", "pct_disability", "pct_single_parent"]
    theme3_vars = ["pct_minority", "pct_limited_english"]
    theme4_vars = ["pct_multi_unit", "pct_mobile_home", "pct_overcrowded", "pct_no_vehicle", "pct_group_quarters"]

    for theme_name, theme_vars in [
        ("SVI_theme1_socioeconomic", theme1_vars_rank),
        ("SVI_theme2_hh_composition", theme2_vars),
        ("SVI_theme3_minority_language", theme3_vars),
        ("SVI_theme4_housing_transport", theme4_vars),
    ]:
        for v in theme_vars:
            df[f"{v}_pctile"] = df[v].rank(pct=True, na_option="keep")
        df[theme_name] = df[[f"{v}_pctile" for v in theme_vars]].mean(axis=1)
        df[theme_name] = df[theme_name].rank(pct=True, na_option="keep") * 100

    # Overall SVI
    all_theme_cols = ["SVI_theme1_socioeconomic", "SVI_theme2_hh_composition",
                      "SVI_theme3_minority_language", "SVI_theme4_housing_transport"]
    df["SVI_overall"] = df[all_theme_cols].mean(axis=1)

    return df[["ZCTA"] + all_theme_cols + ["SVI_overall"]]


def compute_ice(acs):
    """
    Compute Index of Concentration at the Extremes (ICE).
    Three variants:
      ICE_income: (affluent HH - poor HH) / total HH
      ICE_race: (NH White - NH Black) / total pop (from B03002 not available per race, approx)
      ICE_race_income: (affluent White HH - poor Black HH) / total HH
    Range: -1 (total concentration of deprivation) to +1 (total concentration of privilege)
    """
    df = acs.copy()

    # ICE-income: affluent (>=125k) vs poor (<15k)
    affluent_cols = ["B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E"]
    poor_cols = ["B19001_002E", "B19001_003E"]
    df["affluent_hh"] = df[affluent_cols].sum(axis=1)
    df["poor_hh"] = df[poor_cols].sum(axis=1)
    df["ICE_income"] = (df["affluent_hh"] - df["poor_hh"]) / df["B19001_001E"]

    # ICE-race_income: affluent White HH vs poor Black HH
    affluent_white_cols = ["B19001A_014E", "B19001A_015E", "B19001A_016E", "B19001A_017E"]
    poor_black_cols = ["B19001B_002E", "B19001B_003E", "B19001B_004E", "B19001B_005E"]
    df["affluent_white_hh"] = df[affluent_white_cols].sum(axis=1)
    df["poor_black_hh"] = df[poor_black_cols].sum(axis=1)
    df["ICE_race_income"] = (df["affluent_white_hh"] - df["poor_black_hh"]) / df["B19001_001E"]

    return df[["ZCTA", "ICE_income", "ICE_race_income"]]


def compute_dci_proxy(acs):
    """
    Compute DCI-like proxy from ACS data.
    The actual DCI (Economic Innovation Group) uses proprietary business data.
    This proxy uses: poverty rate, median income, HS attainment, housing vacancy,
    employment ratio, and change metrics (unavailable in single cross-section).
    """
    df = acs.copy()

    df["pct_poverty"] = df["B17001_002E"] / df["B17001_001E"]
    df["median_hh_income"] = df["B19013_001E"]
    less_hs_cols = [f"B15003_{str(i).zfill(3)}E" for i in range(2, 17)]
    df["pct_no_hs"] = df[less_hs_cols].sum(axis=1) / df["B15003_001E"]
    df["pct_vacant"] = df["B25002_003E"] / df["B25002_001E"]
    df["employment_rate"] = df["B23025_004E"] / df["B23025_001E"]

    # Composite: percentile rank approach
    df["pct_poverty_pctile"] = df["pct_poverty"].rank(pct=True, na_option="keep")
    df["median_income_inv_pctile"] = (-df["median_hh_income"]).rank(pct=True, na_option="keep")
    df["pct_no_hs_pctile"] = df["pct_no_hs"].rank(pct=True, na_option="keep")
    df["pct_vacant_pctile"] = df["pct_vacant"].rank(pct=True, na_option="keep")
    df["employment_inv_pctile"] = (-df["employment_rate"]).rank(pct=True, na_option="keep")

    dci_comps = [
        "pct_poverty_pctile", "median_income_inv_pctile",
        "pct_no_hs_pctile", "pct_vacant_pctile", "employment_inv_pctile"
    ]
    df["DCI_proxy"] = df[dci_comps].mean(axis=1).rank(pct=True, na_option="keep") * 100

    return df[["ZCTA", "DCI_proxy"]]


def fetch_ruca_codes():
    """
    Fetch RUCA codes at ZIP level from USDA ERS.
    Falls back to a simplified version if download fails.
    """
    # RUCA ZIP-level codes from USDA (2020 version)
    url = "https://www.ers.usda.gov/media/5442/2020-rural-urban-commuting-area-codes-zip-codes.xlsx?v=17942"
    print("  Fetching RUCA 2020 codes from USDA...")
    try:
        ruca = pd.read_excel(url, sheet_name="RUCA 2020 ZIP Code Data", dtype=str)
        # First row is the actual header
        ruca.columns = ruca.iloc[0].values
        ruca = ruca.iloc[1:].reset_index(drop=True)
        ruca = ruca.rename(columns={"ZIPCode": "ZCTA", "PrimaryRUCA": "RUCA_primary", "SecondaryRUCA": "RUCA_secondary"})
        ruca["ZCTA"] = ruca["ZCTA"].str.zfill(5)

        # Classify: 1-3 = Metropolitan, 4-6 = Micropolitan, 7-9 = Small town, 10 = Rural
        def ruca_category(code):
            try:
                c = float(code)
            except (ValueError, TypeError):
                return np.nan
            if c <= 3:
                return "Metropolitan"
            elif c <= 6:
                return "Micropolitan"
            elif c <= 9:
                return "Small town"
            else:
                return "Rural"

        ruca["RUCA_category"] = ruca["RUCA_primary"].apply(ruca_category)
        print(f"  Retrieved RUCA codes for {len(ruca)} ZIPs")
        return ruca[["ZCTA", "RUCA_primary", "RUCA_secondary", "RUCA_category"]]
    except Exception as e:
        print(f"  WARNING: Could not fetch RUCA codes: {e}")
        print("  You can manually download from: https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes/")
        return None


# ============================================================
# PART 3: MERGE INDICES WITH PATIENT DATA
# ============================================================
def merge_all(patient_df, sdi, svi, ice, dci, ruca):
    """Merge all indices onto patient data by ZIP/ZCTA."""
    merged = patient_df.copy()

    for idx_df, name in [(sdi, "SDI"), (svi, "SVI"), (ice, "ICE"), (dci, "DCI"), (ruca, "RUCA")]:
        if idx_df is not None:
            merged = merged.merge(idx_df, left_on="zip_clean", right_on="ZCTA", how="left")
            matched = merged["ZCTA"].notna().sum()
            merged = merged.drop(columns=["ZCTA"], errors="ignore")
            print(f"  {name}: matched {matched}/{len(merged)} patients")

    return merged


# ============================================================
# PART 4: STATISTICAL ANALYSIS
# ============================================================
def run_association_analyses(df):
    """
    Test associations between deprivation indices and psychiatric outcomes.
    Returns a summary DataFrame of results.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = []

    # Index variables
    index_vars = [
        "SDI", "SVI_overall", "SVI_theme1_socioeconomic", "SVI_theme2_hh_composition",
        "SVI_theme3_minority_language", "SVI_theme4_housing_transport",
        "ICE_income", "ICE_race_income", "DCI_proxy",
    ]

    # Outcome variables
    outcomes = [
        ("preop_any_psych_dx", "Pre-op any psychiatric dx"),
        ("postop_any_psych_dx", "Post-op any psychiatric dx"),
        ("new_onset_postop_psych", "New-onset post-op psychiatric dx"),
        ("preop_depression", "Pre-op depression"),
        ("postop_depression", "Post-op depression"),
        ("preop_anxiety", "Pre-op anxiety"),
        ("postop_anxiety", "Post-op anxiety"),
    ]

    # Covariates for adjusted models
    covariates = [
        "Male=0, \nFemale=1",
        "Age at surgery",
        "Duration of epilepsy \n(years, decimal okay)",
    ]
    # Rename covariates for formula compatibility
    df = df.rename(columns={
        "Male=0, \nFemale=1": "female",
        "Age at surgery": "age_at_surgery",
        "Duration of epilepsy \n(years, decimal okay)": "epilepsy_duration",
    })
    covar_formula = ["female", "age_at_surgery", "epilepsy_duration"]

    print("\n" + "=" * 80)
    print("ASSOCIATION ANALYSES: Deprivation Indices → Psychiatric Outcomes")
    print("=" * 80)

    for outcome_var, outcome_label in outcomes:
        print(f"\n--- Outcome: {outcome_label} (n events = {df[outcome_var].sum()}) ---")

        if df[outcome_var].sum() < 10:
            print("  SKIPPED: fewer than 10 events, insufficient for logistic regression")
            continue

        for idx_var in index_vars:
            if idx_var not in df.columns:
                continue

            # Subset to complete cases
            analysis_cols = [outcome_var, idx_var] + covar_formula
            subset = df[analysis_cols].dropna()

            if len(subset) < 20:
                continue

            # Standardize index for interpretability (per 1-SD increase)
            subset[f"{idx_var}_z"] = (subset[idx_var] - subset[idx_var].mean()) / subset[idx_var].std()

            # --- Unadjusted logistic regression ---
            try:
                formula_unadj = f"{outcome_var} ~ {idx_var}_z"
                model_unadj = smf.logit(formula_unadj, data=subset).fit(disp=0)
                or_unadj = np.exp(model_unadj.params[f"{idx_var}_z"])
                ci_unadj = np.exp(model_unadj.conf_int().loc[f"{idx_var}_z"])
                p_unadj = model_unadj.pvalues[f"{idx_var}_z"]
            except Exception:
                continue

            # --- Adjusted logistic regression ---
            try:
                formula_adj = f"{outcome_var} ~ {idx_var}_z + " + " + ".join(covar_formula)
                model_adj = smf.logit(formula_adj, data=subset).fit(disp=0)
                or_adj = np.exp(model_adj.params[f"{idx_var}_z"])
                ci_adj = np.exp(model_adj.conf_int().loc[f"{idx_var}_z"])
                p_adj = model_adj.pvalues[f"{idx_var}_z"]
            except Exception:
                or_adj = ci_adj = p_adj = np.nan
                ci_adj = pd.Series([np.nan, np.nan])

            results.append({
                "Outcome": outcome_label,
                "Index": idx_var,
                "N": len(subset),
                "OR_unadj": round(or_unadj, 3),
                "CI95_unadj": f"({ci_unadj.iloc[0]:.3f}–{ci_unadj.iloc[1]:.3f})",
                "p_unadj": round(p_unadj, 4),
                "OR_adj": round(or_adj, 3) if not np.isnan(or_adj) else np.nan,
                "CI95_adj": f"({ci_adj.iloc[0]:.3f}–{ci_adj.iloc[1]:.3f})" if not any(pd.isna(ci_adj)) else np.nan,
                "p_adj": round(p_adj, 4) if not np.isnan(p_adj) else np.nan,
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print("\n" + "=" * 80)
        print("RESULTS TABLE (OR per 1-SD increase in index)")
        print("=" * 80)
        print(results_df.to_string(index=False))

        # Flag significant results
        sig = results_df[results_df["p_unadj"] < 0.05]
        if len(sig) > 0:
            print(f"\n*** {len(sig)} associations significant at p<0.05 (unadjusted) ***")
            print(sig[["Outcome", "Index", "OR_unadj", "CI95_unadj", "p_unadj"]].to_string(index=False))

        sig_adj = results_df[results_df["p_adj"] < 0.05]
        if len(sig_adj) > 0:
            print(f"\n*** {len(sig_adj)} associations significant at p<0.05 (adjusted) ***")
            print(sig_adj[["Outcome", "Index", "OR_adj", "CI95_adj", "p_adj"]].to_string(index=False))

    return results_df


def run_ruca_analysis(df):
    """Chi-square / Fisher's exact test for RUCA category × psychiatric outcomes."""
    if "RUCA_category" not in df.columns:
        print("RUCA categories not available, skipping.")
        return None

    print("\n" + "=" * 80)
    print("RUCA (Rural-Urban) × Psychiatric Outcomes")
    print("=" * 80)

    outcomes = [
        ("preop_any_psych_dx", "Pre-op any psychiatric dx"),
        ("postop_any_psych_dx", "Post-op any psychiatric dx"),
    ]

    for outcome_var, outcome_label in outcomes:
        subset = df[["RUCA_category", outcome_var]].dropna()
        if len(subset) < 20:
            continue

        ct = pd.crosstab(subset["RUCA_category"], subset[outcome_var])
        print(f"\n--- {outcome_label} ---")
        print(ct)

        # Chi-square or Fisher's exact
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(ct)
            print(f"  Chi-square = {chi2:.2f}, df = {dof}, p = {p:.4f}")

            # Rates by RUCA category
            rates = subset.groupby("RUCA_category")[outcome_var].mean()
            print("  Rates by RUCA category:")
            for cat, rate in rates.items():
                n = (subset["RUCA_category"] == cat).sum()
                print(f"    {cat}: {rate:.1%} (n={n})")


def compute_poisson_count_models(df):
    """Poisson regression for count of psychiatric diagnoses."""
    import statsmodels.formula.api as smf

    print("\n" + "=" * 80)
    print("POISSON REGRESSION: Deprivation Indices → Count of Psychiatric Diagnoses")
    print("=" * 80)

    # Rename covariates if not already done
    if "female" not in df.columns:
        df = df.rename(columns={
            "Male=0, \nFemale=1": "female",
            "Age at surgery": "age_at_surgery",
            "Duration of epilepsy \n(years, decimal okay)": "epilepsy_duration",
        })

    index_vars = ["SDI", "SVI_overall", "ICE_income", "ICE_race_income", "DCI_proxy"]
    count_outcomes = [
        ("preop_psych_count", "Pre-op psych dx count"),
        ("postop_psych_count", "Post-op psych dx count"),
    ]

    results = []
    for outcome_var, outcome_label in count_outcomes:
        print(f"\n--- {outcome_label} (mean={df[outcome_var].mean():.2f}, max={df[outcome_var].max()}) ---")

        for idx_var in index_vars:
            if idx_var not in df.columns:
                continue

            analysis_cols = [outcome_var, idx_var, "female", "age_at_surgery", "epilepsy_duration"]
            subset = df[analysis_cols].dropna()
            if len(subset) < 20:
                continue

            subset[f"{idx_var}_z"] = (subset[idx_var] - subset[idx_var].mean()) / subset[idx_var].std()

            try:
                formula = f"{outcome_var} ~ {idx_var}_z + female + age_at_surgery + epilepsy_duration"
                model = smf.poisson(formula, data=subset).fit(disp=0)
                irr = np.exp(model.params[f"{idx_var}_z"])
                ci = np.exp(model.conf_int().loc[f"{idx_var}_z"])
                p = model.pvalues[f"{idx_var}_z"]

                results.append({
                    "Outcome": outcome_label,
                    "Index": idx_var,
                    "N": len(subset),
                    "IRR": round(irr, 3),
                    "CI95": f"({ci.iloc[0]:.3f}–{ci.iloc[1]:.3f})",
                    "p": round(p, 4),
                })
                print(f"  {idx_var}: IRR={irr:.3f} ({ci.iloc[0]:.3f}–{ci.iloc[1]:.3f}), p={p:.4f}")
            except Exception as e:
                print(f"  {idx_var}: model failed — {e}")

    return pd.DataFrame(results)


# ============================================================
# PART 5: VISUALIZATION
# ============================================================
def create_visualizations(df, results_df):
    """Generate publication-quality figures."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping visualizations")
        return

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # --- Figure 1: Forest plot of ORs ---
    if results_df is not None and len(results_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))

        for ax_idx, (outcome_filter, title) in enumerate([
            ("Pre-op", "Pre-op Psychiatric Diagnosis"),
            ("Post-op any", "Post-op Psychiatric Diagnosis"),
        ]):
            ax = axes[ax_idx]
            subset = results_df[results_df["Outcome"].str.contains(outcome_filter)].copy()
            if len(subset) == 0:
                continue

            # Parse CIs
            subset["CI_lo"] = subset["CI95_unadj"].str.extract(r"\(([0-9.]+)").astype(float)
            subset["CI_hi"] = subset["CI95_unadj"].str.extract(r"–([0-9.]+)\)").astype(float)

            y_pos = range(len(subset))
            ax.errorbar(
                subset["OR_unadj"], y_pos,
                xerr=[subset["OR_unadj"] - subset["CI_lo"], subset["CI_hi"] - subset["OR_unadj"]],
                fmt="o", color="steelblue", capsize=3, markersize=6
            )
            ax.axvline(x=1, color="red", linestyle="--", alpha=0.5)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(subset["Index"])
            ax.set_xlabel("Odds Ratio (95% CI) per 1-SD increase")
            ax.set_title(title)

            # Add p-values
            for i, (_, row) in enumerate(subset.iterrows()):
                p_str = f"p={row['p_unadj']:.3f}" if row['p_unadj'] >= 0.001 else "p<0.001"
                sig_marker = " *" if row['p_unadj'] < 0.05 else ""
                ax.annotate(f"{p_str}{sig_marker}", xy=(row["CI_hi"] + 0.02, i), fontsize=8, va="center")

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/forest_plot_deprivation_psych.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}/forest_plot_deprivation_psych.png")

    # --- Figure 2: Distribution of indices in cohort ---
    index_cols = [c for c in ["SDI", "SVI_overall", "ICE_income", "ICE_race_income", "DCI_proxy"]
                  if c in df.columns]
    if index_cols:
        fig, axes = plt.subplots(1, len(index_cols), figsize=(4 * len(index_cols), 4))
        if len(index_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, index_cols):
            data = df[col].dropna()
            ax.hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_title(col)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.axvline(data.median(), color="red", linestyle="--", label=f"Median={data.median():.1f}")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/index_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}/index_distributions.png")

    # --- Figure 3: Psychiatric dx rate by deprivation quartile ---
    if "SDI" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (outcome, label) in zip(axes, [
            ("preop_any_psych_dx", "Pre-op Psych Dx"),
            ("postop_any_psych_dx", "Post-op Psych Dx"),
        ]):
            subset = df[["SDI", outcome]].dropna()
            subset["SDI_quartile"] = pd.qcut(subset["SDI"], 4, labels=["Q1\n(least deprived)", "Q2", "Q3", "Q4\n(most deprived)"])
            rates = subset.groupby("SDI_quartile", observed=True)[outcome].agg(["mean", "count"])
            rates["se"] = np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["count"])

            ax.bar(range(4), rates["mean"], yerr=1.96 * rates["se"],
                   color=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
                   edgecolor="white", capsize=5)
            ax.set_xticks(range(4))
            ax.set_xticklabels(rates.index)
            ax.set_ylabel("Proportion with Psychiatric Dx")
            ax.set_title(f"{label} by SDI Quartile")
            ax.set_ylim(0, 1)

            # Add n per bar
            for i, (_, row) in enumerate(rates.iterrows()):
                ax.text(i, row["mean"] + 1.96 * row["se"] + 0.03,
                        f"n={int(row['count'])}", ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/psych_dx_by_sdi_quartile.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}/psych_dx_by_sdi_quartile.png")


# ============================================================
# PART 6: GEOSPATIAL ANALYSIS
# ============================================================
def geospatial_analysis(df):
    """
    Geospatial analyses: clustering, hotspot detection, mapping.
    Requires geopandas. Prints recommendations if not available.
    """
    print("\n" + "=" * 80)
    print("GEOSPATIAL ANALYSIS")
    print("=" * 80)

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        HAS_GEO = True
    except ImportError:
        HAS_GEO = False
        print("  geopandas not installed. Install with: pip install geopandas")

    # Try to get ZCTA centroids for mapping
    if HAS_GEO and "zip_clean" in df.columns:
        print("\n  Attempting to create point map from ZIP centroids...")
        try:
            # Use Census ZCTA shapefile
            zcta_url = "https://www2.census.gov/geo/tiger/TIGER2022/ZCTA520/tl_2022_us_zcta520.zip"
            print(f"  Downloading ZCTA shapefiles (this may take a moment)...")
            zcta_gdf = gpd.read_file(zcta_url)
            zcta_gdf = zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"})

            # Merge patient data
            patient_agg = df.groupby("zip_clean").agg(
                n_patients=("zip_clean", "count"),
                preop_psych_rate=("preop_any_psych_dx", "mean"),
                postop_psych_rate=("postop_any_psych_dx", "mean"),
                mean_SDI=("SDI", "mean") if "SDI" in df.columns else ("zip_clean", "count"),
            ).reset_index()

            geo_merged = zcta_gdf.merge(patient_agg, left_on="ZCTA", right_on="zip_clean", how="inner")
            print(f"  Matched {len(geo_merged)} ZCTAs for mapping")

            if len(geo_merged) > 0:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use("Agg")

                fig, axes = plt.subplots(1, 2, figsize=(16, 8))

                geo_merged.plot(column="preop_psych_rate", cmap="RdYlGn_r",
                                legend=True, ax=axes[0], edgecolor="gray", linewidth=0.3)
                axes[0].set_title("Pre-op Psychiatric Dx Rate by ZCTA")
                axes[0].set_xlim(-73.5, -69.5)  # MA bounds
                axes[0].set_ylim(41, 43)

                if "mean_SDI" in geo_merged.columns and geo_merged["mean_SDI"].notna().any():
                    geo_merged.plot(column="mean_SDI", cmap="RdYlGn_r",
                                    legend=True, ax=axes[1], edgecolor="gray", linewidth=0.3)
                    axes[1].set_title("SDI by ZCTA (Patient ZCTAs)")
                    axes[1].set_xlim(-73.5, -69.5)
                    axes[1].set_ylim(41, 43)

                plt.tight_layout()
                plt.savefig(f"{OUTPUT_DIR}/geospatial_map.png", dpi=300, bbox_inches="tight")
                plt.close()
                print(f"  Saved: {OUTPUT_DIR}/geospatial_map.png")

        except Exception as e:
            print(f"  Geospatial mapping failed: {e}")
            print("  This is optional — the statistical analyses above are the primary results.")

    # --- Recommendations ---
    print("\n" + "-" * 60)
    print("GEOSPATIAL ANALYSIS RECOMMENDATIONS FOR YOUR PAPER")
    print("-" * 60)
    print("""
    Given your dataset (n=284, ~199 unique ZIPs, mostly Massachusetts):

    FEASIBLE & HIGH-VALUE:
    1. Choropleth maps of ZCTA-level deprivation indices overlaid with patient
       locations (centroids). This provides a visual narrative of where your
       patients come from relative to neighborhood deprivation.

    2. Spatial clustering of psychiatric outcomes: Use Moran's I (global) and
       LISA (local) statistics to test whether psychiatric comorbidity clusters
       geographically. This answers: "Are patients from nearby ZCTAs more
       similar in psychiatric burden than expected by chance?"
       → Python: PySAL/esda library; R: spdep package

    3. Getis-Ord Gi* hotspot analysis: Identifies statistically significant
       hot/cold spots of psychiatric comorbidity. Publishable as a figure
       showing where psychiatric burden concentrates.

    4. Distance to nearest epilepsy center as a covariate: Compute great-circle
       distance from each patient's ZCTA centroid to your hospital. This
       captures access-to-care effects beyond deprivation.

    CAVEATS FOR YOUR PAPER:
    - With ~199 unique ZIPs and n=284, many ZIPs have only 1 patient → unstable
      ZCTA-level rates. Aggregate into larger regions (county or HSA) for
      mapping, or use individual-level models with ZCTA as a random effect.

    - ZIP ≠ ZCTA exactly. Most map 1:1, but some ZIPs (PO boxes, large
      employers) don't correspond to a geographic ZCTA. Document this.

    - MAUP sensitivity: If you can geocode even a subset to Census tract,
      compare tract-level vs ZCTA-level index associations as a sensitivity
      analysis. This is increasingly expected by reviewers.

    - Spatial autocorrelation in residuals: If you find significant Moran's I,
      consider spatial regression models (spatial lag or spatial error) instead
      of standard logistic regression. This is advanced but strengthens the paper.

    PACKAGES NEEDED:
      pip install geopandas pysal esda libpysal mapclassify contextily
      # or in R: spdep, sf, tmap, spatialreg
    """)


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("=" * 80)
    print("NEIGHBORHOOD DEPRIVATION & PSYCHIATRIC COMORBIDITY IN EPILEPSY SURGERY")
    print("=" * 80)

    # Step 1: Load patient data
    print("\n[1/6] Loading patient data...")
    patient_df = load_patient_data(DATA_FILE)

    # Step 2: Fetch ACS data
    if CENSUS_API_KEY == "YOUR_KEY_HERE":
        print("\n⚠ WARNING: No Census API key set!")
        print("  Get one at: https://api.census.gov/data/key_signup.html")
        print("  Then paste it in CENSUS_API_KEY at the top of this script.")
        print("  Skipping ACS-based indices (SDI, SVI, ICE, DCI).")
        acs_data = None
    else:
        print("\n[2/6] Fetching ACS data from Census API...")
        acs_data = fetch_acs_data(CENSUS_API_KEY, ACS_YEAR)

    # Step 3: Compute indices
    sdi = svi = ice = dci = None
    if acs_data is not None:
        print("\n[3/6] Computing deprivation indices...")
        print("  Computing SDI...")
        sdi = compute_sdi(acs_data)
        print("  Computing SVI-like index...")
        svi = compute_svi_like(acs_data)
        print("  Computing ICE...")
        ice = compute_ice(acs_data)
        print("  Computing DCI proxy...")
        dci = compute_dci_proxy(acs_data)

    # Step 4: Fetch RUCA
    print("\n[4/6] Fetching RUCA codes...")
    ruca = fetch_ruca_codes()

    # Step 5: Merge
    print("\n[5/6] Merging indices with patient data...")
    merged = merge_all(patient_df, sdi, svi, ice, dci, ruca)

    # Save merged dataset
    merged_output = f"{OUTPUT_DIR}/merged_deprivation_psych_data.csv"
    merged.to_csv(merged_output, index=False)
    print(f"  Saved merged dataset: {merged_output}")

    # Also save as Excel
    merged.to_excel(f"{OUTPUT_DIR}/merged_deprivation_psych_data.xlsx", index=False)

    # Step 6: Analyses
    print("\n[6/6] Running analyses...")

    if acs_data is not None:
        results_df = run_association_analyses(merged)
        results_df.to_csv(f"{OUTPUT_DIR}/association_results.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR}/association_results.csv")

        poisson_results = compute_poisson_count_models(merged)
        if len(poisson_results) > 0:
            poisson_results.to_csv(f"{OUTPUT_DIR}/poisson_results.csv", index=False)

        run_ruca_analysis(merged)

        print("\n  Creating visualizations...")
        create_visualizations(merged, results_df)
    else:
        print("  Skipping statistical analyses (no ACS data). Add Census API key and re-run.")
        results_df = None

    # Geospatial recommendations
    geospatial_analysis(merged)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Patients: {len(merged)}")
    print(f"  With ZIP codes: {merged['zip_clean'].notna().sum()}")
    if "SDI" in merged.columns:
        print(f"  With SDI: {merged['SDI'].notna().sum()}")
        print(f"  SDI range: {merged['SDI'].min():.1f}–{merged['SDI'].max():.1f}")
    if "RUCA_category" in merged.columns:
        print(f"  RUCA distribution:")
        print(merged["RUCA_category"].value_counts().to_string(header=False))
    print(f"\n  Output files in: {OUTPUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
