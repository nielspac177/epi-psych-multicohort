#!/usr/bin/env python3
"""
Extract epilepsy cohort from NIS 2011-2020 SPSS file.
Identifies epilepsy patients (DX1 = 345.xx or G40.x) and flags:
  - Epilepsy surgery procedures (resective, neuromodulation, monitoring)
  - DSM-5 Axis I psychiatric comorbidities (any DX position)
  - Demographics and outcomes
Uses CHUNKED reading to avoid OOM on 8 GB systems.
Excludes 2015 (ICD-9/10 transition year).
Saves to epilepsy_cohort.parquet.
"""

import pyreadstat
import pandas as pd
import numpy as np
import re
import os
import time
import gc

savfile = "/Volumes/Niels 2/NIS_new_version/2011-2020 Updated with 2015 Proc.sav"
outdir  = "/Volumes/Niels 2/NIS_new_version/NIS_epilepsy_psych/output"

CHUNK_SIZE = 500_000

print("=" * 60)
print("NIS EPILEPSY COHORT EXTRACTION")
print("=" * 60)

# ============================================================
# PASS 1: Chunked read, filter epilepsy DX1 rows
# ============================================================
print(f"\nReading .sav in chunks of {CHUNK_SIZE:,} rows...")
t0 = time.time()

epi_chunks = []
total_rows = 0
year_counts = {}

reader = pyreadstat.read_file_in_chunks(pyreadstat.read_sav, savfile, chunksize=CHUNK_SIZE)

for i, (chunk, meta) in enumerate(reader):
    n = len(chunk)
    total_rows += n

    if 'YEAR' in chunk.columns:
        for y, c in chunk['YEAR'].value_counts().items():
            year_counts[y] = year_counts.get(y, 0) + c

    # Include 2015 — ICD version determined from DX1 code format below

    # Filter: Epilepsy in DX1
    dx1 = chunk['DX1'].astype(str).str.strip()
    mask = (
        dx1.str.match(r'^345', na=False) |      # ICD-9 epilepsy
        dx1.str.match(r'^G40', na=False)         # ICD-10 epilepsy
    )
    matched = chunk[mask].copy()
    if len(matched) > 0:
        epi_chunks.append(matched)

    elapsed = time.time() - t0
    epi_so_far = sum(len(c) for c in epi_chunks)
    print(f"  Chunk {i+1}: {total_rows:>12,} rows processed | Epilepsy found: {epi_so_far:,} | {elapsed:.0f}s", flush=True)

    del chunk, matched
    gc.collect()

t1 = time.time()
print(f"\nFinished reading in {t1-t0:.0f} seconds.")

print("\n=== YEAR DISTRIBUTION (all NIS) ===")
for y in sorted(year_counts.keys()):
    print(f"  {int(y)}: {year_counts[y]:>10,}")
print(f"  Total: {total_rows:>10,}")

epi = pd.concat(epi_chunks, ignore_index=True)
del epi_chunks
gc.collect()

print(f"\nTotal epilepsy DX1 (excl 2015): {len(epi):,}")

# ============================================================
# Ensure all DX/PR cols are string type
# ============================================================
dx_cols = [c for c in epi.columns if re.match(r'^(DX|I10_DX)\d+$', c)]
pr_cols = [c for c in epi.columns if re.match(r'^(PR|I10_PR)\d+$', c)]

for col in dx_cols + pr_cols:
    epi[col] = epi[col].fillna('').astype(str).str.strip()

# DX1 clean
epi['DX1'] = epi['DX1'].astype(str).str.strip()

# ============================================================
# ICD CODE SYSTEM FLAG
# ============================================================
# Determine ICD version from the actual DX1 code format, not year
# This correctly handles 2015 (mixed ICD-9 Jan-Sep / ICD-10 Oct-Dec)
dx1_clean = epi['DX1'].astype(str).str.strip()
epi['icd_version'] = np.where(dx1_clean.str.match(r'^[A-Z]', na=False), 10, 9)

print(f"\nICD-9 rows (DX1 starts with digit): {(epi['icd_version'] == 9).sum():,}")
print(f"ICD-10 rows (DX1 starts with letter): {(epi['icd_version'] == 10).sum():,}")
print(f"\nBy year and ICD version:")
print(pd.crosstab(epi['YEAR'], epi['icd_version']))

# ============================================================
# EPILEPSY TYPE CLASSIFICATION
# ============================================================
print("\n" + "=" * 60)
print("EPILEPSY TYPE CLASSIFICATION")
print("=" * 60)

def classify_epilepsy(row):
    dx1 = row['DX1']
    ver = row['icd_version']
    if ver == 10:
        if re.match(r'^G40[012]', dx1): return "Focal"
        if re.match(r'^G40[34]', dx1):  return "Generalized"
        if re.match(r'^G405', dx1):     return "Seizures_external"
        if re.match(r'^G40[AB]', dx1):  return "Generalized"  # Absence, JME
        if re.match(r'^G408', dx1):     return "Other"
        if re.match(r'^G409', dx1):     return "Unspecified"
    elif ver == 9:
        if re.match(r'^345[45]', dx1):  return "Focal"         # 345.4, 345.5
        if re.match(r'^345[0136]', dx1): return "Generalized"  # 345.0,1,3,6
        if re.match(r'^3452', dx1):     return "Generalized"   # petit mal status
        if re.match(r'^3457', dx1):     return "Focal"         # epilepsia partialis continua
        if re.match(r'^3458', dx1):     return "Other"
        if re.match(r'^3459', dx1):     return "Unspecified"
    return "Other"

epi['epilepsy_type'] = epi.apply(classify_epilepsy, axis=1)

# Intractable flag
def is_intractable(row):
    dx1 = row['DX1']
    ver = row['icd_version']
    if ver == 10:
        # G40.x11 or G40.x19 = intractable
        return bool(re.match(r'^G40.{1,2}1[19]$', dx1))
    elif ver == 9:
        # 345.x1 = intractable
        return bool(re.match(r'^345.\d1$', dx1))
    return False

epi['intractable'] = epi.apply(is_intractable, axis=1)

print("\nEpilepsy type:")
for t, n in epi['epilepsy_type'].value_counts().items():
    print(f"  {t}: {n:,} ({100*n/len(epi):.1f}%)")

print(f"\nIntractable: {epi['intractable'].sum():,} ({100*epi['intractable'].mean():.1f}%)")

# ============================================================
# EPILEPSY SURGERY PROCEDURES
# ============================================================
print("\n" + "=" * 60)
print("EPILEPSY SURGERY PROCEDURES")
print("=" * 60)

# -- Helper: check if any PR column matches pattern
def has_pr(df, patterns, pr_columns):
    """Return boolean Series: True if any PR col matches any pattern."""
    result = pd.Series(False, index=df.index)
    for col in pr_columns:
        if col not in df.columns:
            continue
        v = df[col]
        for pat in patterns:
            if pat.startswith('^') or '*' in pat:
                result = result | v.str.match(pat, na=False)
            else:
                result = result | (v == pat)
    return result

# --- Resective / Ablative ---
# NOTE: ICD-9 PCS codes in NIS have NO leading zeros (e.g., 153 not 0153)
resective_icd10 = ['00B70ZZ', '00B00ZZ']
resective_icd9 = ['^152$', '^153$', '^159$']

disconnection_icd10 = ['00870ZZ', '00800ZZ']
disconnection_icd9 = ['^132$']

litt_icd10 = ['D0Y0KZZ', '00504Z3', '00503ZZ']
litt_icd9 = ['^1761$']

# --- Neuromodulation ---
vns_lead_icd10 = ['00HE0MZ', '00HE3MZ']
vns_gen_icd10 = ['0JH60BZ', '0JH80BZ']
vns_lead_icd9 = ['^492$']
vns_gen_icd9 = ['^8694$', '^8697$']

rns_lead_icd10 = ['00H00MZ', '00H03MZ']
rns_gen_icd10 = ['0JH60DZ']
rns_lead_icd9 = ['^293$']
rns_gen_icd9 = ['^8695$', '^8698$']

# --- Monitoring ---
# Intracranial electrodes only (SEEG, grids/strips)
# EXCLUDED: 4A10X4Z (video-EEG) — routine diagnostic, not epilepsy surgery
# Note: 00H00MZ/00H03MZ overlap with RNS lead codes; hierarchical assignment resolves this
monitoring_icd10 = ['00H00MZ', '00H03MZ', '00H04MZ']
monitoring_icd9 = ['^293$']

# --- Video-EEG (separate flag, NOT counted as surgery) ---
veeg_icd10 = ['4A10X4Z']

# Flag each procedure type
epi['pr_resective'] = (
    has_pr(epi, resective_icd10, pr_cols) |
    has_pr(epi, resective_icd9, pr_cols) |
    has_pr(epi, disconnection_icd10, pr_cols) |
    has_pr(epi, disconnection_icd9, pr_cols) |
    has_pr(epi, litt_icd10, pr_cols) |
    has_pr(epi, litt_icd9, pr_cols)
)

epi['pr_vns'] = (
    has_pr(epi, vns_lead_icd10, pr_cols) |
    has_pr(epi, vns_gen_icd10, pr_cols) |
    has_pr(epi, vns_lead_icd9, pr_cols) |
    has_pr(epi, vns_gen_icd9, pr_cols)
)

epi['pr_rns'] = (
    has_pr(epi, rns_lead_icd10, pr_cols) |
    has_pr(epi, rns_gen_icd10, pr_cols) |
    has_pr(epi, rns_lead_icd9, pr_cols) |
    has_pr(epi, rns_gen_icd9, pr_cols)
)

epi['pr_monitoring'] = (
    has_pr(epi, monitoring_icd10, pr_cols) |
    has_pr(epi, monitoring_icd9, pr_cols)
)

# Surgical group assignment (hierarchical)
# Note: monitoring-only patients (intracranial electrodes without definitive surgery)
# are classified as "None" — too few cases and not true epilepsy surgery
def assign_surgery_group(row):
    if row['pr_resective']:
        return "Resective"
    if row['pr_rns']:
        return "RNS"
    if row['pr_vns']:
        return "VNS"
    return "None"

epi['surgery_group'] = epi.apply(assign_surgery_group, axis=1)
epi['any_surgery'] = epi['surgery_group'] != 'None'

print("\nSurgery group:")
for g in ['Resective', 'VNS', 'RNS', 'Monitoring', 'None']:
    n = (epi['surgery_group'] == g).sum()
    print(f"  {g}: {n:,} ({100*n/len(epi):.1f}%)")

print(f"\nAny surgery: {epi['any_surgery'].sum():,}")

# ============================================================
# PSYCHIATRIC COMORBIDITIES (any DX position)
# ============================================================
print("\n" + "=" * 60)
print("PSYCHIATRIC COMORBIDITIES")
print("=" * 60)

def has_dx(df, patterns_icd10, patterns_icd9, dx_columns):
    """Check if any DX column matches psychiatric code patterns."""
    result = pd.Series(False, index=df.index)
    is_icd10 = df['icd_version'] == 10
    for col in dx_columns:
        if col not in df.columns:
            continue
        v = df[col]
        for pat in patterns_icd10:
            result = result | (is_icd10 & v.str.match(pat, na=False))
        for pat in patterns_icd9:
            result = result | (~is_icd10 & v.str.match(pat, na=False))
    return result

# Depression
epi['psych_depression'] = has_dx(epi,
    patterns_icd10=[r'^F3[23]', r'^F341'],
    patterns_icd9=[r'^296[23]', r'^3004$', r'^311$'],
    dx_columns=dx_cols)

# Bipolar
epi['psych_bipolar'] = has_dx(epi,
    patterns_icd10=[r'^F31'],
    patterns_icd9=[r'^296[014568]', r'^2968'],
    dx_columns=dx_cols)

# Anxiety (any: GAD, panic, social, phobic, unspecified)
epi['psych_anxiety'] = has_dx(epi,
    patterns_icd10=[r'^F4[01]'],
    patterns_icd9=[r'^300[02]'],
    dx_columns=dx_cols)

# PTSD
epi['psych_ptsd'] = has_dx(epi,
    patterns_icd10=[r'^F431'],
    patterns_icd9=[r'^30981$'],
    dx_columns=dx_cols)

# OCD
epi['psych_ocd'] = has_dx(epi,
    patterns_icd10=[r'^F42'],
    patterns_icd9=[r'^3003$'],
    dx_columns=dx_cols)

# Schizophrenia spectrum
epi['psych_schizophrenia'] = has_dx(epi,
    patterns_icd10=[r'^F2[05]'],
    patterns_icd9=[r'^295'],
    dx_columns=dx_cols)

# Other psychosis
epi['psych_psychosis'] = has_dx(epi,
    patterns_icd10=[r'^F2[2389]'],
    patterns_icd9=[r'^29[78]'],
    dx_columns=dx_cols)

# ADHD
epi['psych_adhd'] = has_dx(epi,
    patterns_icd10=[r'^F90'],
    patterns_icd9=[r'^314'],
    dx_columns=dx_cols)

# Alcohol use disorder
epi['psych_alcohol'] = has_dx(epi,
    patterns_icd10=[r'^F10[12]'],
    patterns_icd9=[r'^303', r'^3050'],
    dx_columns=dx_cols)

# Drug use disorder
epi['psych_drug'] = has_dx(epi,
    patterns_icd10=[r'^F1[1-69][12]'],
    patterns_icd9=[r'^304', r'^305[2-9]'],
    dx_columns=dx_cols)

# Suicidal ideation
epi['psych_suicidal'] = has_dx(epi,
    patterns_icd10=[r'^R45851$'],
    patterns_icd9=[r'^V6284$'],
    dx_columns=dx_cols)

# PNES / conversion with seizures
epi['psych_pnes'] = has_dx(epi,
    patterns_icd10=[r'^F445$'],
    patterns_icd9=[r'^30011$'],
    dx_columns=dx_cols)

# Psychiatric disorder due to physiological condition (F06.x)
epi['psych_organic'] = has_dx(epi,
    patterns_icd10=[r'^F06'],
    patterns_icd9=[r'^293', r'^294'],
    dx_columns=dx_cols)

# Any psychiatric
psych_cols = [c for c in epi.columns if c.startswith('psych_')]
epi['any_psych'] = epi[psych_cols].any(axis=1)

print("\nPsychiatric comorbidity prevalence (unweighted):")
for col in psych_cols + ['any_psych']:
    n = epi[col].sum()
    label = col.replace('psych_', '').replace('any_psych', 'ANY PSYCHIATRIC')
    print(f"  {label:25s}: {n:>7,} ({100*n/len(epi):.1f}%)")

print("\nBy surgery status:")
for grp in ['Surgery', 'No Surgery']:
    mask = epi['any_surgery'] if grp == 'Surgery' else ~epi['any_surgery']
    sub = epi[mask]
    print(f"\n  {grp} (N={len(sub):,}):")
    for col in psych_cols + ['any_psych']:
        n = sub[col].sum()
        label = col.replace('psych_', '').replace('any_psych', 'ANY PSYCHIATRIC')
        print(f"    {label:25s}: {n:>7,} ({100*n/len(sub):.1f}%)")

# ============================================================
# COMORBIDITIES (Elixhauser-style)
# ============================================================
print("\n" + "=" * 60)
print("MEDICAL COMORBIDITIES")
print("=" * 60)

# Hypertension
epi['cm_hypertension'] = has_dx(epi,
    patterns_icd10=[r'^I1[0-3]'],
    patterns_icd9=[r'^40[1-5]'],
    dx_columns=dx_cols)

# Diabetes
epi['cm_diabetes'] = has_dx(epi,
    patterns_icd10=[r'^E1[01]'],
    patterns_icd9=[r'^250'],
    dx_columns=dx_cols)

# Obesity
epi['cm_obesity'] = has_dx(epi,
    patterns_icd10=[r'^E66'],
    patterns_icd9=[r'^2780'],
    dx_columns=dx_cols)

# Cerebrovascular disease
epi['cm_cvd'] = has_dx(epi,
    patterns_icd10=[r'^I6[0-9]'],
    patterns_icd9=[r'^43[0-8]'],
    dx_columns=dx_cols)

# Brain tumor
epi['cm_brain_tumor'] = has_dx(epi,
    patterns_icd10=[r'^C71', r'^D33[0-2]', r'^D43[0-2]'],
    patterns_icd9=[r'^191', r'^225[01]', r'^237[015]'],
    dx_columns=dx_cols)

cm_cols = [c for c in epi.columns if c.startswith('cm_')]
for col in cm_cols:
    n = epi[col].sum()
    label = col.replace('cm_', '')
    print(f"  {label:25s}: {n:>7,} ({100*n/len(epi):.1f}%)")

# ============================================================
# DEMOGRAPHICS & OUTCOMES SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DEMOGRAPHICS")
print("=" * 60)

if 'AGE' in epi.columns:
    print(f"  Age mean: {epi['AGE'].mean():.1f} (SD {epi['AGE'].std():.1f})")
if 'FEMALE' in epi.columns:
    print(f"  Female: {(epi['FEMALE']==1).sum():,} ({100*(epi['FEMALE']==1).mean():.1f}%)")
if 'DIED' in epi.columns:
    print(f"  Mortality: {(epi['DIED']==1).sum():,} ({100*(epi['DIED']==1).mean():.1f}%)")
if 'LOS' in epi.columns:
    print(f"  LOS mean: {epi['LOS'].mean():.1f} (SD {epi['LOS'].std():.1f})")

print("\nBy year:")
for y in sorted(epi['YEAR'].dropna().unique()):
    sub = epi[epi['YEAR'] == y]
    print(f"  {int(y)}: {len(sub):>8,}")

# ============================================================
# SAVE
# ============================================================
parquet_path = os.path.join(outdir, "epilepsy_cohort.parquet")
epi.to_parquet(parquet_path, index=False)
print(f"\nSaved to: {parquet_path}")
print(f"File size: {os.path.getsize(parquet_path)/1e6:.1f} MB")
print(f"Columns: {epi.shape[1]}")
print(f"Rows: {epi.shape[0]:,}")
print(f"\nTotal time: {time.time()-t0:.0f} seconds")
print("DONE.")
