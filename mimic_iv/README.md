# MIMIC-IV analyses

Propensity-score matched comparison of surgical vs non-surgical epilepsy patients in MIMIC-IV v3.1.

## Pipeline order

1. `01_cohort_identification.ipynb` — identify epilepsy patients, surgical status, psychiatric comorbidities.
2. `02_visualizations.ipynb` — exploratory prevalence plots.
3. `03_psm_analysis.py` — main PSM pipeline (Analyses A and B): builds features, runs 1:1 caliper matching, McNemar tests on psychiatric categories, PNES verification.
4. `04_psm_figures.py` — forest and prevalence-bar figures for Analyses A and B.
5. `05_temporal_split.py` — psychiatric prevalence across de-identified time bins.
6. `06_balance_prepost.py` — pre/post PSM covariate-balance table (LaTeX).
7. `07_logreg.py` — simple logistic regression (surgery + age + sex) on full and matched cohorts.
8. `08_loveplot_prepost.py` — pre vs post SMD love plot.
9. `09_logreg_full.py` — multivariable logistic regression with all PSM covariates (including insurance).
10. `10_psm_C_no_insurance.py` — Analysis C: PSM and multivariable logreg with demographics + severity only.

Shared: `icd_codes.py` (ICD-9/10 definitions for epilepsy, intractability, focal, status epilepticus, psychiatric comorbidities).

## Data access

MIMIC-IV requires PhysioNet credentialed access: <https://physionet.org/content/mimiciv/3.1/>.

## Paths to edit

Every script begins with a `ROOT` / `ANA` / `OUT` constant pointing to the authors' mount point. Update to point to your MIMIC-IV install.
