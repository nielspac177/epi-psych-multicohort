# epi-psych-multicohort

Analysis code accompanying the manuscript:

> *Prevalence and Postoperative Trajectory of Psychiatric Comorbidities in Epilepsy Surgery: A Multi-Cohort Study*
> Pacheco-Barrios N, Limbania D, Mathew GT, Kumar I, Mazurek MH, Jha R, Glazer A, Rolston JD.
> (Under review.)

## Scope

Three complementary cohorts:

| Folder        | Source                                      | Language       | Data access                                          |
|---------------|---------------------------------------------|----------------|------------------------------------------------------|
| `bwh/`        | Brigham and Women's Hospital chart review   | Python         | Institutional chart review (not shareable)           |
| `mimic_iv/`   | MIMIC-IV v3.1 (BIDMC)                       | Python + Jupyter | Credentialed access via PhysioNet (DUA required)   |
| `nis/`        | HCUP National Inpatient Sample (2012-2020)  | R + Python     | Licensed access via HCUP Central Distributor (DUA)   |

## What's included

- Cohort identification, ICD mapping, and feature engineering
- MIMIC-IV propensity-score matching (Analyses A / B / C), balance diagnostics, love plots, McNemar tests
- MIMIC-IV multivariable logistic regression (with and without insurance)
- NIS survey-weighted prevalence and temporal trend models
- NIS ZIP-code income-quartile analysis
- BWH neighborhood-deprivation analysis (SVI, SDI, ICE)
- BWH seizure-outcome → psychiatric-improvement model

## What's NOT included

No data. All scripts expect the user to supply their own credentialed MIMIC-IV / NIS / BWH extracts.
`*.csv`, `*.parquet`, `*.duckdb`, `*.sav`, `*.xlsx`, `*.rds` are `.gitignore`-d under the DUAs.

## Running the code

Paths are hardcoded to the authors' local mount points (`/Volumes/Niels 2/...`). To reproduce:

1. Clone this repo.
2. Obtain each dataset through the proper channel (PhysioNet for MIMIC-IV; HCUP for NIS; institutional chart review for BWH).
3. Edit the `ROOT`, `ANA`, `DB_PATH`, `datadir`, `outdir`, and `OUT` constants at the top of each script to point to your local data.
4. For `bwh/deprivation_indices_analysis.py`, request a free Census Data API key and paste it into `CENSUS_API_KEY`.

## Environment

- Python 3.11+ (`duckdb`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`, `nbformat`)
- R 4.4+ (`survey`, `MatchIt`, `tableone`, `boot`, `ggplot2`)

## License

Code is released under the MIT License (see `LICENSE`). The underlying datasets remain governed by their respective DUAs.

## Contact

Niels Pacheco-Barrios — nielspacheco1997@gmail.com
John D. Rolston, MD PhD — jrolston@bwh.harvard.edu
