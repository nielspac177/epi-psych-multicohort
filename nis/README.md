# NIS analyses

Survey-weighted analyses of the HCUP National Inpatient Sample (2012-2020, excluding 2015 ICD transition year).

## Pipeline order

1. `extract_epilepsy.py` — extract epilepsy cohort from the raw NIS SPSS file; chunked reader for low-memory systems.
2. `01_cohort_building.ipynb` — build analytic cohort, define psychiatric comorbidity flags.
3. `02_prevalence.ipynb` — survey-weighted prevalence of each psychiatric comorbidity, surgical vs non-surgical.
4. `03_trends.ipynb` — year-on-year temporal trends, OR per calendar year.
5. `04_outcomes.ipynb` — in-hospital outcomes by psychiatric status.
6. `05_trends_ses.py` — combined temporal + PNES audit + SES (ZIP income quartile) replication.
7. `06_jama_trends_figure.py` — two-panel JAMA-style temporal trends figure (surgical / non-surgical).

## Data access

NIS requires a HCUP data-use agreement: <https://hcup-us.ahrq.gov/tech_assist/centdist.jsp>.

The R notebooks use the `survey` package for Taylor-linearized variance with `HOSP_NIS` cluster, `NIS_STRATUM`, and `DISCWT` weights.

## Paths to edit

Each script begins with `datadir`, `outdir`, `ROOT`, and/or `savfile` constants pointing to the authors' mount point. Update them to match your NIS install.
