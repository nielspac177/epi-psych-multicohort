# BWH analyses

Scripts for the single-center Brigham and Women's Hospital surgical cohort.

| File                              | Purpose                                                                               |
|-----------------------------------|---------------------------------------------------------------------------------------|
| `deprivation_indices_analysis.py` | Builds SVI / SDI / ICE from ACS data and tests association with psychiatric comorbidity |
| `epilepsy_outcomes_analysis.py`   | Seizure outcomes + the postoperative psychiatric-improvement model (OR for seizure freedom) |

## Input

Both scripts expect `Epilepsy_surgery_rohan_dataset_v1.xlsx` (BWH chart-review extract) in the working directory. The dataset is PHI and is not shared.

## Census API key

`deprivation_indices_analysis.py` queries the Census Data API. Request a free key at <https://api.census.gov/data/key_signup.html> and paste it into the `CENSUS_API_KEY` constant near the top of the script.
