"""
ICD-9 and ICD-10 code definitions for epilepsy and psychiatric comorbidity study.
These are used as prefix matches (e.g., 'G40' matches G40.001, G40.101, etc.)
"""

# =============================================================================
# EPILEPSY DIAGNOSIS CODES
# =============================================================================

EPILEPSY_ICD10 = {
    "G40": "Epilepsy and recurrent seizures (all subtypes)",
}

EPILEPSY_ICD9 = {
    "345": "Epilepsy and recurrent seizures (all subtypes)",
}

# Drug-resistant / intractable epilepsy (more likely surgical candidates)
# ICD-10: 5th character = 1 means intractable
INTRACTABLE_EPILEPSY_ICD10 = [
    "G4001", "G4011", "G4021", "G4031", "G4041", "G4051",
    "G4081", "G4091", "G40A1", "G40B1",
]
# ICD-9: 5th digit = 1 means intractable
INTRACTABLE_EPILEPSY_ICD9 = [
    "34501", "34511", "34541", "34551", "34561", "34571", "34581", "34591",
]

# =============================================================================
# EPILEPSY SURGERY PROCEDURE CODES
# =============================================================================

# ICD-10-PCS procedure codes
EPILEPSY_SURGERY_ICD10_PCS = {
    # Temporal lobectomy / brain excision
    "00BT0ZZ": "Excision of temporal lobe, open",
    "00BT3ZZ": "Excision of temporal lobe, percutaneous",
    "00BT4ZZ": "Excision of temporal lobe, percutaneous endoscopic",
    "00B70ZZ": "Excision of cerebral hemisphere, open",
    "00B74ZZ": "Excision of cerebral hemisphere, percutaneous endoscopic",
    "00BV0ZZ": "Excision of frontal lobe, open",
    "00BW0ZZ": "Excision of parietal lobe, open",
    "00BX0ZZ": "Excision of occipital lobe, open",
    "00B00ZZ": "Excision of brain, open",
    # Hemispherectomy
    "00T70ZZ": "Resection of cerebral hemisphere (hemispherectomy)",
    # Corpus callosotomy
    "00880ZZ": "Division of basal ganglia/corpus callosum, open",
    # Laser ablation / destruction
    "00550ZZ": "Destruction of brain, open",
    "00553ZZ": "Destruction of brain, percutaneous",
    "00554ZZ": "Destruction of brain, percutaneous endoscopic",
    # VNS implantation
    "00HE0MZ": "Insertion neurostimulator lead, cranial nerve, open",
    "00HE3MZ": "Insertion neurostimulator lead, cranial nerve, percutaneous",
    "0JH60BZ": "Insertion stimulator generator, chest subcutaneous",
    "0JH80BZ": "Insertion stimulator generator, abdomen subcutaneous",
    # Intracranial electrode placement (SEEG, grids)
    "00H00MZ": "Insertion neurostimulator lead, brain, open",
    "00H03MZ": "Insertion neurostimulator lead, brain, percutaneous",
}

# ICD-9 procedure codes
EPILEPSY_SURGERY_ICD9 = {
    "0152": "Hemispherectomy",
    "0153": "Lobectomy of brain",
    "0159": "Other excision/destruction of brain lesion (focal resection)",
    "0293": "Implantation intracranial neurostimulator",
    "0294": "Insertion subcutaneous neurostimulator pulse generator",
    "0492": "Implantation peripheral neurostimulator (VNS)",
    "0124": "Burr holes with electrode insertion",
    "0125": "Craniotomy with electrode insertion (subdural grid/strip)",
}

# =============================================================================
# PSYCHIATRIC DISORDER CODES (DSM-5 AXIS I)
# =============================================================================

PSYCH_CATEGORIES = {
    "depression": {
        "label": "Depressive Disorders",
        "icd10": ["F32", "F33", "F341"],  # MDD single, recurrent, dysthymia
        "icd9": ["2962", "2963", "3004", "311"],  # MDD single/recurrent, dysthymia, NOS
    },
    "bipolar": {
        "label": "Bipolar Disorders",
        "icd10": ["F31", "F340"],  # Bipolar, cyclothymia
        "icd9": ["2960", "2961", "2964", "2965", "2966", "2967", "29680", "29689"],
    },
    "anxiety": {
        "label": "Anxiety Disorders",
        "icd10": ["F40", "F41"],  # Phobic + other anxiety
        "icd9": ["30000", "30001", "30002", "30021", "30022", "30023", "30029"],
    },
    "ptsd": {
        "label": "PTSD / Trauma-Related",
        "icd10": ["F431"],  # PTSD
        "icd9": ["30981"],
    },
    "ocd": {
        "label": "Obsessive-Compulsive Disorders",
        "icd10": ["F42"],
        "icd9": ["3003"],
    },
    "psychotic": {
        "label": "Schizophrenia / Psychotic Disorders",
        "icd10": ["F20", "F21", "F22", "F23", "F24", "F25", "F28", "F29"],
        "icd9": ["295", "297", "298"],
    },
    "organic_psych": {
        "label": "Psychiatric Disorder Due to Medical Condition",
        "icd10": ["F060", "F062", "F063", "F064"],  # Psychosis/mood/anxiety due to medical condition
        "icd9": ["29382", "29383", "29384"],
    },
    "adhd": {
        "label": "ADHD",
        "icd10": ["F90"],
        "icd9": ["31400", "31401", "3149"],
    },
    "substance_use": {
        "label": "Substance Use Disorders",
        "icd10": ["F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"],
        "icd9": ["303", "304", "305"],
    },
    "suicidal_ideation": {
        "label": "Suicidal Ideation",
        "icd10": ["R45851"],
        "icd9": ["V6284"],
    },
    "pnes": {
        "label": "Psychogenic Non-Epileptic Seizures (flag/exclude)",
        "icd10": ["F445"],
        "icd9": ["30011"],
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_icd_prefix_sql(codes, column="icd_code", version=None, version_column="icd_version"):
    """
    Build a SQL WHERE clause for ICD prefix matching.

    Args:
        codes: list of ICD code prefixes
        column: column name containing ICD codes
        version: 9 or 10 (if None, no version filter)
        version_column: column name containing ICD version

    Returns:
        SQL WHERE clause string
    """
    conditions = [f"{column} LIKE '{code}%'" for code in codes]
    where = "(" + " OR ".join(conditions) + ")"
    if version is not None:
        where = f"({version_column} = {version} AND {where})"
    return where


def get_all_psych_codes(version=10):
    """Get all psychiatric ICD codes (excluding PNES) for a given ICD version."""
    key = "icd10" if version == 10 else "icd9"
    all_codes = []
    for cat_name, cat_data in PSYCH_CATEGORIES.items():
        if cat_name == "pnes":  # Exclude PNES from "any psychiatric" count
            continue
        all_codes.extend(cat_data[key])
    return all_codes
