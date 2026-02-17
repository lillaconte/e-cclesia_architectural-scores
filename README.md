# EccLexica : Christian Architectural Vocabulary Detection in Latin Texts

This repository contains a dictionary-based method for identifying architectural discourse in Late Antique and Medieval Latin texts. It is applied to a test corpus of Latin inscriptions.

The workflow combines a **dictionary-based approach** with an **evaluation against GLiNER named-entity recognition models**.

EccLexica is being developed as part of the ANR E-cclesia project.

<div align="center">
  <img src="assets/ANR_Logo.svg.png" alt="logo anr" width="300" />
   <img src="assets/logo_ecclesia.jpg" alt="logo anr" width="300" />
</div>


---

## Repository Structure

```text
.
├── data/
│   └── dicts
│     └── auto_terms.csv
│     └── asso_terms.csv
│     └── mat_terms.csv
│   └── sample.csv
│
├── output/
│
├── arch-score.py
├── gliner-eval.py
└── README.md
```

---

## Data

- **Main dataset**

The main dataset is not stored in this repository due to its size.

Please download **EDCS_text_cleaned_2022-09-12.json** from Zenodo:

👉 https://zenodo.org/records/7072337

After downloading, place the file in the following location:

data/EDCS_text_cleaned_2022-09-12.json

- **Sample dataset** (`data/sample.csv`)
  - 100 inscriptions.
  - Used for testing and GLiNER evaluation.

---

## Scripts

### `arch-score.py`

Computes an **architectural concentration score (0–100)** for each inscription based on the presence and distribution of architectural vocabulary.

**Main steps:**
1. **Filtering**
   - Date range: 399–1199 CE.
   - Excludes a predefined list of eastern provinces.
2. **Lemmatization**
   - Uses spaCy’s Latin model (`la_core_web_md`) on cleaned interpretive text.
3. **Scoring**
   - Uses three term categories:
     - *Autonomous* (e.g. `basilica`, `ecclesia`)
     - *Associative* (e.g. `murus`, `porta`)
     - *Material* (e.g. `marmor`, `aurum`)
   - Score combines:
     - Term count
     - Co-occurrence of term types
     - Proximity of terms in the text
     - Term density
4. **Output generation**
   - Full scored dataset.
   - Score-based subsets (per 10-point bin).
   - High-score subset (score > 50).

**Outputs (CSV):**
- `output/edcs_architectural_scores.csv`
- `output/edcs_architectural_scores_<bin>_<bin+9>.csv`
- `output/edcs_architectural_scores_gt50.csv`
- `output/edcs_filtered_inscriptions.csv`

**Key added columns:**
- `lemmatized_text`
- `arch_score`
- `autonomous_terms`
- `associative_terms`
- `material_terms`

---

### `gliner-eval.py`

Evaluates **GLiNER NER models** against the dictionary-based architectural terms.

**What it does:**
- Loads a scored sample dataset from `output/sample.csv`.
- Runs multiple GLiNER models on:
  - Lemmatized text
  - Cleaned interpretive text
- Compares GLiNER-extracted entities with dictionary-based terms.

**Evaluated labels:**
- `building/type`
- `building/part`
- `building/material`

**Metrics computed:**
- Precision
- Recall
- F1 score
- Percentage and count of overlapping terms
- Number of inscriptions with at least one shared term

**Outputs:**
- Printed comparison table in the console
- `output/gliner_comparison_metrics_detailed.csv`

---

## Requirements

- Python 3.9+
- Core libraries:
  - `pandas`, `numpy`, `scikit-learn`
  - `spacy` (+ `la_core_web_md`)
  - `gliner`

---

## Typical Workflow

1. Run `arch-score.py` to filter inscriptions and compute architectural scores.
2. Inspect or subset results in the `output/` directory.
3. Run `gliner-eval.py` on a scored sample to compare dictionary-based detection with GLiNER models.

---

## Notes

- Scores are heuristic and intended for **comparative and exploratory analysis**, not absolute classification.

---

## Citations 

Heřmánková, Petra. “EDCS”. Zenodo, September 12, 2022. https://doi.org/10.5281/zenodo.7072337.

