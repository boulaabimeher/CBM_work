# Concept Bottleneck Models (CBM) – Medical Imaging Datasets Exploration

This repository contains an exploratory and preparatory study of multiple medical imaging datasets for future research on **Concept Bottleneck Models (CBM)** and concept-based learning.

The main objective of this work is to:
- understand how **high-level medical concepts** are defined in existing datasets,
- analyze how these concepts are related to the final **task labels (Y)**,
- and prepare clean and reusable data representations for future CBM pipelines.

Each notebook focuses on one dataset and provides:
- a detailed dataset overview,
- an explicit separation between **concept annotations (C)** and **task labels (Y)**,
- an analysis of metadata and available visual information,
- and a curated list of reference papers (4 papers per dataset) to better understand how the dataset is used in the literature.

This repository is designed as a foundation for future work on:
**interpretable learning, concept prediction (X → C), and concept-based reasoning (C → Y)** in medical imaging.

---

## Datasets explored

The repository includes four main notebooks:

### 1. MIMIC-CXR

**Notebook:** `MIMIC-CXR_overview.ipynb`

Chest X-ray dataset with structured radiology labels and clinical concepts.

This notebook covers:
- dataset structure and file organization,
- metadata and label distributions,
- medical concepts extracted from structured annotations,
- separation of:
  - image → concepts (C),
  - concepts → diagnostic labels (Y),
- discussion of how these concepts can be used for CBM training.

Four reference papers related to MIMIC-CXR and concept-based or interpretable learning are included at the end of the notebook.

---

### 2. CheXpert

**Notebook:** `chexpert_overview.ipynb`

Large-scale chest radiography dataset with uncertainty-aware labels.

This notebook provides:
- analysis of uncertainty labels and their interpretation,
- study of medical findings as candidate concepts,
- definition of:
  - concept matrix (C),
  - target label(s) (Y),
- visualization and statistical exploration of the dataset,
- preparation guidelines for concept prediction models.

Four reference papers related to CheXpert and interpretability / weak supervision are included.

---

### 3. Derm7pt

**Notebook:** `derm7pt_overview.ipynb`

Dermoscopic skin lesion dataset based on the **7-point checklist**.

This dataset is particularly well-suited for CBM because it is explicitly designed around human-interpretable criteria.

The notebook includes:
- description of the seven diagnostic criteria as concepts,
- mapping between visual attributes (concepts) and final diagnosis,
- separation of:
  - concepts (C),
  - diagnosis labels (Y),
- analysis of concept distributions and correlations.

Four reference papers related to Derm7pt and concept-based dermatology diagnosis are provided.

---

### 4. HAM10000

**Notebook:** `ham10000_overview.ipynb`

Large dermoscopic image dataset for skin lesion classification.

This notebook focuses on:
- dataset organization and metadata,
- label taxonomy and class imbalance,
- construction of candidate concepts from available annotations and metadata,
- definition of concept sets (C) and classification labels (Y),
- discussion on how HAM10000 can be adapted for concept-based learning even though it was not originally designed for CBM.

Four reference papers related to HAM10000 and concept-based or interpretable dermatology models are included.

---

## Methodology and perspective

For each dataset, the notebooks follow the same methodology:

1. Dataset inspection and metadata analysis  
2. Identification of medically meaningful concepts  
3. Explicit separation between:
   - **X** : input images,
   - **C** : concept annotations or derived concept variables,
   - **Y** : final prediction targets  
4. Statistical and visual exploration of concepts and labels  
5. Literature review (4 papers per dataset)

This unified structure makes the datasets directly reusable for:
- sequential CBM training (X → C → Y),
- joint CBM training,
- concept probing and concept attribution studies.

---

## Research motivation

This repository supports my ongoing research on:
**interpretability and concept-based representation learning for medical image analysis**.

The goal is to build reliable and clinically meaningful concept bottlenecks that help:
- improve model transparency,
- facilitate error analysis,
- and enable medically grounded decision support.

---
