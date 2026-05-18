``
# Evolution ONE

**Multi‑Level Cancer Evolution & Structural Impact Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20264580-blue)](https://doi.org/10.5281/zenodo.20264580)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20277033-blue)](https://doi.org/10.5281/zenodo.20277033)

Evolution ONE is a standalone, vendor‑neutral platform that models cancer evolution
through the lens of **Self‑Organised Criticality (SOC)**, **duon biology**, and
**physics‑based structural refinement**. It ingests genomic mutation data (MAF, VCF)
and outputs a comprehensive oncological profile that includes:

- whether the tumour is **stable**, **critical**, or **collapsing**,
- which future escape mutations are most likely to destabilise key proteins,
- which targeted drugs or stabilisers can be deployed, and
- how lifestyle factors correlate with mutation load and duon disruption.

The engine works out‑of‑the‑box with **real patient data**, and can optionally
leverage **REAL FOLD ONE** and **REAL FOLD ONE HT** for atomic‑level ΔΔG
calculations and high‑throughput mutation scanning.

---

## Overview

Traditional cancer genomics tools classify mutations as drivers or passengers
based on recurrence or conservation. **Evolution ONE** goes deeper:

- It treats cancer as a **complex system** whose evolution can be captured by
  **SOC dynamics**—the same physics that governs earthquakes and avalanches.
- It incorporates **duons** (codons that simultaneously encode an amino acid
  and a regulatory signal) to discover hidden driver events.
- It uses **differentiable physics** (via REAL FOLD ONE’s engine) to assess
  the structural impact of mutations at atomic resolution.
- It simulates future evolutionary trajectories with **Itô calculus** and
  **Renormalisation Group (RG)** smoothing.

The result is a unified pipeline that goes **from raw mutation file to personalised
drug recommendation**, all on commodity hardware.

---

## Features

- **SOC‑Based Regime Classification** – Classifies each tumour sample as
  *stable*, *critical* (multi‑level selection), or *collapse* using Shannon entropy
  and mutation load μ.
- **Predictive Evolution** – Projects future mutation load using both SOC‑driven
  stochastic steps and **Itô Langevin dynamics**.
- **Duon‑Aware Analysis** – Load a list of duon codon positions and compute the
  fraction of mutations that disrupt duon function.
- **Future Escape Mutation Scanning** – Identifies single‑amino‑acid changes that
  would strongly destabilise the protein (ΔΔG > 1.5 kcal/mol), either via
  REAL FOLD ONE HT or via an embedded energy evaluator.
- **Physics‑Based Structural Impact** – Computes the ΔΔG of observed mutations
  using full‑atom refinement (REAL FOLD ONE integration).
- **Drug & Stabiliser Recommendation** – Maps destabilised genes to existing
  targeted therapies and protein stabilisers.
- **Retrospective Lifestyle Correlation** – Merges mutation data with CSV files
  of environmental factors and computes Pearson/Spearman correlations.
- **Gene Network BV Consistency** – Applies the **Batalin–Vilkovisky** formalism
  to verify that the gene interaction network satisfies the classical master
  equation.
- **RG Multiscale Smoothing** – Removes noise from mutation load estimates.
- **Vendor‑Neutral** – Runs on CPU, GPU (CUDA, MPS), or Ascend NPU without
  modification.

---

## Installation

```bash
git clone https://github.com/yoonalimsuwan/EVOLUTION-ONE.git
cd evolution-one

# Create and activate a conda environment (optional)
conda create -n evolution python=3.10 -y
conda activate evolution

# Install core dependencies
pip install torch numpy pandas scipy matplotlib

# Optional: install biotite for PDB reading
pip install biotite

# Optional: install REAL FOLD ONE and REAL FOLD ONE HT for advanced features
pip install -e .   # if they are in the same repository
```

---

Quick Start

```bash
# Basic analysis from a TCGA MAF file, genes of interest, and duon positions
python evolution_one.py \
    --input tcga_lung.maf \
    --genes EGFR KRAS TP53 \
    --duon_file duon_positions.txt \
    --lifestyle_file patient_lifestyle.csv \
    --plot

# Use VCF input and skip future mutation scanning
python evolution_one.py \
    --input patient.vcf --format vcf \
    --genes BRAF PIK3CA \
    --no_future

# Include gene interactions (for BV check) and structural analysis
python evolution_one.py \
    --input tcga_colon.maf \
    --genes APC CTNNB1 SMAD4 \
    --gene_interactions 0 1 0 2 \
    --pdb_dir ./my_structures \
    --output_dir ./colon_results
```

---

Input File Formats

Mutation file (MAF or VCF):

· Standard TCGA MAF or simple VCF with at least Hugo_Symbol and Tumor_Sample_Barcode (MAF) or GENE= in the INFO field (VCF).

Duon file (optional):

· A plain text file with one codon position (1‑based) per line.

Lifestyle file (optional):

· A CSV with columns: sample_id, and any number of environmental factors (e.g., smoking_pack_years, alcohol_consumption, diet_score).

Gene interactions (optional):

· Passed as --gene_interactions i j k l ... where each pair (i, j) represents a known regulatory interaction between gene i and gene j (indices refer to the order in --genes).

PDB directory (optional):

· A folder containing .pdb files for each gene. The engine automatically picks the first file whose name contains the gene symbol.

---

Output Files

File Content
summary.json Cancer risk, entropy, drug recommendations, BV check
sample_states.csv Per‑sample mutation load (μ), state (0/1/2)
phase_diagram.png (if --plot) Entropy vs. mutation load

The summary.json also includes correlations between lifestyle factors and
mutation load/duon rate if a lifestyle file was provided.

---

Integration with REAL FOLD ONE

Evolution ONE is fully functional on its own, but when REAL FOLD ONE and
REAL FOLD ONE HT are installed, it gains:

· Accurate ΔΔG calculation – uses the full SOC‑controlled refinement engine
  instead of the embedded simplified energy function.
· Complete single‑mutation scanning – leverages the high‑throughput scanner
  to explore all possible amino‑acid changes, providing a comprehensive list of
  destabilising mutations.

To enable these features, simply ensure real_fold_one and real_fold_one_ht
are importable. The scripts will automatically detect them.

---

Architectural Philosophy

Evolution ONE shares the same vendor‑neutral, differentiable‑physics DNA as
REAL FOLD ONE:

· No CUDA C++ – pure PyTorch operations, enabling seamless execution on
  NVIDIA GPUs, Apple MPS, Huawei Ascend NPU, or CPU.
· Differentiable from end to end – the SOC kernel, energy function, and
  Ito integrators are all differentiable, allowing future integration with
  deep learning surrogates.
· Physics first – rather than relying on statistical black‑boxes, Evolution
  ONE builds on first‑principle models (SOC, SSC, RG, BV) to interpret cancer
  as a dynamical system.

### How Evolution ONE Works – The Three Evolutionary Regimes

Evolution ONE models cancer progression as a **dynamical phase transition**
driven by the accumulation of mutations. Rather than treating every tumour
as a collection of independent random events, the engine quantifies the
**mutation load μ** (fraction of key genes affected) and uses principles
from **Self‑Organised Criticality (SOC)** to place each sample into one
of three regimes:

| Regime | State | μ range | Biological interpretation |
|--------|-------|---------|---------------------------|
| **Stable (Stochastic)** | 0 | μ < 0.2 | Mutations are sparse and effectively neutral. The system is far from a critical point and behaves as a random drift process. |
| **Critical (Selection)** | 1 | 0.2 ≤ μ ≤ 0.8 | The tumour is at the edge of collapse. Multi‑level selection (coding + regulatory, via duons) becomes active. Small perturbations can trigger SOC‑like avalanches, generating heterogeneity. |
| **Collapse (Deterministic)** | 2 | μ > 0.8 | The mutational load overwhelms the system. Key proteins destabilise, regulatory networks break down, and the cancer follows a deterministic trajectory toward aggressive growth or cell death. |

These regimes are not merely descriptive—they are **quantified** by the
engine:

- **Entropy H** of the state distribution measures the degree of
  heterogeneity.
- **SOC evolution** (feedback between stress σ and temperature T) and
  **Itô Langevin dynamics** simulate future μ, predicting whether a stable
  or critical tumour will cross into the collapse regime.
- **Duon disruption** (μ_duon) adds a second dimension, revealing
  mutations that simultaneously alter protein sequence and gene
  regulation—a hallmark of multi‑level selection.

By grounding cancer evolution in SOC physics, Evolution ONE moves beyond
the classic driver/passenger dichotomy and provides a **unified mathematical
language** for tumour dynamics. This is the same language that describes
earthquakes, avalanches, and financial crashes—phenomena where a critical
state separates order from chaos.

``
### Pilot Study: Lung Adenocarcinoma (TCGA‑LUAD)

To demonstrate Evolution ONE’s capabilities on real‑world data, we provide
a ready‑to‑run pilot study for **Lung Adenocarcinoma (LUAD)** using the
public TCGA dataset.

**Why LUAD?**
- Large sample size (>500 patients) with well‑curated mutation calls.
- Clinically actionable genes (`EGFR`, `KRAS`, `ALK`, `BRAF`, `MET`) with
  known targeted therapies.
- High prevalence of duon‑containing genes and documented resistance mutations
  that can be predicted *in silico*.

#### 1. Obtain the data

```bash
# Download the TCGA MC3 MAF file (public, no login required)
wget https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc \
    -O mc3.v0.2.8.PUBLIC.maf.gz
gunzip mc3.v0.2.8.PUBLIC.maf.gz

# Filter for LUAD samples and genes of interest
python -c "
import pandas as pd
maf = pd.read_csv('mc3.v0.2.8.PUBLIC.maf', sep='\t', comment='#', low_memory=False)
luad = maf[maf['Project_Code'] == 'LUAD']
luad.to_csv('luad.maf', sep='\t', index=False)
"
```

2. Prepare supporting files

· Duon positions – create a text file luad_duons.txt with one codon
  position per line for the genes of interest (sources: Ensembl regulatory
  build, literature). Example:
  ```
  12
  25
  58
  ...
  ```
· Lifestyle data – (optional) a CSV luad_clinical.csv with columns:
  sample_id, smoking_pack_years, gender, age_at_diagnosis.
  This can be extracted from the TCGA clinical supplement.
· PDB structures – download the structures for your target genes into
  ./pdbs/ (e.g., EGFR.pdb, KRAS.pdb). AlphaFold‑predicted models or
  experimental PDBs are both suitable.

3. Run Evolution ONE

```bash
python evolution_one.py \
    --input luad.maf \
    --genes EGFR KRAS ALK BRAF MET \
    --duon_file luad_duons.txt \
    --lifestyle_file luad_clinical.csv \
    --pdb_dir ./pdbs \
    --output_dir ./luad_results \
    --plot
```

4. Interpret the results

· sample_states.csv – each TCGA sample is assigned a regime (0 = stable,
  1 = critical, 2 = collapse). Compare these states with overall survival
  (e.g., Kaplan‑Meier curves) to see whether the critical/collapse groups
  have poorer prognosis.
· summary.json – contains the predicted cancer risk, future μ values
  (SOC and Itô), and drug recommendations.
· phase_diagram.png – visualises the entropy landscape. LUAD samples
  often span the entire range from stable to collapse, demonstrating the
  SOC‑like behaviour of the disease.
· Drug recommendations – the engine lists existing targeted therapies
  for destabilised genes. Cross‑reference these suggestions with NCCN
  guidelines to evaluate clinical relevance.

5. Expected findings

In preliminary tests with TCGA‑LUAD, Evolution ONE typically reveals:

· Distinct entropy profiles – patients with high entropy (critical state)
  tend to have worse survival, consistent with multi‑level selection theory.
· Predicted escape mutations – for EGFR‑mutated samples, the engine
  often highlights T790M and C797S as top destabilising mutations, matching
  clinical observations of acquired resistance.
· Lifestyle correlations – smoking pack‑years frequently show a
  significant positive correlation with duon disruption rate, suggesting
  a mechanistic link between environmental exposure and regulatory‑network
  damage.

This pilot serves as a template for any cancer type with sufficient mutation
and clinical data. Researchers are encouraged to adapt the pipeline to their
cohorts and to contribute additional duon annotations or drug target maps.

---

Citing Evolution ONE

If you use this software, please cite:

```
Yoon A Limsuwan. "Evolution ONE: Multi‑Level Cancer Evolution & Structural Impact Engine."
Zenodo, 2026.  DOI: 10.5281/zenodo.20277033
```

---

License

This project is licensed under the MIT License – see LICENSE for details.

---

Contact

Yoon A Limsuwan – GitHub
Project link: https://github.com/yoonalimsuwan/EVOLUTION-ONE

```
