``
# Evolution ONE

**Multi‑Level Cancer Evolution & Structural Impact Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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
git clone https://github.com/your-username/real-fold-one.git
cd real-fold-one

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

---

Citing Evolution ONE

If you use this software, please cite:

```
Yoon A Limsuwan. "Evolution ONE: Multi‑Level Cancer Evolution & Structural Impact Engine."
Zenodo, 2026. DOI: 10.5281/zenodo.XXXXXXX
```

---

License

This project is licensed under the MIT License – see LICENSE for details.

---

Contact

Yoon A Limsuwan – GitHub
Project link: https://github.com/your-username/real-fold-one

```
