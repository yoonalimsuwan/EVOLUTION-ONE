`
# Evolution ONE

**Multi‑Level Cancer Evolution & Structural Impact Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Evolution ONE is a standalone, vendor‑neutral platform that models cancer
progression through the lens of **Self‑Organised Criticality (SOC)**,
**duon biology**, and **physics‑based structural refinement**. It ingests
genomic mutation data (MAF, VCF) and outputs a complete oncological profile:

- the evolutionary regime of the tumour (stable, critical, or collapsing),
- future escape mutations likely to destabilise key proteins,
- CRISPR‑Cas editing designs (gRNA + repair templates),
- epigenetic targets for methylation editing,
- personalised drug and stabiliser recommendations,
- correlations between lifestyle factors and mutation load.

The engine can operate entirely on its own, or optionally leverage
**REAL FOLD ONE** and **REAL FOLD ONE HT** for atomic‑resolution ΔΔG
calculation and high‑throughput mutation scanning.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input File Formats](#input-file-formats)
- [Output Files](#output-files)
- [Integration with REAL FOLD ONE](#integration-with-real-fold-one)
- [How Evolution ONE Works](#how-evolution-one-works)
- [Pilot Study: Lung Adenocarcinoma (TCGA‑LUAD)](#pilot-study-lung-adenocarcinoma-tcga-luad)
- [CRISPR & Epigenetic Editing Support](#crispr--epigenetic-editing-support)
- [Hyperparameter Tuning & Checkpointing](#hyperparameter-tuning--checkpointing)
- [Architectural Philosophy](#architectural-philosophy)
- [Citing Evolution ONE](#citing-evolution-one)
- [License](#license)
- [Contact](#contact)

---

## Features

- **SOC‑Based Regime Classification** – Classifies each tumour sample as
  *stable* (stochastic), *critical* (multi‑level selection), or *collapse*
  (deterministic domino) using Shannon entropy and mutation load μ.
- **Predictive Evolution** – Projects future mutation burden with both
  SOC‑driven stochastic steps and **Itô Langevin dynamics**.
- **Duon‑Aware Analysis** – Load a list of duon codon positions and compute
  the fraction of mutations that disrupt duon function.
- **Future Escape Mutation Scanning** – Identifies single‑amino‑acid changes
  that would strongly destabilise the protein (ΔΔG > 1.5 kcal/mol), either
  via REAL FOLD ONE HT or an embedded energy evaluator.
- **Physics‑Based Structural Impact** – Computes the ΔΔG of observed
  mutations using full‑atom refinement (REAL FOLD ONE integration).
- **Drug & Stabiliser Recommendation** – Maps destabilised genes to
  existing targeted therapies and protein stabilisers.
- **CRISPR‑Cas Editing Design** – Generates gRNA sequences and HDR repair
  templates for the most destabilising predicted mutations.
- **Epigenetic Editing Targets** – Identifies duon positions suitable for
  dCas9‑DNMT/TET methylation editing.
- **Trainable SOC Thresholds** – Optimises the stable/collapse boundaries
  against clinical labels using differential evolution or Optuna.
- **Hyperparameter Tuning** – Optuna integration for Bayesian optimisation
  of model parameters (with fallback to scipy’s differential evolution).
- **Data Pipeline & Batching** – Batched processing of large MAF/VCF files
  keeps memory usage low (runs on 3 GB RAM).
- **Checkpoint / Resume** – Saves all intermediate results and trained
  parameters to disk; resume interrupted runs seamlessly.
- **RG Multiscale Smoothing** – Renormalisation‑group filtering removes
  noise from mutation load estimates.
- **Gene Network BV Consistency** – Applies the **Batalin–Vilkovisky**
  formalism to verify the topological consistency of gene regulatory
  networks.
- **Vendor‑Neutral** – Runs on CPU, GPU (CUDA, MPS), Huawei Ascend NPU,
  multi‑GPU, and supercomputers without code modification.

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

# Optional: install Optuna for advanced hyperparameter tuning
pip install optuna

# Optional: install REAL FOLD ONE and REAL FOLD ONE HT for enhanced features
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

# Train SOC thresholds from clinical labels (Optuna)
python evolution_one.py \
    --input tcga_colon.maf \
    --genes APC CTNNB1 SMAD4 \
    --clinical_labels_file labels.csv \
    --train_thresholds \
    --tune_method optuna \
    --output_dir ./colon_results

# Resume a previous run from checkpoint
python evolution_one.py \
    --input tcga_luad.maf \
    --genes EGFR KRAS \
    --resume ./evo_output/checkpoint.pkl
```

---

Input File Formats

Mutation file (MAF or VCF):

· Standard TCGA MAF or simple VCF with at least Hugo_Symbol and
  Tumor_Sample_Barcode (MAF) or GENE= in the INFO field (VCF).

Duon file (optional):

· A plain text file with one codon position (1‑based) per line.

Lifestyle file (optional):

· A CSV with columns: sample_id, and any number of environmental factors
  (e.g., smoking_pack_years, alcohol_consumption, diet_score).

Clinical labels file (for training):

· A CSV with columns: sample_id and label (0 = poor prognosis,
  1 = good prognosis).

Gene interactions (optional):

· Passed as --gene_interactions i j k l ... where each pair (i, j)
  represents a known regulatory interaction between gene i and gene j
  (indices refer to the order in --genes).

PDB directory (optional):

· A folder containing .pdb files for each gene. The engine automatically
  picks the first file whose name contains the gene symbol.

---

Output Files

File Content
summary.json Cancer risk, entropy, drug recommendations, BV check, tuning results
sample_states.csv Per‑sample mutation load (μ), evolutionary state (0/1/2)
phase_diagram.png (if --plot) Entropy vs. mutation load
crispr_designs.json gRNA sequences and repair templates for destabilising mutations
epigenetic_targets.json Duon positions and recommended methylation edits
checkpoint.pkl Full engine state for resumption

---

Integration with REAL FOLD ONE

When REAL FOLD ONE and REAL FOLD ONE HT are installed, Evolution ONE
gains:

· Accurate ΔΔG calculation – uses the full SOC‑controlled refinement
  engine instead of the embedded simplified energy function.
· Complete single‑mutation scanning – leverages the high‑throughput
  scanner to explore all possible amino‑acid changes, providing a
  comprehensive list of destabilising mutations.

The scripts automatically detect the presence of these libraries.

---

How Evolution ONE Works

Evolution ONE models cancer progression as a dynamical phase transition
driven by the accumulation of mutations. It quantifies the mutation load
μ (fraction of key genes affected) and uses principles from Self‑Organised
Criticality (SOC) to place each sample into one of three regimes:

Regime State μ range Biological interpretation
Stable (Stochastic) 0 μ < θ_stable Mutations are sparse and neutral. The system is far from a critical point.
Critical (Selection) 1 θ_stable ≤ μ ≤ θ_collapse The tumour is at the edge of collapse. Multi‑level selection (coding + regulatory, via duons) is active. SOC‑like avalanches generate heterogeneity.
Collapse (Deterministic) 2 μ > θ_collapse Mutational load overwhelms the system. Key proteins destabilise, regulatory networks break down, and the cancer follows a deterministic trajectory.

The thresholds θ_stable and θ_collapse can be trained from clinical data.
The engine also simulates future μ using SOC‑controlled stochastic steps
and Itô Langevin dynamics, predicting whether a tumour will cross into
the collapse regime.

Duon disruption (μ_duon) adds a second dimension, revealing mutations that
simultaneously alter protein sequence and gene regulation—a hallmark of
multi‑level selection.

---

Pilot Study: Lung Adenocarcinoma (TCGA‑LUAD)

A ready‑to‑run pilot for Lung Adenocarcinoma (LUAD) is included as a
template.  The data can be obtained from the public TCGA MC3 MAF file and
clinical supplement.  The pilot:

1. Filters the MAF for LUAD samples and genes of interest.
2. Computes μ, entropy, and duon disruption.
3. Trains SOC thresholds from survival labels (optional).
4. Predicts future escape mutations and designs CRISPR gRNA.
5. Outputs drug recommendations and phase diagrams.

See the full walk‑through in the repository’s documentation.

---

CRISPR & Epigenetic Editing Support

Evolution ONE extends beyond diagnosis into therapy design:

· CRISPRDesigner – for each destabilising mutation predicted by the HT
  scanner, it generates a gRNA sequence (20‑mer placeholder; full genome
  alignment can be added) and an HDR repair template encoding the
  desired codon change flanked by homology arms.
· EpigeneticDesigner – identifies duon positions that are frequently
  mutated and suggests whether dCas9‑TET (demethylation) or
  dCas9‑DNMT (hypermethylation) should be applied to restore normal
  regulation.

Both modules output JSON files that can be directly used by experimental
teams.

---

Hyperparameter Tuning & Checkpointing

· SOC threshold training – automatically adjusts θ_stable and
  θ_collapse to maximise correlation with clinical labels. Supports
  differential evolution (scipy) and Bayesian optimisation (Optuna).
· Optuna integration – when Optuna is installed, it provides
  state‑of‑the‑art hyperparameter tuning with pruning and visualisation.
· Checkpoint / Resume – all engine state (μ values, states, thresholds,
  future predictions) is saved to checkpoint.pkl. Interrupted runs can
  be resumed with --resume.

---

Architectural Philosophy

Evolution ONE shares the same vendor‑neutral, differentiable‑physics DNA as
REAL FOLD ONE:

· No CUDA C++ – pure PyTorch operations, enabling seamless execution on
  NVIDIA GPUs, Apple MPS, Huawei Ascend NPU, or CPU (3 GB RAM minimum).
· Differentiable from end to end – the SOC kernel, energy function, and
  Itô integrators are all differentiable, allowing future integration with
  deep learning surrogates.
· Physics first – rather than relying on statistical black‑boxes,
  Evolution ONE builds on first‑principle models (SOC, SSC, RG, BV) to
  interpret cancer as a dynamical system.

---

Citing Evolution ONE

If you use this software, please cite:

```
Yoon A Limsuwan. "Evolution ONE: Multi‑Level Cancer Evolution & Structural Impact Engine."
Zenodo, 2026. DOI: 10.5281/zenodo.XXXXXXX
```

---

License

This project is licensed under the MIT License – see LICENSE for
details.

---

Contact

Yoon A Limsuwan – GitHub
Project link: https://github.com/yoonalimsuwan/EVOLUTION-ONE

```
