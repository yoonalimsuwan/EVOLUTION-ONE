``
# Evolution ONE

**Multi‑Level Cancer Evolution & Structural Impact Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20753447-blue)](https://doi.org/10.5281/zenodo.20753447)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20451155-blue)](https://doi.org/10.5281/zenodo.20451155)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20453128-blue)](https://doi.org/10.5281/zenodo.20453128)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20633681-blue)](https://doi.org/10.5281/zenodo.20633681)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20623622-blue)](https://doi.org/10.5281/zenodo.20623622)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20755856-blue)](https://doi.org/10.5281/zenodo.20755856)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20755892-blue)](https://doi.org/10.5281/zenodo.20755892)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20730429-blue)](https://doi.org/10.5281/zenodo.20730429)


Evolution ONE is a standalone, vendor‑neutral platform that models cancer
progression through the lens of **Self‑Organised Criticality (SOC)**,
**duon biology**, and **physics‑based structural refinement**. It ingests
genomic mutation data (MAF, VCF) and outputs a complete oncological profile:

- the evolutionary regime of the tumour (stable, critical, or collapsing),
- future escape mutations likely to destabilise key proteins,
- CRISPR‑Cas editing designs with scored gRNA and off‑target search,
- epigenetic targets for methylation editing,
- personalised drug and stabiliser recommendations,
- correlations between lifestyle factors and mutation load.

The engine can operate entirely on its own, or optionally leverage
**REAL FOLD ONE** and **REAL FOLD ONE HT** for atomic‑resolution ΔΔG
calculation and high‑throughput mutation scanning.

---

## Features

- **SOC‑Based Regime Classification** – Classifies each tumour sample as
  *stable* (stochastic), *critical* (multi‑level selection), or *collapse*
  (deterministic domino) using Shannon entropy and mutation load μ.
- **Predictive Evolution** – Projects future mutation burden with both
  SOC‑driven stochastic steps and **Itô Langevin dynamics**.
- **Duon‑Aware Analysis** – Load a BED file of duon regions (coding + regulatory
  overlap) and compute the fraction of mutations that fall within duons,
  using fast genomic interval overlap with PyRanges.
- **Future Escape Mutation Scanning** – Identifies single‑amino‑acid changes
  that would strongly destabilise the protein (ΔΔG > 1.5 kcal/mol), either
  via REAL FOLD ONE HT or an embedded energy evaluator.
- **Physics‑Based Structural Impact** – Computes the ΔΔG of observed
  mutations using full‑atom refinement (REAL FOLD ONE integration).
- **Drug & Stabiliser Recommendation** – Maps destabilised genes to
  existing targeted therapies and protein stabilisers.
- **CRISPR‑Cas Editing Design** – Generates fully scored gRNA sequences
  (Doench 2016‑style model) and HDR repair templates. Off‑target search
  is performed via **bowtie** (if installed) or brute‑force approximate
  matching against a user‑supplied reference genome FASTA.
- **Epigenetic Editing Targets** – Identifies duon positions suitable for
  dCas9‑DNMT/TET methylation editing, optionally integrating methylation
  BED data.
- **Trainable SOC Thresholds** – Optimises the stable/collapse boundaries
  against clinical labels using differential evolution (scipy) or Optuna
  (Bayesian).
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
git clone https://github.com/yoonalimsuwan/EVOLUTION-ONE.git
cd evolution-one

# Create and activate a conda environment (optional)
conda create -n evolution python=3.10 -y
conda activate evolution

# Install core dependencies
pip install torch numpy pandas scipy matplotlib

# For genomic interval operations (duon BED overlap)
pip install pyranges

# For sequence handling (gRNA design, reverse complement)
pip install biopython

# Optional: install biotite for PDB reading (if using structural features)
pip install biotite

# Optional: install Optuna for advanced hyperparameter tuning
pip install optuna

# Optional: install REAL FOLD ONE and REAL FOLD ONE HT for enhanced features
pip install -e .   # if they are in the same repository

# Optional: install bowtie for CRISPR off‑target search
# (requires bowtie executable in PATH; see http://bowtie-bio.sourceforge.net)
```

---

Quick Start

```bash
# Basic analysis from a TCGA MAF file with duon BED and reference genome
python evolution_one_v4.py \
    --input tcga_lung.maf \
    --genes EGFR KRAS TP53 \
    --duon_bed duons.bed \
    --ref_genome hg38.fa \
    --lifestyle_file patient_lifestyle.csv \
    --plot

# Use VCF input and skip future mutation scanning
python evolution_one_v4.py \
    --input patient.vcf --format vcf \
    --genes BRAF PIK3CA \
    --no_future

# Train SOC thresholds from clinical labels (Optuna) and generate CRISPR designs
python evolution_one_v4.py \
    --input tcga_colon.maf \
    --genes APC CTNNB1 SMAD4 \
    --clinical_labels_file labels.csv \
    --train_thresholds \
    --tune_method optuna \
    --ref_genome hg38.fa \
    --output_dir ./colon_results

# Resume a previous run from checkpoint
python evolution_one_v4.py \
    --input tcga_luad.maf \
    --genes EGFR KRAS \
    --resume ./evo_output/checkpoint.pkl
```

---

Input File Formats

· Mutation file (MAF or VCF)
    Standard TCGA MAF or simple VCF with at least Hugo_Symbol and Tumor_Sample_Barcode (MAF) or GENE= in the INFO field (VCF).
· Duon BED file (optional)
    A BED file of duon regions (coding + regulatory overlap). Must contain at least chromosome, start, and end columns. Used for fast PyRanges overlap with mutations.
· Reference genome FASTA (optional, required for gRNA design and off‑target search)
    An indexed (.fa or .fasta) human genome assembly (e.g. hg38). Enables extraction of genomic context for gRNA and repair template design, and off‑target search via bowtie or built‑in approximate matching.
· Lifestyle file (optional)
    A CSV with columns: sample_id, and any number of environmental factors (e.g., smoking_pack_years, alcohol_consumption, diet_score).
· Clinical labels file (for training)
    A CSV with columns: sample_id and label (0 = poor prognosis, 1 = good prognosis).
· Gene interactions (optional)
    Passed as --gene_interactions i j k l ... where each pair (i, j) represents a known regulatory interaction between gene i and gene j (indices refer to the order in --genes).
· PDB directory (optional)
    A folder containing .pdb files for each gene. The engine automatically picks the first file whose name contains the gene symbol.

---

Output Files

File Content
summary.json Cancer risk, entropy, drug recommendations, BV check, tuning results
sample_states.csv Per‑sample mutation load (μ), evolutionary state (0/1/2)
phase_diagram.png (if --plot) Entropy vs. mutation load
crispr_designs.json Scored gRNA sequences, off‑target counts, and HDR repair templates
epigenetic_targets.json Duon positions and recommended methylation edits
checkpoint.pkl Full engine state for resumption

---

Integration with REAL FOLD ONE

When REAL FOLD ONE and REAL FOLD ONE HT are installed, Evolution ONE gains:

· Accurate ΔΔG calculation – uses the full SOC‑controlled refinement engine instead of the embedded simplified energy function.
· Complete single‑mutation scanning – leverages the high‑throughput scanner to explore all possible amino‑acid changes, providing a comprehensive list of destabilising mutations.

The scripts automatically detect the presence of these libraries.

---

How Evolution ONE Works

Evolution ONE models cancer progression as a dynamical phase transition driven by the accumulation of mutations. It quantifies the mutation load μ (fraction of key genes affected) and uses principles from Self‑Organised Criticality (SOC) to place each sample into one of three regimes:

Regime State μ range Biological interpretation
Stable (Stochastic) 0 μ < θ_stable Mutations are sparse and neutral. The system is far from a critical point.
Critical (Selection) 1 θ_stable ≤ μ ≤ θ_collapse The tumour is at the edge of collapse. Multi‑level selection (coding + regulatory, via duons) is active. SOC‑like avalanches generate heterogeneity.
Collapse (Deterministic) 2 μ > θ_collapse Mutational load overwhelms the system. Key proteins destabilise, regulatory networks break down, and the cancer follows a deterministic trajectory.

The thresholds θ_stable and θ_collapse can be trained from clinical data. The engine also simulates future μ using SOC‑controlled stochastic steps and Itô Langevin dynamics, predicting whether a tumour will cross into the collapse regime.

Duon disruption (μ_duon) adds a second dimension, revealing mutations that simultaneously alter protein sequence and gene regulation—a hallmark of multi‑level selection.

---

CRISPR & Epigenetic Editing Support

Evolution ONE extends beyond diagnosis into therapy design:

· CRISPRDesigner
    For each destabilising mutation predicted by the HT scanner, it:
  1. Extracts the genomic context around the target codon from the provided reference genome.
  2. Identifies all possible 20‑mer gRNAs with an NGG PAM.
  3. Scores each guide using a rule‑based model (GC content, poly‑T avoidance, position 20 G‑preference, and self‑complementarity) inspired by Doench et al. (2016).
  4. Searches for off‑target sites (up to 3 mismatches) using bowtie (if installed) or a built‑in approximate string matching against the reference genome.
  5. Selects the best guide (high on‑target score, few off‑targets).
  6. Constructs an HDR repair template with homology arms (default 30 bp) and the desired codon change.
· EpigeneticDesigner
    Identifies duon regions that are frequently mutated. When a methylation BED is available, it assesses the current methylation level and recommends:
  · dCas9‑TET (demethylation) for hypermethylated, mutation‑prone duons,
  · dCas9‑DNMT (hypermethylation) for hypomethylated duons,
  · further investigation when methylation data are unavailable.

Both modules output JSON files that can be directly used by experimental teams.

---

Pilot Study: Lung Adenocarcinoma (TCGA‑LUAD)

A ready‑to‑run pilot for Lung Adenocarcinoma (LUAD) is included as a template. The data can be obtained from the public TCGA MC3 MAF file and clinical supplement. The pilot:

1. Filters the MAF for LUAD samples and genes of interest.
2. Computes μ, entropy, and duon disruption (with a BED of known duons).
3. Trains SOC thresholds from survival labels (optional).
4. Predicts future escape mutations and designs CRISPR gRNA (requires a reference genome).
5. Outputs drug recommendations and phase diagrams.

See the full walk‑through in the repository’s documentation.

---

Hyperparameter Tuning & Checkpointing

· SOC threshold training – automatically adjusts θ_stable and θ_collapse to maximise correlation with clinical labels. Supports differential evolution (scipy) and Bayesian optimisation (Optuna).
· Optuna integration – when Optuna is installed, it provides state‑of‑the‑art hyperparameter tuning with pruning and visualisation.
· Checkpoint / Resume – all engine state (μ values, states, thresholds, future predictions) is saved to checkpoint.pkl. Interrupted runs can be resumed with --resume.

---

Evolution ONE as a Data Source for AI

Evolution ONE is fully differentiable and AI‑ready. Every component—the CSOC kernel, Semantic‑State Contraction, Renormalisation Group, and Itô calculus—runs inside PyTorch’s autograd engine, meaning it can backpropagate gradients directly to any neural network.

Beyond differentiability, Evolution ONE produces rich, structured outputs that serve as high‑quality training data for:

· Graph Neural Networks – for gene regulatory network analysis and duon‑interaction prediction.
· Reinforcement Learning – to design optimal CRISPR strategies that minimise off‑target effects.
· Bayesian Optimization – for personalised therapy selection and dosage scheduling.
· Transformer models – for survival prediction and patient stratification from evolutionary state profiles.
· Diffusion models – for generating synthetic cancer evolution trajectories that respect SOC dynamics.

When coupled with REAL FOLD ONE, Evolution ONE also helps create physics‑based training sets for AI surrogate models that can predict ΔΔG or protein stability orders of magnitude faster than full simulation—a cornerstone of the O(1) refinement pipeline described in the REAL FOLD ONE documentation.

Here is a highly professional, academic-grade GitHub README.md tailored specifically for applying the **EVOLUTION ONE** engine to advanced epidemiological modeling and viral evolutionary dynamics.

# EVOLUTION ONE: High-Performance Epidemiological Forecasting & Viral Evolution Engine
## Overview
**EVOLUTION ONE** is a high-performance, fully differentiable translational medicine platform modified for **predictive epidemiology and viral evolutionary dynamics**. By pivoting its core mathematical architecture from oncological mutation modeling to viral pathogen tracking, the engine bypasses traditional probabilistic frameworks in favor of a **deterministic, sub-quantum variable approach**.
Powered by **Structural Calculus** and multi-scale physics engines, EVOLUTION ONE models the transition of infectious diseases from localized outbreaks to global pandemics as structural phase transitions, while simultaneously forecasting viral antigenic drift and immune escape vectors before they manifest in the wild.
## Core Mathematical Framework & Translation
The engine maps micro-level genetic mutations to macro-level epidemiological outcomes by translating its core mathematical modules:
### 1. Self-Organized Criticality (SOC) & Phase Transitions
Instead of tracking cell mutation loads (\mu), the **SOC Engine** evaluates transmission dynamics and viral shedding rates. It identifies the exact critical thresholds where localized endemic pathogens undergo a phase transition into uncontrollable global pandemics (Pandemic\ Phase\ Transition).
 * **Stable Regime:** Controlled transmission with localized equilibrium.
 * **Critical Phase:** The tipping point where public health mitigation efficacy begins to collapse.
 * **Collapse (Pandemic State):** Cascading transmission dynamics across highly interconnected networks.
### 2. Deterministic Transmission via Itô Calculus
Traditional epidemiological models (e.g., SIR, SEIR) rely on stochastic approximations that fail under high-noise environmental conditions. EVOLUTION ONE utilizes **Itô Calculus and Langevin Dynamics** to solve transmission trajectories deterministically:
By integrating a differentiable noise-filtering layer, the engine transforms stochastic societal noise into structured, predictable transmission variables.
### 3. Multi-Scale Scaling via Renormalization Group (RG)
The **DiffRGRefiner** module applies the Renormalization Group framework to eliminate localized spatial noise. This allows seamless mathematical scaling from **individual viral loads (micro-scale)** to **community clusters (meso-scale)**, up to **continental transmission webs (macro-scale)** without losing computational precision.
## Key Epidemiological Modules
```
EVOLUTION_ONE (Epidemiology Architecture)
├── Core Engines (Structural & Regime Calculus)
├── ViralEvolutionPredictor (Antigenic Drift & Structural Bio)
├── PandemicPhaseEngine (SOC Criticality Solver)
└── StochasticTransmissionSolver (Itô Integration & RG Scaling)

```
### ViralEvolutionPredictor
 * **Antigenic Drift Forecasting:** Utilizes the structural biophysics engine to simulate amino acid substitutions within viral attachment proteins (e.g., Influenza Hemagglutinin, SARS-CoV-2 Spike Protein).
 * **Affinity Shift Mapping:** Computes binding free energy changes (\Delta\Delta G) between mutated viral proteins and human cellular receptors (e.g., ACE2) to identify high-affinity emerging variants.
 * **Immune Escape Prediction:** Evaluates whether predicted structural variations will render existing neutralizing antibodies or vaccine templates obsolete.
### PandemicPhaseEngine
 * **Infrastructure Strain Modeling:** Simulates real-time hospital, ICU, and ventilator capacity over varied demographic landscapes.
 * **Intervention Simulation:** Evaluates the deterministic impact of non-pharmaceutical interventions (NPIs) like border controls, lock-downs, or targeted ring-vaccination protocols.
### StochasticTransmissionSolver
 * **PyTorch Autograd Backend:** The entire transmission framework is fully differentiable, allowing it to generate high-fidelity gradient streams to train secondary neural network architectures or foundation models.
 * **High-Performance Batch Loading:** Features optimized data pipelines to ingest massive, real-time global sequencing datasets (GISAID/NCBI formats) or regional VCF/MAF epidemiological line-lists under minimal memory overhead.
## Getting Started
### Prerequisites
 * Python 3.10+
 * PyTorch 2.0+ (with CUDA support)
 * Biopython
 * NumPy / SciPy / Pandas


```
## Academic & Open Science Foundation
EVOLUTION ONE is built upon an original interdisciplinary framework utilizing proprietary systems of **Structural Calculus** and **Regime Calculus**. This epidemiological translation acts as a deterministic verification platform for modeling complex, non-linear biological systems under real-world clinical and structural stress tests.


---
```

Citing Evolution ONE

If you use this software, please cite:

`
Yoon A Limsuwan. "Evolution ONE: Multi‑Level Cancer Evolution & Structural Impact Engine And Epidemiology."
Zenodo, 2026.
https://doi.org/10.5281/zenodo.20753447
```

---

License

This project is licensed under the MIT License – see LICENSE for details.

---

Contact

Yoon A Limsuwan – GitHub
Project link: https://github.com/yoonalimsuwan/EVOLUTION-ONE

```
