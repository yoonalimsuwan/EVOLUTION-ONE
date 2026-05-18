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

The engine can work entirely on its own, or optionally leverage **REAL FOLD ONE**
and **REAL FOLD ONE HT** for full‑atom structural impact calculations and
high‑throughput mutation scanning.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input File Formats](#input-file-formats)
- [Output Files](#output-files)
- [Integration with REAL FOLD ONE](#integration-with-real-fold-one)
- [Architectural Philosophy](#architectural-philosophy)
- [Citing Evolution ONE](#citing-evolution-one)
- [License](#license)

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
