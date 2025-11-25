---
icon: lucide/rocket
title: Getting Started
---

# Getting Started with FEPA

**FEPA** (Free Energy Perturbation Analysis) is a Python package for analyzing molecular dynamics (MD) trajectories from FEP simulations, particularly **ABFEs**. FEPA allows you to identify and visualize conformational changes in the trajectories and set up simulations to correct free energy estimates.

This guide covers installation, basic usage, and key workflows.

---

## Prerequisites

To make full use of FEPA's functionality, several additional programs are required:

- [GROMACS](https://www.gromacs.org/) – for analyzing MD trajectories
- [MODELLER](https://salilab.org/modeller/) – from the Sali Lab, used for building missing residues or homology modeling.
- [PLUMED](https://www.plumed.org/) – for enhanced sampling or collective variable analysis.
- [WHAM](http://membrane.urmc.rochester.edu/?page_id=126) – from Grossfield Lab, for estimating free energies from umbrella sampling or REUS simulations.

Make sure these programs are installed and available in your system PATH so that FEPA can call them from Python.

## Installation

FEPA must be installed from GitHub:

```bash
git clone https://github.com/Nithishwer/FEPA.git
cd FEPA
pip install -e .
```
