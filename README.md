# QC-FCIQMC
Quantum Computing for Quantum Monte Carlo methods

# FCIQMC Simulator for 1D Hubbard Model

This project implements a basic Full Configuration Interaction Quantum Monte Carlo (FCIQMC) algorithm for simulating the 1D Fermi-Hubbard model. It's a Monte Carlo method designed to stochastically sample the ground state wavefunction of quantum many-body systems.

The implementation is based on the principles outlined in the paper:
**"Fermion Monte Carlo without fixed nodes: A game of life, death, and annihilation in Slater determinant space"** by George H. Booth, Alex J. W. Thom, and Ali Alavi (J. Chem. Phys. 131, 054106 (2009)).


## Getting Started
### Prerequisites

You need Python 3 installed. The project also relies on the following Python libraries:

* `numpy`
* `matplotlib`
* `scipy`
* `tqdm` (for progress bars)

You can install these using pip:
```bash
pip install numpy matplotlib scipy tqdm
