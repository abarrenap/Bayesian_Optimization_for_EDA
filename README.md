# Bayesian Optimization for EDA

This project evaluates Bayesian Optimization (BO) as a local-search operator inside a continuous Estimation of Distribution Algorithm (EDA). It compares three methods under the same objective-evaluation budget:

- Plain Gaussian EDA
- Gaussian EDA with L-BFGS-B local search
- Gaussian EDA with Bayesian Optimization

The notebook contains the implementations, runs the experiments, saves the results, and generates the comparison tables and plots. The accompanying short report uses the Springer Lecture Notes in Computer Science format.

## Experimental setup

| Setting | Value |
| --- | --- |
| Benchmarks | Sphere, Rastrigin, Ackley, Rosenbrock |
| Dimensions | 5 |
| Population size | 30 |
| Selection ratio | 50% |
| Runs per benchmark | 10 |
| Maximum evaluations per run | 3,000 |
| Local-search subset | Best 25% |
| Local budget per generation | 40 evaluations |
| Local budget per solution | 10 evaluations |
| BO acquisition | Expected Improvement |
| BO trust region | 15% of the global range |

## Repository structure

```text
bo_eda_experiments.ipynb   Main implementation and experimental analysis
results/                   CSV results for the three methods
report/                    LNCS report source and generated PDF
report/chapters/           Individual report chapter files
proyect.txt                Course project requirements
```

## Setup

Create a Python environment and install the required packages:

```bash
pip install jupyter numpy pandas matplotlib seaborn scipy tqdm scikit-optimize pateda
```

Open the notebook:

```bash
jupyter lab bo_eda_experiments.ipynb
```

Run the notebook from top to bottom to reproduce the experiment CSV files and figures.

## Results

The notebook writes one file per method:

- `results/plain_eda_results.csv`
- `results/classical_ls_results.csv`
- `results/bo_results.csv`

Each row records the benchmark, run number, seed, number of evaluations, and best objective value found. Lower objective values are better.

## Report

The report is split into chapter files imported by `report/main.tex`. To compile it locally:

```bash
make -C report
```

The generated document is available at `report/main.pdf`.
