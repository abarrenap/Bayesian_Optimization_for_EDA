# Bayesian Optimization for EDA

Compare three continuous Estimation of Distribution Algorithm (EDA) methods:

- Plain Gaussian EDA
- EDA with L-BFGS-B local search
- EDA with Bayesian Optimization (BO) local search

Experiments use Sphere, Rastrigin, Ackley, and Rosenbrock in 5 dimensions, with 10 runs per benchmark and a budget of 600 objective evaluations per run, including local search. Lower objective values are better.

## Run the experiments

Install the Python dependencies:

```bash
pip install jupyterlab ipywidgets numpy pandas matplotlib seaborn scipy tqdm scikit-optimize pateda
```

From the project folder, open the notebook and run its cells in order:

```bash
jupyter lab bo_eda_experiments.ipynb
```

The notebook saves final results, evaluation histories, and convergence summaries to `results/`, and displays comparison tables and plots. Running it again overwrites the generated CSV files.

## Files

- `bo_eda_experiments.ipynb` — implementation, experiments, and analysis.
- `results/` — experiment results and convergence data in CSV format.
- `report/` — LaTeX report, figures, tables, and compiled PDF.

## Build the report

With a LaTeX installation and `latexmk` available:

```bash
make -C report
```

The report is saved as `report/main.pdf`.
