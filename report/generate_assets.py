"""Rebuild report assets from the current experiment CSV files."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
METHODS = ['Plain EDA', 'EDA + L-BFGS-B', 'EDA + BO']
FILES = ['plain_eda_results.csv', 'classical_ls_results.csv', 'bo_results.csv']
BENCHMARKS = ['sphere', 'rastrigin', 'ackley', 'rosenbrock']
frames = []
for method, filename in zip(METHODS, FILES):
    frame = pd.read_csv(ROOT.parent / 'results' / filename)
    assert len(frame) == 40
    assert not frame.duplicated(['benchmark', 'seed']).any()
    assert (frame.evaluations == 600).all()
    assert np.isfinite(frame.best_value).all() and (frame.best_value > 0).all()
    for benchmark in BENCHMARKS:
        assert set(frame.loc[frame.benchmark == benchmark, 'seed']) == set(range(10))
    frames.append(frame.assign(method=method))
data = pd.concat(frames, ignore_index=True)
summary = data.groupby(['benchmark', 'method']).best_value.agg(
    median='median', q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75))

def fmt(x):
    if x < .01:
        mantissa, exponent = f'{x:.2e}'.split('e')
        return rf'{mantissa}\!\times\!10^{{{int(exponent)}}}'
    return f'{x:.3g}'

rows = [r'\begin{tabular}{lrrr}', r'\toprule',
        r'Benchmark & Plain EDA & EDA + L-BFGS-B & EDA + BO \\', r'\midrule']
for benchmark in BENCHMARKS:
    vals = [summary.loc[(benchmark, method)] for method in METHODS]
    best = min(v['median'] for v in vals)
    cells = [r'$\mathbf{' + fmt(v['median']) + '}$' if v['median'] == best
             else '$' + fmt(v['median']) + '$' for v in vals]
    rows.append(benchmark.capitalize() + ' & ' + ' & '.join(cells) + r' \\')
    rows.append(' & ' + ' & '.join('$[' + fmt(v['q1']) + ',' + fmt(v['q3']) + ']$'
                                  for v in vals) + r' \\[3pt]')
rows += [r'\bottomrule', r'\end{tabular}']
(ROOT / 'tables/main_results.tex').write_text('\n'.join(rows) + '\n')
plt.rcParams.update({'font.family':'serif', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'pdf.fonttype':42})
fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), constrained_layout=True)
colors = ['#3B7B4B', '#B95645', '#3479A8']
for ax, benchmark in zip(axes.flat, BENCHMARKS):
    groups = [data.loc[(data.benchmark == benchmark) & (data.method == method),
                       'best_value'].to_numpy() for method in METHODS]
    box = ax.boxplot(groups, widths=.5, patch_artist=True, showfliers=False,
                     medianprops={'color':'black','linewidth':1.3})
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(.4)
    for index, values in enumerate(groups, 1):
        ax.scatter(index + np.linspace(-.13,.13,len(values)), values,
                   s=13, c='#222222', alpha=.8, zorder=3)
    ax.set_xticks([1,2,3], ['Plain EDA','L-BFGS-B','BO'])
    ax.set_yscale('log'); ax.set_title(benchmark.capitalize())
    ax.set_ylabel('Final objective')
    ax.grid(axis='y', alpha=.2)
(ROOT / 'figures').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figures/final_distributions.pdf', bbox_inches='tight')
print(summary.to_string())
print('Validated 120 records and rebuilt table and figure.')
