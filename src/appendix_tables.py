"""Appendix table generation for the paper-2 (Decoding Alignment without Encoding
Alignment) submission.

Produces, from the already-computed sweep/GW caches in ``fig/paper/cache/``:
  1. The strategy-dependence table (SD across the 11 selection strategies of each
     decoding metric vs. GW similarity, by fraction of neurons kept) --
     ``tab:strategy_sd`` in the paper.
  2. A cell-type / selectivity-robustness table (OSI, lifetime sparseness, and
     amplitude for high- vs. low-PC1-contribution subpopulations) -- the table in
     Appendix "The Region-Sampling Effect Is Not a Coarse Excitatory/Inhibitory
     Split".

Both write a submission-ready ``\\begin{tabular}...\\end{tabular}`` snippet to
``fig/paper/tables/``, so the numbers can be dropped straight into the LaTeX
source instead of hand-transcribed.

Run as a script (``python -m src.appendix_tables`` from the repo root, or import
and call the two ``make_*`` functions from a notebook) once the sweep caches in
``fig/paper/cache/`` exist (see ``notebooks/13_paper_figures.ipynb`` cells 11-15
for how those caches are produced/loaded).
"""

import os
import pickle

import numpy as np

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASEDIR_CACHE = os.path.join(BASEDIR, 'fig', 'paper', 'cache')
BASEDIR_TABLES = os.path.join(BASEDIR, 'fig', 'paper', 'tables')

# Decoding metrics shown in the strategy-dependence table, in the order the
# paper's Table reports them (RSA, CKA, Procrustes R^2, tRSA, Speed, tProc).
_METRIC_KEYS = ['rdm', 'cka', 'r2', 'trsa', 'spc', 'tproc']
_METRIC_LABELS = {
    'rdm': 'RSA', 'cka': 'CKA', 'r2': 'Procrustes $R^2$',
    'trsa': 'tRSA', 'spc': 'Speed', 'tproc': 'tProc',
}

_SWEEP_CACHE_FILES = {
    'Retina': 'sweep_Retina.pkl',
    'V1': 'sweep_V1.pkl',
    'allen_VISp': 'sweep_allen_VISp.pkl',
    'VISp_NM': 'sweep_VISp_nat.pkl',
}


# ─────────────────────────── neuron-level tuning descriptors ────────────────

def tuning_descriptors(tensor4d, metrics):
    """Per-neuron OSI / entropy / lifetime sparseness / amplitude.

    Ported from ``experiments/p2_rebuttal/exp_b_celltypes.py::tuning_descriptors``
    in the parent analysis repository (the code that produced the rebuttal's
    OSI/sparseness/amplitude numbers). Self-contained (numpy only).

    Parameters
    ----------
    tensor4d : ndarray (N, S, D, T)
    metrics : dict with key 'osi' (ndarray, shape (N,)) -- as returned by
        ``load_dataset_for_sweep`` in notebooks/13_paper_figures.ipynb.
    """
    N, S, D, T = tensor4d.shape
    R = np.clip(np.nan_to_num(tensor4d.mean(axis=3).reshape(N, S * D), nan=0.0), 0, None)
    tot = R.sum(1, keepdims=True)
    P = np.divide(R, tot, out=np.full_like(R, 1.0 / (S * D)), where=tot > 0)
    entropy = -(P * np.log(P + 1e-12)).sum(1)
    n = S * D
    num, den = R.mean(1) ** 2, (R ** 2).mean(1)
    sparseness = (1 - np.divide(num, den, out=np.zeros_like(num), where=den > 0)) \
        / (1 - 1.0 / n)
    amplitude = np.sqrt(((R - R.mean(1, keepdims=True)) ** 2).mean(1))
    return dict(osi=np.asarray(metrics['osi'], float), entropy=entropy,
                sparseness=sparseness, amplitude=amplitude)


# ───────────────────────────── strategy-dependence table ────────────────────

def _load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_strategy_sweep_caches(basedir_cache=BASEDIR_CACHE):
    """Load the per-strategy decoding-metric sweeps and the GW-per-strategy sweep.

    Returns
    -------
    decoding : dict[dataset_key] -> (results, fracs)
        ``results[strategy][metric_key]`` is a list of (mean, std) tuples, one
        per entry in ``fracs`` -- exactly the structure notebook 13 caches to
        ``fig/paper/cache/sweep_{PREFIX}.pkl``.
    gw : dict with keys 'results', 'fractions', 'strategies', 'gw_max' -- the
        structure cached to ``fig/paper/cache/rebuttal_gw_strategy_sweep.pkl``.
    """
    decoding = {}
    for key, fname in _SWEEP_CACHE_FILES.items():
        path = os.path.join(basedir_cache, fname)
        if os.path.exists(path):
            decoding[key] = _load_pkl(path)
        else:
            print(f'[appendix_tables] missing {path}, skipping {key}')

    gw_path = os.path.join(basedir_cache, 'rebuttal_gw_strategy_sweep.pkl')
    gw = _load_pkl(gw_path) if os.path.exists(gw_path) else None
    if gw is None:
        print(f'[appendix_tables] missing {gw_path} -- GW column will be empty')
    return decoding, gw


def _nearest_idx(fracs, target):
    return int(np.argmin(np.abs(np.asarray(fracs) - target)))


def compute_strategy_sd_table(fractions=(0.01, 0.05, 0.10, 0.20),
                              basedir_cache=BASEDIR_CACHE):
    """SD across the 11 selection strategies, per metric, at each fraction.

    For each decoding metric and for GW, and at each requested fraction: takes
    the metric's mean value under each of the 11 strategies (nearest available
    fraction in that dataset's own sweep grid), computes the SD across
    strategies within each dataset, then averages that per-dataset SD across
    the datasets that have data for the given fraction. This mirrors the
    per-dataset-then-averaged aggregation used for the paper's Table (see
    ``paper_text/paper_2_iclr.md``, table label ``tab:strategy_sd``); re-verify
    against a from-scratch main-repo run before treating exact digits as final.

    Returns
    -------
    dict: metric_label -> {frac: sd_value}, plus 'ratio' -> {frac: mean(6)/GW}.
    """
    decoding, gw = load_strategy_sweep_caches(basedir_cache)
    if not decoding:
        raise RuntimeError('No sweep caches found -- run notebooks/13_paper_figures.ipynb '
                            'cells 11-15 first (or copy fig/paper/cache/sweep_*.pkl).')

    strategies = next(iter(decoding.values()))[0].keys()

    table = {m: {} for m in _METRIC_KEYS}
    table['GW'] = {}

    for frac in fractions:
        # decoding metrics
        for mkey in _METRIC_KEYS:
            per_dataset_sd = []
            for dkey, (results, fracs) in decoding.items():
                fi = _nearest_idx(fracs, frac)
                vals = [results[s][mkey][fi][0] for s in strategies if s in results]
                vals = [v for v in vals if np.isfinite(v)]
                if len(vals) >= 2:
                    per_dataset_sd.append(np.std(vals))
            table[mkey][frac] = float(np.mean(per_dataset_sd)) if per_dataset_sd else np.nan

        # GW
        if gw is not None:
            per_dataset_sd = []
            for dkey in gw['results']:
                fi = _nearest_idx(gw['fractions'][dkey], frac)
                vals = [gw['results'][dkey][s]['mean'][fi]
                        for s in gw['strategies'] if s in gw['results'][dkey]]
                vals = [v for v in vals if np.isfinite(v)]
                if len(vals) >= 2:
                    per_dataset_sd.append(np.std(vals))
            table['GW'][frac] = float(np.mean(per_dataset_sd)) if per_dataset_sd else np.nan

    table['ratio'] = {
        frac: (np.mean([table[m][frac] for m in _METRIC_KEYS]) / table['GW'][frac])
              if table['GW'].get(frac) else np.nan
        for frac in fractions
    }
    return table


def strategy_sd_table_to_latex(table, fractions=(0.01, 0.05, 0.10, 0.20)):
    header = ' & ' + ' & '.join(f'{int(f * 100)}\\%' for f in fractions) + r' \\'
    lines = [
        r'\begin{table}[h]', r'\centering', r'\small',
        r'\caption{Standard deviation across the 11 selection strategies, by fraction of neurons kept.}',
        r'\label{tab:strategy_sd}',
        r'\begin{tabular}{l' + 'r' * len(fractions) + '}',
        r'\toprule',
        header,
        r'\midrule',
    ]
    for mkey in _METRIC_KEYS:
        vals = ' & '.join(f'{table[mkey][f]:.3f}' for f in fractions)
        lines.append(f'{_METRIC_LABELS[mkey]}              & {vals} \\\\')
    gw_vals = ' & '.join(f'\\textbf{{{table["GW"][f]:.3f}}}' for f in fractions)
    lines.append(r'\textbf{GW (ours)} & ' + gw_vals + r' \\')
    lines.append(r'\midrule')
    ratio_vals = ' & '.join(f'{table["ratio"][f]:.1f}$\\times$' for f in fractions)
    lines.append(r'ratio (mean of 6 / GW) & ' + ratio_vals + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ─────────────────────── cell-type / selectivity robustness table ───────────

def compute_celltype_robustness_table(datasets):
    """OSI / lifetime-sparseness / amplitude for high- vs. low-PC1 subpopulations.

    Parameters
    ----------
    datasets : dict[label] -> dict with keys 'tensor4d', 'metrics' (as returned
        by ``load_dataset_for_sweep`` in notebooks/13_paper_figures.ipynb) and
        'frac' (fraction of neurons in the high/low-PC1 subpopulation, default
        0.05 to match the paper's Figure 2 5% condition).

    Returns
    -------
    dict[label] -> {'osi_hi', 'osi_lo', 'sparseness_hi', 'sparseness_lo',
                    'amplitude_hi', 'amplitude_lo'}
    """
    from src.subpop_utils import select_top_k_by_metric

    out = {}
    for label, ds in datasets.items():
        tensor4d, metrics = ds['tensor4d'], ds['metrics']
        frac = ds.get('frac', 0.05)
        N = tensor4d.shape[0]
        k = max(1, int(round(frac * N)))
        desc = tuning_descriptors(tensor4d, metrics)
        idx_hi = select_top_k_by_metric(metrics, 'pc_contrib', k=k, high=True)
        idx_lo = select_top_k_by_metric(metrics, 'pc_contrib', k=k, high=False)
        row = {}
        for name, idx in (('hi', idx_hi), ('lo', idx_lo)):
            for dkey in ('osi', 'sparseness', 'amplitude'):
                row[f'{dkey}_{name}'] = float(np.nanmean(desc[dkey][idx]))
        out[label] = row
    return out


def celltype_table_to_latex(table):
    lines = [
        r'\begin{table}[h]', r'\centering', r'\small',
        r'\caption{5\% subpopulations selected by PC1 contribution: tuning statistics (mean over selected neurons).}',
        r'\label{tab:celltype_robustness}',
        r'\begin{tabular}{lrrr}', r'\toprule',
        r'5\% subpop. & OSI hi/lo & Lifetime sparseness hi/lo & Amplitude hi/lo \\',
        r'\midrule',
    ]
    def _fmt(v):
        return '--' if np.isnan(v) else f'{v:.2f}'

    for label, row in table.items():
        lines.append(
            f'{label} & {_fmt(row["osi_hi"])} / {_fmt(row["osi_lo"])} '
            f'& {_fmt(row["sparseness_hi"])} / {_fmt(row["sparseness_lo"])} '
            f'& {_fmt(row["amplitude_hi"])} / {_fmt(row["amplitude_lo"])} \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ───────────────────── Procrustes forms (R^2 / Frobenius / geodesic) ─────────

def compute_procrustes_forms_table(datasets, frac=0.05,
                                   fps_seeds=(42, 43, 44, 45, 46)):
    """R^2 / Frobenius distance / geodesic angle for the region-sampling
    (hi-/lo-PC1) and FPS (continuous/clustered) contrasts, each dataset's
    subpopulation coordinates aligned against its own full-population coords.

    Ported from ``experiments/p2_rebuttal/exp_defg.py::exp_d`` in the parent
    analysis repository; uses the ``procrustes_forms`` already defined in
    ``src/subpop_utils.py``.

    Parameters
    ----------
    datasets : dict[label] -> dict with 'tensor4d', 'metrics', 'embedding'
        (as returned by ``load_dataset_for_sweep`` plus the encoding-manifold
        embedding already loaded for Figures 2/3 in notebook 13).
    frac : fraction of neurons in each subpopulation (default 0.05, matching
        Figure 2's 5% condition and Figure 3's FPS fraction).

    Returns
    -------
    dict[label] -> dict[condition] -> {'r2': [mean, std], 'd_frob': [mean, std],
                                        'theta_deg': [mean, std]}
        conditions: 'hiPC', 'loPC' (single draw, no seed averaging, matching
        Figure 2), 'continuous', 'clustered' (averaged over fps_seeds, matching
        Figure 3).
    """
    from src.subpop_utils import (compute_decoding_manifold, procrustes_forms,
                                   select_top_k_by_metric)
    out = {}
    for label, ds in datasets.items():
        tensor4d, metrics, embedding = ds['tensor4d'], ds['metrics'], ds['embedding']
        fps_select_fn = ds['fps_select']
        N = tensor4d.shape[0]
        k = max(1, int(round(frac * N)))
        coords_ref, _ = compute_decoding_manifold(tensor4d, n_components=15)

        def _forms_for(idx):
            cs, _ = compute_decoding_manifold(tensor4d[idx], n_components=15)
            if cs.shape[1] < coords_ref.shape[1]:
                cs = np.hstack([cs, np.zeros((len(cs), coords_ref.shape[1] - cs.shape[1]))])
            return procrustes_forms(coords_ref, cs)

        idx_hi = select_top_k_by_metric(metrics, 'pc_contrib', k=k, high=True)
        idx_lo = select_top_k_by_metric(metrics, 'pc_contrib', k=k, high=False)
        row = {'hiPC': {k2: [v, 0.0] for k2, v in _forms_for(idx_hi).items()},
               'loPC': {k2: [v, 0.0] for k2, v in _forms_for(idx_lo).items()}}

        for cond_name, fps_kwargs in (('continuous', dict(n_seeds=50, m_neighbors=1)),
                                       ('clustered', dict(n_seeds=10, m_neighbors=9))):
            per_seed = []
            for seed in fps_seeds:
                idx, _ = fps_select_fn(embedding, rng_seed=seed, **fps_kwargs)
                per_seed.append(_forms_for(idx))
            row[cond_name] = {
                k2: [float(np.mean([d[k2] for d in per_seed])),
                     float(np.std([d[k2] for d in per_seed]))]
                for k2 in ('r2', 'd_frob', 'theta_deg')
            }
        out[label] = row
    return out


def procrustes_forms_table_to_latex(table):
    lines = [
        r'\begin{table}[h]', r'\centering', r'\small',
        r'\caption{Procrustes alignment in three parameterisations ($R^2$ / $d_F$ / $\theta$) '
        r'for the region-sampling (hi-/lo-PC1) and FPS (continuous/clustered) contrasts.}',
        r'\label{tab:procrustes_forms}',
        r'\begin{tabular}{lccc}', r'\toprule',
        r' & hi-PC / lo-PC & continuous / clustered & '
        r'$\Delta$(cont$-$clust): $R^2$ / $d_F$ / $\theta$ \\',
        r'\midrule',
    ]
    def _fmt3(d):
        return (f'{d["r2"][0]:.3f}/{d["d_frob"][0]:.3f}/{d["theta_deg"][0]:.1f}'
                r'\textdegree')
    for label, row in table.items():
        d_r2 = row['continuous']['r2'][0] - row['clustered']['r2'][0]
        d_df = row['continuous']['d_frob'][0] - row['clustered']['d_frob'][0]
        d_th = row['continuous']['theta_deg'][0] - row['clustered']['theta_deg'][0]
        lines.append(
            f'{label} & {_fmt3(row["hiPC"])}\\ / {_fmt3(row["loPC"])} '
            f'& {_fmt3(row["continuous"])}\\ / {_fmt3(row["clustered"])} '
            f'& {d_r2:+.3f} / {d_df:+.3f} / {d_th:+.1f}\\textdegree \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ───────────────────── single-functional-group GW table ──────────────────────

def compute_functional_group_table(tensor4d, embedding, stim_labels,
                                    compute_all_metrics_fn, gw_dist_fn, gw_max_fn,
                                    n_groups=3, min_cluster_size=25, min_samples=4,
                                    n_comp=3):
    """Decoding metrics + GW restricted to each of the top-``n_groups`` largest
    HDBSCAN clusters of the encoding manifold (a single "functional group" /
    putative cell type), vs. the full population.

    Ported from ``experiments/p2_rebuttal/exp_b_celltypes.py`` (its B4 analysis)
    in the parent analysis repository. Uses the same clustering hyperparameters
    already used for the pipeline-robustness check in Appendix
    "Encoding Pipeline Robustness" (``min_cluster_size=25, min_samples=4``).

    Parameters
    ----------
    tensor4d : ndarray (N, S, D, T) -- full population response tensor.
    embedding : ndarray (N, >=n_comp) -- encoding-manifold embedding (diffusion
        coordinates) used to cluster neurons into functional groups.
    stim_labels : ndarray (S*D,) -- stimulus condition label per trial, as used
        by ``compute_all_metrics`` elsewhere in notebook 13.
    compute_all_metrics_fn : the notebook-local ``compute_all_metrics`` function
        (RSA/CKA/Procrustes R^2/... between a subpopulation and the reference).
    gw_dist_fn, gw_max_fn : the notebook-local ``_gw_dist``/``_gw_max_dist``
        helpers (GW similarity between subpopulation and full-population
        embeddings, normalized against the full-vs-collapsed baseline).

    Returns
    -------
    list of dict: {'group', 'k', 'frac', 'rsa', 'cka', 'r2', 'gw'}, largest
    cluster first.
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        from sklearn.cluster import HDBSCAN

    N = tensor4d.shape[0]
    labels = HDBSCAN(min_cluster_size=min_cluster_size,
                     min_samples=min_samples).fit_predict(embedding[:, :n_comp].astype(float))
    sizes = [(int(np.sum(labels == c)), int(c)) for c in sorted(set(labels[labels >= 0]))]
    gw_max = gw_max_fn(embedding)

    rows = []
    group_names = 'ABCDEFGHIJ'
    for gi, (size, c) in enumerate(sorted(sizes, reverse=True)[:n_groups]):
        idx = np.flatnonzero(labels == c)
        m = compute_all_metrics_fn(tensor4d, tensor4d[idx], stim_labels)
        gw = 1.0 - gw_dist_fn(embedding, idx) / gw_max if gw_max else float('nan')
        rows.append(dict(group=group_names[gi], k=size, frac=size / N,
                         rsa=float(m.get('rdm', m.get('rsa', np.nan))),
                         cka=float(m.get('cka', np.nan)),
                         r2=float(m.get('r2', np.nan)),
                         gw=float(gw)))
    return rows


def functional_group_table_to_latex(rows, dataset_label='Retina'):
    lines = [
        r'\begin{table}[h]', r'\centering', r'\small',
        r'\caption{' + dataset_label + r', samples confined to a single functional '
        r'group of the encoding manifold: decoding metrics still vary substantially '
        r'while GW stays comparatively low.}',
        r'\label{tab:functional_group}',
        r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Group & $k$ (\% pop.) & RSA & CKA & Proc. $R^2$ & GW \\',
        r'\midrule',
    ]
    for r in rows:
        lines.append(
            f'{r["group"]} & {r["k"]} ({100*r["frac"]:.0f}\\%) & {r["rsa"]:.3f} & '
            f'{r["cka"]:.3f} & {r["r2"]:.3f} & {r["gw"]:.3f} \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ────────────────────────────────── I/O helper ───────────────────────────────

def write_table(latex_str, filename, basedir_tables=BASEDIR_TABLES):
    os.makedirs(basedir_tables, exist_ok=True)
    path = os.path.join(basedir_tables, filename)
    with open(path, 'w') as f:
        f.write(latex_str + '\n')
    print(f'Wrote {path}')
    return path


if __name__ == '__main__':
    table = compute_strategy_sd_table()
    write_table(strategy_sd_table_to_latex(table), 'tab_strategy_sd.tex')
    print('Strategy-SD table:')
    for k, v in table.items():
        print(f'  {k}: {v}')
    print('\nNOTE: the cell-type robustness table requires `datasets` (tensor4d + metrics '
          'per dataset, e.g. from notebook 13\'s load_dataset_for_sweep calls) and is not '
          'run from this __main__ block -- call compute_celltype_robustness_table(datasets) '
          'from the notebook once those are loaded.')
