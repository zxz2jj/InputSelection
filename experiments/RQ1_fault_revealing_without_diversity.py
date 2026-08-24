import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent
_SEL_DIR = _REPO_ROOT / 'selection_method'
for _p in (_REPO_ROOT, _SEL_DIR, _EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')
logging.getLogger().setLevel(logging.WARNING)

from baseline_datis import DATIS_K, DATIS_T, evaluate_datis_pool
from baseline_deepgini import evaluate_deepgini_pool
from baseline_dsa import (
    DATASET_CONFIG as DSA_DATASET_CONFIG,
    evaluate_dsa_pool,
)
from baseline_mcp import MCP_RATIO_MIN, evaluate_mcp_pool
from baseline_mc import MC_N_PASSES, evaluate_mc_pool
from baseline_nns import (
    DATASET_CONFIG as NNS_DATASET_CONFIG,
    NNS_K,
    NNS_LAMBDA,
    evaluate_nns_pool,
)
from baseline_nss import (
    DATASET_CONFIG as NSS_DATASET_CONFIG,
    NSS_IDENTIFIER_SEED,
    NSS_SENSITIVE_FRACTION,
    evaluate_nss_pool,
)
from baseline_random import evaluate_random_pool
from baseline_sets import SETS_UNCERTAINTY, DATASET_CONFIG as SETS_DATASET_CONFIG, evaluate_sets_pool
from training_models.load_data import load_cifar10, load_fmnist, load_svhn
from risk_scoring import (
    DEFAULT_RISK_FEATURE_KEYS,
    build_or_load_class_prototypes_dict,
    build_or_load_mahalanobis_stats,
    build_or_load_risk_features,
    compute_trc_by_budget,
    risk_scoring_function,
    required_risk_feature_keys,
)

METHOD_OURS = 'ours_pure_risk'
METHOD_DEEPGINI = 'deepgini'
METHOD_DATIS = 'datis_uncertainty'
METHOD_MAXP = 'maxp'
METHOD_NSS = 'nss'
METHOD_NNS = 'nns'
METHOD_DSA = 'dsa'
METHOD_MCP = 'mcp'
METHOD_MC = 'mc'
METHOD_RANDOM = 'random'
ALL_METHODS = (
    METHOD_OURS, METHOD_DEEPGINI, METHOD_DATIS, METHOD_MAXP, METHOD_NSS, METHOD_NNS,
    METHOD_DSA, METHOD_MCP, METHOD_MC, METHOD_RANDOM,
)

RQ1_BUDGET_RATIOS = [0.01, 0.03, 0.05, 0.10]
RQ1_SEEDS = [0, 1, 2, 3, 4]
RQ1_POOL_TYPES = ['adversarial', 'transformation']
RQ1_ERROR_RATIOS = [0.10, 0.20]
ERROR_SAMPLING_MODES = ('random', 'high_conf')

METHOD_COLORS = {
    METHOD_OURS: '#1f77b4',
    METHOD_DEEPGINI: '#ff7f0e',
    METHOD_DATIS: '#2ca02c',
    METHOD_MAXP: '#9467bd',
    METHOD_NSS: '#d62728',
    METHOD_NNS: '#17becf',
    METHOD_DSA: '#e377c2',
    METHOD_MCP: '#8c564b',
    METHOD_MC: '#bcbd22',
    METHOD_RANDOM: '#7f7f7f',
}
METHOD_MARKERS = {
    METHOD_OURS: 'o',
    METHOD_DEEPGINI: 's',
    METHOD_DATIS: 'D',
    METHOD_MAXP: 'P',
    METHOD_NSS: 'X',
    METHOD_NNS: 'v',
    METHOD_DSA: '*',
    METHOD_MCP: 'h',
    METHOD_MC: 'p',
    METHOD_RANDOM: '^',
}

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        'distance_layer_index': -4,
        'consistency_layer_indices': [-10, -8, -6, -4],
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
        'distance_layer_index': -5,
        'consistency_layer_indices': [-19, -15, -11, -5],
    },
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
        'distance_layer_index': -4,
        'consistency_layer_indices': [-40, -23, -14, -4],
    },
}


def _budget_key(budget_ratio):
    return round(float(budget_ratio), 4)


def _trc_by_budget_dict(budget_ratios, trc_result):
    trc = np.asarray(trc_result['trc'], dtype=np.float64).reshape(-1)
    if len(trc) != len(budget_ratios):
        raise ValueError('curve TRC length mismatch with requested budget ratios')
    return {_budget_key(b): float(trc[i]) for i, b in enumerate(budget_ratios)}


def _lookup_trc(trc_by_budget, budget_ratio):
    return trc_by_budget[_budget_key(budget_ratio)]


def _as_label_vector(arr):
    out = np.asarray(arr)
    if out.ndim > 1:
        out = np.argmax(out, axis=-1)
    return out.reshape(-1).astype(np.int64, copy=False)


def _fmt_mean_std(mean_val, std_val):
    if not np.isfinite(mean_val):
        return 'nan'
    if not np.isfinite(std_val):
        return f'{mean_val:.4f}±nan'
    return f'{mean_val:.4f}±{std_val:.4f}'


def load_sampled_pool(sampled_root, dataset_name, pool_type, error_ratio, seed, sampling_mode='random'):
    ratio_name = f'error_ratio_{int(round(float(error_ratio) * 100)):02d}'
    stem = f'seed_{int(seed)}_{sampling_mode}'
    npz_path = Path(sampled_root) / dataset_name / pool_type / ratio_name / f'{stem}.npz'
    json_path = npz_path.with_suffix('.json')
    if not npz_path.is_file():
        raise FileNotFoundError(f'missing sampled pool: {npz_path}')
    z = np.load(npz_path)
    metadata = json.loads(json_path.read_text(encoding='utf-8')) if json_path.is_file() else {}
    return {
        'data': np.asarray(z['data']),
        'clean_labels': np.asarray(z['clean_labels'], dtype=np.int64).reshape(-1),
        'is_error': np.asarray(z['is_error'], dtype=bool).reshape(-1),
        'predictions': np.asarray(z['predictions'], dtype=np.int64).reshape(-1),
        'metadata': metadata,
        'path': npz_path,
    }


def explore_feature_cache_name(dataset_name, pool_type, error_ratio, seed, sampling_mode='random'):
    ratio_pct = int(round(float(error_ratio) * 100))
    tag = f'explore_{dataset_name}_{pool_type}_r{ratio_pct:02d}_s{int(seed)}_{sampling_mode}'
    return f'all_risk_features_{tag}.npz'


def evaluate_ours_pure_risk(
    pool,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    seed,
    sampling_mode,
    cnn_model,
    prototypes_by_layer,
    class_means,
    class_inv_covs,
    dataset_cfg,
    cache_dir,
    budget_ratios,
    force_recompute=False,
):
    pool_data = pool['data']
    n_pool = len(pool_data)
    feature_cache_name = explore_feature_cache_name(
        dataset_name, pool_type, error_ratio, seed, sampling_mode,
    )
    required_keys = required_risk_feature_keys(dataset_cfg['consistency_layer_indices'])

    risk_features = build_or_load_risk_features(
        cache_dir=cache_dir,
        cache_name=feature_cache_name,
        data=pool_data,
        model=cnn_model,
        prototypes_by_layer_map=prototypes_by_layer,
        distance_feature_layer_index=dataset_cfg['distance_layer_index'],
        consistency_feature_layer_indices=dataset_cfg['consistency_layer_indices'],
        batch_size=16,
        force_recompute=force_recompute,
        class_means=class_means,
        class_inv_covs=class_inv_covs,
        required_feature_keys=required_keys,
        write_cache=False,
    )
    risk_scores = risk_scoring_function(
        risk_features,
        feature_keys=list(DEFAULT_RISK_FEATURE_KEYS),
        sample_indices=np.arange(n_pool, dtype=np.int64),
    )
    trc_result = compute_trc_by_budget(
        risk_score=risk_scores['risk_score'],
        error_mask=pool['is_error'],
        budget_ratios=budget_ratios,
    )
    ratio_to_trc = _trc_by_budget_dict(budget_ratios, trc_result)
    return {
        'method': METHOD_OURS,
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_errors': int(trc_result['total_errors']),
        'feature_keys': list(DEFAULT_RISK_FEATURE_KEYS),
        'trc_by_budget': ratio_to_trc,
    }


def aggregate_results(run_rows):
    groups = {}
    for row in run_rows:
        key = (
            row['method'],
            row['dataset'],
            row['pool_type'],
            row['sampling_mode'],
            row['error_ratio'],
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (method, dataset, pool_type, sampling_mode, error_ratio), rows in sorted(groups.items()):
        for budget in RQ1_BUDGET_RATIOS:
            vals = [_lookup_trc(r['trc_by_budget'], budget) for r in rows]
            arr = np.asarray(vals, dtype=np.float64)
            summary_rows.append({
                'method': method,
                'dataset': dataset,
                'pool_type': pool_type,
                'sampling_mode': sampling_mode,
                'error_ratio': error_ratio,
                'budget': float(budget),
                'seed_trc': {
                    int(r['seed']): _lookup_trc(r['trc_by_budget'], budget) for r in rows
                },
                'mean_trc': float(np.nanmean(arr)),
                'std_trc': float(np.nanstd(arr, ddof=0)),
                'n_seeds': len(rows),
            })
    return summary_rows


def _format_method_detail_table(summary_rows, seeds, sampling_mode, method):
    seed_headers = [f'seed{s}' for s in seeds]
    headers = [
        'method', 'dataset', 'pool_type', 'error_ratio', 'budget',
        *seed_headers, 'mean', 'std',
    ]
    lines = [
        f'### method = `{method}`',
        '',
        ' | '.join(headers),
        ' | '.join(['---'] * len(headers)),
    ]
    rows = [
        r for r in summary_rows
        if r['sampling_mode'] == sampling_mode and r['method'] == method
    ]
    for row in rows:
        seed_vals = row['seed_trc']
        cells = [
            row['method'],
            row['dataset'],
            row['pool_type'],
            f'{row["error_ratio"]:.2f}',
            f'{row["budget"]:.0%}',
        ]
        for seed in seeds:
            val = seed_vals.get(seed, np.nan)
            cells.append('nan' if not np.isfinite(val) else f'{val:.4f}')
        cells.append(f'{row["mean_trc"]:.4f}')
        cells.append(f'{row["std_trc"]:.4f}')
        lines.append(' | '.join(cells))
    return '\n'.join(lines)


def _format_method_comparison_table(
    summary_rows,
    sampling_mode,
    methods,
    *,
    with_std=True,
    heading=None,
):
    if heading is None:
        heading = (
            '### method mean±std comparison'
            if with_std
            else '### method mean comparison'
        )
    headers = ['dataset', 'pool_type', 'error_ratio', 'budget', *[f'{m}' for m in methods]]
    lines = [
        heading,
        '',
        ' | '.join(headers),
        ' | '.join(['---'] * len(headers)),
    ]
    by_key = {}
    for row in summary_rows:
        if row['sampling_mode'] != sampling_mode:
            continue
        key = (
            row['dataset'],
            row['pool_type'],
            float(row['error_ratio']),
            float(row['budget']),
        )
        by_key.setdefault(key, {})[row['method']] = row

    for key in sorted(by_key.keys()):
        dataset, pool_type, error_ratio, budget = key
        cells = [
            dataset,
            pool_type,
            f'{error_ratio:.2f}',
            f'{budget:.0%}',
        ]
        method_map = by_key[key]
        for method in methods:
            row = method_map.get(method)
            if row is None:
                cells.append('nan')
            elif with_std:
                cells.append(_fmt_mean_std(row['mean_trc'], row['std_trc']))
            else:
                mean_val = row['mean_trc']
                cells.append('nan' if not np.isfinite(mean_val) else f'{mean_val:.4f}')
        lines.append(' | '.join(cells))
    return '\n'.join(lines)


def format_results_markdown(summary_rows, seeds, sampling_modes, methods):
    parts = [
        '# RQ1 pure-risk ranking TRC (no diversity)',
        '',
        f'- Methods: `{list(methods)}`',
        f'- Ours risk features: `{list(DEFAULT_RISK_FEATURE_KEYS)}`',
        f'- DATIS uncertainty: k={DATIS_K}, T={DATIS_T} (no redundancy)',
        f'- SETS stage-1: `{SETS_UNCERTAINTY}` (1 - max softmax, no diversity)',
        f'- NSS: paper L1 TNSScore on last-encoder top-{NSS_SENSITIVE_FRACTION:.0%} neurons '
        f'(identifier_seed={NSS_IDENTIFIER_SEED}, no diversity)',
        f'- NNS: kNN softmax smoothing + DeepGini (k={NNS_K}, lambda={NNS_LAMBDA}, '
        f'last-encoder cosine kNN, no NNS_R)',
        f'- DSA: last-hidden surprise adequacy (predicted-class 1-NN, no LSA/coverage)',
        f'- MCP: full clustering + prioritization (per-budget independent sets, '
        f'ratio_min={MCP_RATIO_MIN})',
        f'- MC: dropout variation ratio ({MC_N_PASSES} stochastic forwards, prefix ranking)',
        f'- Budgets: {list(RQ1_BUDGET_RATIOS)}',
        f'- Seeds: {list(seeds)}',
        f'- Sampling modes: {list(sampling_modes)}',
        '',
    ]
    for mode in sampling_modes:
        parts.append(f'## sampling_mode = `{mode}`')
        parts.append('')
        for method in methods:
            parts.append(_format_method_detail_table(summary_rows, seeds, mode, method))
            parts.append('')
        parts.append(_format_method_comparison_table(
            summary_rows, mode, methods, with_std=True,
        ))
        parts.append('')

    parts.append('## method mean comparison (no std)')
    parts.append('')
    for mode in sampling_modes:
        parts.append(_format_method_comparison_table(
            summary_rows,
            mode,
            methods,
            with_std=False,
            heading=f'### sampling_mode = `{mode}`',
        ))
        parts.append('')
    return '\n'.join(parts).rstrip() + '\n'


def _panel_title(dataset, pool_type, sampling_mode, error_ratio):
    return f'{dataset} | {pool_type} | {sampling_mode} | err={float(error_ratio):.0%}'


def _panel_filename_stem(dataset, pool_type, sampling_mode, error_ratio):
    err_pct = int(round(float(error_ratio) * 100))
    return f'rq1_pure_risk_trc_{dataset}_{pool_type}_{sampling_mode}_err{err_pct:02d}'


def _draw_trc_panel(
    ax,
    summary_rows,
    *,
    dataset,
    pool_type,
    sampling_mode,
    error_ratio,
    methods,
    budgets=RQ1_BUDGET_RATIOS,
    show_legend=True,
):
    """One axes: one error_ratio; color=method; y = seed-mean TRC."""
    # Equal spacing for budget ticks (1%/3%/5%/10%), not proportional to ratio value.
    x = np.arange(len(budgets), dtype=np.float64)
    x_labels = [f'{b:.0%}' for b in budgets]
    er_key = round(float(error_ratio), 4)

    by_key = {}
    for row in summary_rows:
        if row['dataset'] != dataset:
            continue
        if row['pool_type'] != pool_type:
            continue
        if row['sampling_mode'] != sampling_mode:
            continue
        if round(float(row['error_ratio']), 4) != er_key:
            continue
        by_key[(row['method'], float(row['budget']))] = float(row['mean_trc'])

    for method in methods:
        ys = [by_key.get((method, float(b)), np.nan) for b in budgets]
        if not np.any(np.isfinite(ys)):
            continue
        ax.plot(
            x,
            ys,
            color=METHOD_COLORS.get(method, None),
            linestyle='-',
            marker=METHOD_MARKERS.get(method, 'o'),
            linewidth=2.0,
            markersize=5,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.2, len(budgets) - 0.8)
    ax.set_xlabel('budget')
    ax.set_ylabel('TRC')
    ax.set_ylim(0.0, 1.02)
    ax.set_title(_panel_title(dataset, pool_type, sampling_mode, error_ratio), fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    if show_legend:
        ax.legend(fontsize=6, loc='best', frameon=True)


def plot_pure_risk_trc_figures(
    summary_rows,
    *,
    out_dir,
    methods,
    datasets=None,
    pool_types=None,
    error_ratios=None,
    sampling_modes=None,
    budgets=RQ1_BUDGET_RATIOS,
):
    """
    Per dataset: 8 independent figures (error_ratio × pool_type × sampling_mode).
    Then one overview canvas per error_ratio: rows=dataset, cols=pool×mode.
    Color=method.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = list(methods)
    datasets = list(datasets) if datasets is not None else list(DATASET_CONFIG)
    pool_types = list(pool_types) if pool_types is not None else list(RQ1_POOL_TYPES)
    error_ratios = list(error_ratios) if error_ratios is not None else list(RQ1_ERROR_RATIOS)
    sampling_modes = (
        list(sampling_modes) if sampling_modes is not None else list(ERROR_SAMPLING_MODES)
    )
    col_specs = [(p, m) for p in pool_types for m in sampling_modes]

    saved = []
    for dataset in datasets:
        for error_ratio in error_ratios:
            for pool_type, sampling_mode in col_specs:
                fig, ax = plt.subplots(figsize=(4.5, 3.2))
                _draw_trc_panel(
                    ax,
                    summary_rows,
                    dataset=dataset,
                    pool_type=pool_type,
                    sampling_mode=sampling_mode,
                    error_ratio=error_ratio,
                    methods=methods,
                    budgets=budgets,
                    show_legend=True,
                )
                fig.tight_layout()
                path = out_dir / f'{_panel_filename_stem(dataset, pool_type, sampling_mode, error_ratio)}.png'
                fig.savefig(path, dpi=150)
                plt.close(fig)
                saved.append(path)
                print(f'Saved plot: {path}')

    n_rows = len(datasets)
    n_cols = len(col_specs)
    for error_ratio in error_ratios:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.4 * n_cols, 2.6 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for r, dataset in enumerate(datasets):
            for c, (pool_type, sampling_mode) in enumerate(col_specs):
                _draw_trc_panel(
                    axes[r][c],
                    summary_rows,
                    dataset=dataset,
                    pool_type=pool_type,
                    sampling_mode=sampling_mode,
                    error_ratio=error_ratio,
                    methods=methods,
                    budgets=budgets,
                    show_legend=(r == 0 and c == n_cols - 1),
                )
        err_pct = int(round(float(error_ratio) * 100))
        fig.suptitle(
            f'RQ1 pure-risk TRC (err={float(error_ratio):.0%}, color=method)',
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()
        overview_path = out_dir / f'rq1_pure_risk_trc_overview_err{err_pct:02d}.png'
        fig.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(overview_path)
        print(f'Saved overview plot: {overview_path}')
    return saved


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'RQ1 pure risk-score ranking TRC (no Snorkel/greedy diversity): '
            'ours + DeepGini + DATIS uncertainty + maxp + NSS + NNS + DSA + MCP + MC + random '
            'on sampled pools.'
        ),
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=list(DATASET_CONFIG),
        choices=sorted(DATASET_CONFIG),
    )
    parser.add_argument('--pool-types', nargs='+', default=RQ1_POOL_TYPES, choices=RQ1_POOL_TYPES)
    parser.add_argument('--error-ratios', nargs='+', type=float, default=RQ1_ERROR_RATIOS)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(RQ1_SEEDS))
    parser.add_argument(
        '--error-sampling-modes',
        nargs='+',
        default=list(ERROR_SAMPLING_MODES),
        choices=list(ERROR_SAMPLING_MODES),
        help='error sampling modes to evaluate (both written into one result set)',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=list(ALL_METHODS),
        choices=list(ALL_METHODS),
    )
    parser.add_argument('--sampled-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--output-dir', default=str(_EXP_DIR / 'results' / 'rq1'))
    parser.add_argument('--deepgini-batch-size', type=int, default=64)
    parser.add_argument(
        '--rng-seed-offset',
        type=int,
        default=0,
        help='added to each pool seed for random baseline shuffle',
    )
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='skip evaluation; plot from rq1_pure_risk_trc_summary.json in --output-dir',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    methods = tuple(args.methods)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        summary_path = out_dir / 'rq1_pure_risk_trc_summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(f'missing summary for --plot-only: {summary_path}')
        summary_rows = json.loads(summary_path.read_text(encoding='utf-8'))
        plot_methods = list(dict.fromkeys(r['method'] for r in summary_rows))
        if args.methods != list(ALL_METHODS):
            plot_methods = [m for m in args.methods if m in plot_methods]
        plot_pure_risk_trc_figures(
            summary_rows,
            out_dir=out_dir,
            methods=plot_methods,
            datasets=args.datasets,
            pool_types=args.pool_types,
            error_ratios=args.error_ratios,
            sampling_modes=args.error_sampling_modes,
        )
        return

    run_rows = []
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIG[dataset_name]
        need_model = (
            METHOD_OURS in methods
            or METHOD_DEEPGINI in methods
            or METHOD_DATIS in methods
            or METHOD_MAXP in methods
            or METHOD_NSS in methods
            or METHOD_NNS in methods
            or METHOD_DSA in methods
            or METHOD_MCP in methods
            or METHOD_MC in methods
        )
        if need_model and not Path(cfg['model_path']).is_file():
            raise FileNotFoundError(f'missing model: {cfg["model_path"]}')

        print(f'=== dataset: {dataset_name} ===')
        cnn_model = None
        prototypes_by_layer = None
        class_means = None
        class_inv_covs = None
        x_train = None
        y_train = None
        x_test = None
        cache_dir = _EXP_DIR / 'cache_files' / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        if need_model:
            cnn_model = tf.keras.models.load_model(cfg['model_path'], compile=False)

        if (
            METHOD_OURS in methods
            or METHOD_DATIS in methods
            or METHOD_NSS in methods
            or METHOD_DSA in methods
        ):
            x_train, y_train, x_test, _ = cfg['loader']()
            y_train = _as_label_vector(y_train)

        if METHOD_OURS in methods:
            all_prototype_layers = list(dict.fromkeys(
                [cfg['distance_layer_index']] + cfg['consistency_layer_indices'],
            ))
            prototypes_by_layer = build_or_load_class_prototypes_dict(
                cnn_model,
                train_data=x_train,
                train_labels=y_train,
                layer_indices=all_prototype_layers,
                dataset_name=dataset_name,
                batch_size=64,
            )
            class_means, class_inv_covs = build_or_load_mahalanobis_stats(
                cnn_model,
                train_data=x_train,
                train_labels=y_train,
                layer_index=cfg['distance_layer_index'],
                dataset_name=dataset_name,
                batch_size=64,
            )

        for pool_type in args.pool_types:
            for error_ratio in args.error_ratios:
                for sampling_mode in args.error_sampling_modes:
                    for seed in args.seeds:
                        pool = load_sampled_pool(
                            args.sampled_root,
                            dataset_name,
                            pool_type,
                            error_ratio,
                            seed,
                            sampling_mode,
                        )
                        n_pool = len(pool['data'])
                        n_errors = int(np.sum(pool['is_error']))
                        print(
                            f'Pool {dataset_name}/{pool_type}/{sampling_mode}/'
                            f'ratio={error_ratio:.2f}/seed={seed} '
                            f'(n={n_pool}, errors={n_errors})',
                        )

                        if METHOD_OURS in methods:
                            print(f'  -> {METHOD_OURS}')
                            row = evaluate_ours_pure_risk(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                cnn_model=cnn_model,
                                prototypes_by_layer=prototypes_by_layer,
                                class_means=class_means,
                                class_inv_covs=class_inv_covs,
                                dataset_cfg=cfg,
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                force_recompute=args.force_recompute,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_DATIS in methods:
                            print(f'  -> {METHOD_DATIS}')
                            row = evaluate_datis_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                model=cnn_model,
                                x_train=x_train,
                                y_train=y_train,
                                support_layer_index=cfg['distance_layer_index'],
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                apply_redundancy=False,
                                batch_size=int(args.deepgini_batch_size),
                                force_recompute=args.force_recompute,
                                k=DATIS_K,
                                T=DATIS_T,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_MAXP in methods:
                            print(f'  -> {METHOD_MAXP}')
                            row = evaluate_sets_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                model=cnn_model,
                                sets_feature_layer_index=SETS_DATASET_CONFIG[dataset_name][
                                    'sets_feature_layer_index'
                                ],
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                apply_diversity=False,
                                uncertainty=SETS_UNCERTAINTY,
                                batch_size=int(args.deepgini_batch_size),
                                force_recompute=args.force_recompute,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_NSS in methods:
                            print(f'  -> {METHOD_NSS}')
                            row = evaluate_nss_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                model=cnn_model,
                                x_test=x_test,
                                nss_feature_layer_index=NSS_DATASET_CONFIG[dataset_name][
                                    'nss_feature_layer_index'
                                ],
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                sensitive_fraction=NSS_SENSITIVE_FRACTION,
                                identifier_seed=NSS_IDENTIFIER_SEED,
                                mutation_seed=int(seed),
                                batch_size=int(args.deepgini_batch_size),
                                force_recompute=args.force_recompute,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_NNS in methods:
                            print(f'  -> {METHOD_NNS}')
                            row = evaluate_nns_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                model=cnn_model,
                                nns_feature_layer_index=NNS_DATASET_CONFIG[dataset_name][
                                    'nns_feature_layer_index'
                                ],
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                k=NNS_K,
                                lam=NNS_LAMBDA,
                                batch_size=int(args.deepgini_batch_size),
                                force_recompute=args.force_recompute,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_DSA in methods:
                            print(f'  -> {METHOD_DSA}')
                            row = evaluate_dsa_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                model=cnn_model,
                                x_train=x_train,
                                dsa_feature_layer_index=DSA_DATASET_CONFIG[dataset_name][
                                    'dsa_feature_layer_index'
                                ],
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                batch_size=int(args.deepgini_batch_size),
                                force_recompute=args.force_recompute,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_MCP in methods:
                            print(f'  -> {METHOD_MCP}')
                            row = evaluate_mcp_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                model=cnn_model,
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                apply_clustering=True,
                                ratio_min=MCP_RATIO_MIN,
                                batch_size=int(args.deepgini_batch_size),
                                force_recompute=args.force_recompute,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_MC in methods:
                            print(f'  -> {METHOD_MC}')
                            row = evaluate_mc_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                model=cnn_model,
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                n_passes=MC_N_PASSES,
                                batch_size=int(args.deepgini_batch_size),
                                force_recompute=args.force_recompute,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_DEEPGINI in methods:
                            print(f'  -> {METHOD_DEEPGINI}')
                            row = evaluate_deepgini_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                model=cnn_model,
                                cache_dir=cache_dir,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                batch_size=int(args.deepgini_batch_size),
                                force_recompute=args.force_recompute,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

                        if METHOD_RANDOM in methods:
                            rng_seed = int(seed) + int(args.rng_seed_offset)
                            print(f'  -> {METHOD_RANDOM} (rng_seed={rng_seed})')
                            row = evaluate_random_pool(
                                pool,
                                dataset_name=dataset_name,
                                pool_type=pool_type,
                                error_ratio=error_ratio,
                                seed=seed,
                                sampling_mode=sampling_mode,
                                budget_ratios=RQ1_BUDGET_RATIOS,
                                rng_seed=rng_seed,
                            )
                            run_rows.append(row)
                            trc_str = ', '.join(
                                f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                                for b in RQ1_BUDGET_RATIOS
                            )
                            print(f'     TRC: {trc_str}')

    summary_rows = aggregate_results(run_rows)
    table_text = format_results_markdown(
        summary_rows, args.seeds, args.error_sampling_modes, methods,
    )
    print('\n=== RQ1 pure-risk TRC summary (mean over seeds) ===')
    print(table_text)

    runs_path = out_dir / 'rq1_pure_risk_trc_runs.json'
    summary_path = out_dir / 'rq1_pure_risk_trc_summary.json'
    table_path = out_dir / 'rq1_pure_risk_trc_summary_table.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text, encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')

    plot_pure_risk_trc_figures(
        summary_rows,
        out_dir=out_dir,
        methods=methods,
        datasets=args.datasets,
        pool_types=args.pool_types,
        error_ratios=args.error_ratios,
        sampling_modes=args.error_sampling_modes,
    )


if __name__ == '__main__':
    main()
