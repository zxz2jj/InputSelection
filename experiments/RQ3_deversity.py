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

tf.get_logger().setLevel('ERROR')
logging.getLogger().setLevel(logging.WARNING)

from training_models.load_data import load_cifar10, load_fmnist
from pseudo_labelling import build_or_load_snorkel_result_from_risk_features
from risk_scoring import (
    build_or_load_class_prototypes_dict,
    build_or_load_risk_features,
    risk_scoring_function,
)
from selection_method import (
    build_or_load_greedy_run,
    compute_flat_hidden_vectors_at_layer,
    real_error_type_universe_from_storage,
    real_error_types_covered_in_prefix,
)

RQ2_BUDGET_RATIOS = [0.01, 0.03, 0.05, 0.10]
RQ2_SEEDS = [0, 1, 2, 3, 4]
RQ2_POOL_TYPES = ['adversarial', 'transformation']
RQ2_ERROR_RATIOS = [0.10, 0.20]
ERROR_SAMPLING_MODES = ('random', 'high_conf')

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        'assist_model_path': _REPO_ROOT / 'models' / 'resnet18_fmnist' / 'tf_model.h5',
        'distance_layer_index': -4,
        'consistency_layer_indices': [-10, -8, -6, -4],
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
        'assist_model_path': _REPO_ROOT / 'models' / 'resnet18_cifar10' / 'tf_model.h5',
        'distance_layer_index': -5,
        'consistency_layer_indices': [-19, -15, -11, -5],
    },
}

_GD_EPS = 1e-10


def _budget_key(budget_ratio):
    return round(float(budget_ratio), 4)


def _metric_by_budget_dict(budget_ratios, values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) != len(budget_ratios):
        raise ValueError('metric length mismatch with requested budget ratios')
    return {_budget_key(b): float(arr[i]) for i, b in enumerate(budget_ratios)}


def _lookup_metric(metric_by_budget, budget_ratio):
    return metric_by_budget[_budget_key(budget_ratio)]


def _as_label_vector(arr):
    out = np.asarray(arr)
    if out.ndim > 1:
        out = np.argmax(out, axis=-1)
    return out.reshape(-1).astype(np.int64, copy=False)


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


def pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode='random'):
    ratio_pct = int(round(float(error_ratio) * 100))
    return f'rq1_{dataset_name}_{pool_type}_r{ratio_pct:02d}_s{int(seed)}_{sampling_mode}'


def build_flat_pool_storage_and_attributes(
    pool_data,
    clean_labels,
    is_error,
    predictions,
    risk_out,
    topk_idx,
    topk_prob,
    hidden_vectors,
):
    n = int(len(pool_data))
    pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    clean = np.asarray(clean_labels, dtype=np.int64).reshape(-1)
    err_flag = np.asarray(is_error, dtype=bool).reshape(-1)
    if not (pred.shape[0] == clean.shape[0] == err_flag.shape[0] == n):
        raise ValueError('flat pool field length mismatch')

    rs = np.asarray(risk_out['risk_score'], dtype=np.float64).reshape(-1)
    si = np.asarray(risk_out['sample_index'], dtype=np.int64).reshape(-1)
    if rs.shape[0] != n or si.shape[0] != n:
        raise ValueError('risk_out length mismatch with pool')
    if not np.array_equal(si, np.arange(n, dtype=np.int64)):
        raise ValueError('risk_out sample_index must be 0..n-1 in order')

    tk = np.asarray(topk_idx, dtype=np.int64).reshape(n, -1)
    tp = np.asarray(topk_prob, dtype=np.float32).reshape(n, -1)
    if tk.shape[1] < 2 or tp.shape[1] < 2:
        raise ValueError('topk_idx/prob must have at least 2 columns')
    hv = np.asarray(hidden_vectors, dtype=np.float32)
    if hv.shape[0] != n:
        raise ValueError('hidden_vectors row count mismatch')

    storage_out = []
    attributes_out = []
    for i in range(n):
        pseudo = (
            (int(tk[i, 0]), float(tp[i, 0])),
            (int(tk[i, 1]), float(tp[i, 1])),
        )
        storage_out.append({
            'idx': i,
            'clean_label': int(clean[i]),
            'is_wrongly_predicted': bool(err_flag[i]),
            'prediction': int(pred[i]),
        })
        attributes_out.append({
            'idx': i,
            'risk_score': float(rs[i]),
            'prediction': int(pred[i]),
            'top2_soft_pseudo_labelling': pseudo,
            'hidden_vector': hv[i],
        })
    return storage_out, attributes_out


def _group_true_error_ids_in_prefix(chosen_prefix, storage_rows):
    rows_by_id = {int(row['idx']): row for row in storage_rows}
    groups = {}
    for sid in chosen_prefix:
        row = rows_by_id.get(int(sid))
        if row is None or not row['is_wrongly_predicted']:
            continue
        fault_type = (int(row['clean_label']), int(row['prediction']))
        groups.setdefault(fault_type, []).append(int(sid))
    return groups


def build_hidden_by_idx(attribute_records, hidden_key='hidden_vector'):
    hidden_by_sample = {}
    for rec in attribute_records:
        sid = int(rec['idx'])
        vec = np.asarray(rec[hidden_key], dtype=np.float64).reshape(-1)
        hidden_by_sample[sid] = vec
    return hidden_by_sample


def minmax_normalize_feature_matrix(feature_rows, eps=1e-12):
    """Per-column min-max normalization (Wei et al., TSE 2023)."""
    v = np.asarray(feature_rows, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError('feature_rows must be 2D')
    vmin = np.min(v, axis=0)
    vmax = np.max(v, axis=0)
    span = vmax - vmin
    constant = span <= float(eps)
    span = np.where(constant, 1.0, span)
    out = (v - vmin) / span
    out[:, constant] = 0.0
    return out


def geometric_diversity_score(feature_rows, eps=_GD_EPS):
    """GD(S) = det(V V^T) on min-max normalized feature rows (Wei et al., TSE 2023)."""
    v = minmax_normalize_feature_matrix(feature_rows)
    n = v.shape[0]
    if n < 2:
        return np.nan

    gram = v @ v.T
    gram = gram + np.eye(n, dtype=np.float64) * float(eps)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet))


def intra_type_gd_stats_for_prefix(chosen_prefix, storage_rows, hidden_by_idx):
    groups = _group_true_error_ids_in_prefix(chosen_prefix, storage_rows)
    gd_values = []
    for sids in groups.values():
        if len(sids) < 2:
            continue
        matrix = np.stack([hidden_by_idx[int(sid)] for sid in sids], axis=0)
        gd = geometric_diversity_score(matrix)
        if np.isfinite(gd):
            gd_values.append(gd)
    if not gd_values:
        return np.nan, 0
    return float(np.sum(gd_values)), int(len(gd_values))


def compute_diversity_curves(
    storage_rows,
    selection_order,
    hidden_by_idx,
    budget_ratios=None,
):
    if budget_ratios is None:
        budget_ratios = RQ2_BUDGET_RATIOS

    n_pool = len(storage_rows)
    real_universe = real_error_type_universe_from_storage(storage_rows)
    total_fault_types = len(real_universe)
    chosen = [int(i) for i in selection_order]

    out_fault_type_count = []
    out_sum_intra_type_gd = []
    out_intra_type_gd_type_count = []

    for r in budget_ratios:
        rr = float(r)
        if rr <= 0 or rr > 1:
            raise ValueError(f'budget ratio must be in (0, 1], got {rr}')
        k = int(np.ceil(rr * n_pool))
        k = max(1, min(k, n_pool))
        prefix = chosen[:k]

        discovered_types = real_error_types_covered_in_prefix(prefix, storage_rows, real_universe)
        ft_count = len(discovered_types)
        sum_gd, gd_type_count = intra_type_gd_stats_for_prefix(
            prefix, storage_rows, hidden_by_idx,
        )

        out_fault_type_count.append(ft_count)
        out_sum_intra_type_gd.append(sum_gd)
        out_intra_type_gd_type_count.append(gd_type_count)

    return {
        'total_fault_types': int(total_fault_types),
        'fault_type_count': np.asarray(out_fault_type_count, dtype=np.int32),
        'sum_intra_type_gd': np.asarray(out_sum_intra_type_gd, dtype=np.float64),
        'intra_type_gd_type_count': np.asarray(out_intra_type_gd_type_count, dtype=np.int32),
        'chosen_ids': np.asarray(chosen, dtype=np.int64),
    }


def evaluate_sampled_pool(
    pool,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    seed,
    sampling_mode,
    cnn_model,
    assist_model,
    prototypes_by_layer,
    dataset_cfg,
    cache_dir,
    greedy_cfg,
    max_budget_ratio,
    budget_ratios,
    force_recompute=False,
):
    pool_data = pool['data']
    n_pool = len(pool_data)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)

    risk_features = build_or_load_risk_features(
        cache_dir=cache_dir,
        cache_name=f'risk_features_{tag}.npz',
        data=pool_data,
        model=cnn_model,
        prototypes_by_layer_map=prototypes_by_layer,
        distance_feature_layer_index=dataset_cfg['distance_layer_index'],
        consistency_feature_layer_indices=dataset_cfg['consistency_layer_indices'],
        batch_size=16,
        force_recompute=force_recompute,
    )
    risk_scores = risk_scoring_function(
        risk_features,
        sample_indices=np.arange(n_pool, dtype=np.int64),
    )

    snorkel_result = build_or_load_snorkel_result_from_risk_features(
        cache_dir=cache_dir,
        cache_name=f'snorkel_result_{tag}.npz',
        data=pool_data,
        risk_feature_map=risk_features,
        lf_assist_model=assist_model,
        lf_layer_indices=dataset_cfg['consistency_layer_indices'],
        cardinality=int(cnn_model.output_shape[-1]),
        gap_threshold=0.05,
        topk=2,
        force_recompute=force_recompute,
    )

    hidden_flat = compute_flat_hidden_vectors_at_layer(
        pool_data,
        cnn_model,
        dataset_cfg['distance_layer_index'],
    )
    sample_storage, sample_attributes = build_flat_pool_storage_and_attributes(
        pool_data,
        pool['clean_labels'],
        pool['is_error'],
        risk_features['pred_classes'],
        risk_scores,
        snorkel_result['topk_idx'],
        snorkel_result['topk_prob'],
        hidden_flat,
    )
    hidden_by_idx = build_hidden_by_idx(sample_attributes)

    greedy_order, _greedy_run = build_or_load_greedy_run(
        cache_dir=cache_dir,
        cache_name=f'greedy_run_{tag}.npz',
        attribute_records=sample_attributes,
        selection_ratio=float(max_budget_ratio),
        greedy_cfg=greedy_cfg,
        force_recompute=force_recompute,
    )

    curve = compute_diversity_curves(
        sample_storage,
        greedy_order,
        hidden_by_idx,
        budget_ratios=budget_ratios,
    )
    return {
        'method': 'greedy',
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_fault_types': int(curve['total_fault_types']),
        'fault_type_count_by_budget': _metric_by_budget_dict(
            budget_ratios, curve['fault_type_count'],
        ),
        'sum_intra_type_gd_by_budget': _metric_by_budget_dict(
            budget_ratios, curve['sum_intra_type_gd'],
        ),
        'intra_type_gd_type_count_by_budget': _metric_by_budget_dict(
            budget_ratios, curve['intra_type_gd_type_count'],
        ),
    }


def aggregate_results(run_rows, metric_key, budget_ratios):
    groups = {}
    for row in run_rows:
        key = (row['method'], row['dataset'], row['pool_type'], row['sampling_mode'], row['error_ratio'])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (method, dataset, pool_type, sampling_mode, error_ratio), rows in sorted(groups.items()):
        for budget in budget_ratios:
            vals = [_lookup_metric(r[metric_key], budget) for r in rows]
            arr = np.asarray(vals, dtype=np.float64)
            summary_rows.append({
                'method': method,
                'dataset': dataset,
                'pool_type': pool_type,
                'sampling_mode': sampling_mode,
                'error_ratio': error_ratio,
                'budget': float(budget),
                'seed_values': {
                    int(r['seed']): _lookup_metric(r[metric_key], budget) for r in rows
                },
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr, ddof=0)),
                'n_seeds': len(rows),
            })
    return summary_rows


def format_results_table(summary_rows, metric_name, budget_ratios, seeds):
    headers = [
        'method', 'dataset', 'pool_type', 'sampling_mode', 'error_ratio', 'budget', metric_name,
    ] + [f'seed{s}' for s in seeds] + ['mean', 'std']
    lines = [
        ' | '.join(headers),
        ' | '.join(['---'] * len(headers)),
    ]
    for row in summary_rows:
        seed_vals = row['seed_values']
        cells = [
            row['method'],
            row['dataset'],
            row['pool_type'],
            row['sampling_mode'],
            f'{row["error_ratio"]:.2f}',
            f'{row["budget"]:.0%}',
            metric_name,
        ]
        for seed in seeds:
            val = seed_vals.get(seed, np.nan)
            if metric_name == 'fault_type_count':
                cells.append('nan' if not np.isfinite(val) else f'{int(round(val))}')
            else:
                cells.append('nan' if not np.isfinite(val) else f'{val:.6f}')
        cells.append(
            'nan' if not np.isfinite(row['mean']) else (
                f'{int(round(row["mean"]))}' if metric_name == 'fault_type_count' else f'{row["mean"]:.6f}'
            ),
        )
        cells.append(
            'nan' if not np.isfinite(row['std']) else (
                f'{row["std"]:.4f}' if metric_name == 'fault_type_count' else f'{row["std"]:.6f}'
            ),
        )
        lines.append(' | '.join(cells))
    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='RQ2 diversity evaluation (fault type count + sum intra-type GD) on sampled pools.',
    )
    parser.add_argument('--datasets', nargs='+', default=['fmnist', 'cifar10'], choices=sorted(DATASET_CONFIG))
    parser.add_argument('--pool-types', nargs='+', default=RQ2_POOL_TYPES, choices=RQ2_POOL_TYPES)
    parser.add_argument('--error-ratios', nargs='+', type=float, default=RQ2_ERROR_RATIOS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument(
        '--error-sampling-mode',
        default='random',
        choices=list(ERROR_SAMPLING_MODES),
        help='sampled pool filename suffix (seed_N_<mode>.npz)',
    )
    parser.add_argument('--sampled-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--output-dir', default=str(_EXP_DIR / 'results' / 'rq2'))
    parser.add_argument('--greedy-alpha', type=float, default=0.5)
    parser.add_argument('--greedy-phi-mode', default='sqrt')
    parser.add_argument('--greedy-risk-gate-power', type=float, default=3.0)
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    greedy_cfg = {
        'alpha': float(args.greedy_alpha),
        'phi_mode': str(args.greedy_phi_mode),
        'risk_gate_power': float(args.greedy_risk_gate_power),
    }
    max_budget = max(RQ2_BUDGET_RATIOS)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIG[dataset_name]
        if not Path(cfg['model_path']).is_file():
            raise FileNotFoundError(f'missing model: {cfg["model_path"]}')
        if not Path(cfg['assist_model_path']).is_file():
            raise FileNotFoundError(f'missing assist model: {cfg["assist_model_path"]}')

        print(f'=== dataset: {dataset_name} ===')
        cnn_model = tf.keras.models.load_model(cfg['model_path'], compile=False)
        assist_model = tf.keras.models.load_model(cfg['assist_model_path'], compile=False)
        x_train, y_train, _, _ = cfg['loader']()
        y_train = _as_label_vector(y_train)

        cache_dir = _EXP_DIR / 'cache_files' / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
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

        for pool_type in args.pool_types:
            for error_ratio in args.error_ratios:
                for seed in args.seeds:
                    pool = load_sampled_pool(
                        args.sampled_root,
                        dataset_name,
                        pool_type,
                        error_ratio,
                        seed,
                        args.error_sampling_mode,
                    )
                    print(
                        f'Evaluating diversity {dataset_name}/{pool_type}/'
                        f'{args.error_sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                        f'(n={len(pool["data"])}, errors={int(np.sum(pool["is_error"]))})',
                    )
                    row = evaluate_sampled_pool(
                        pool,
                        dataset_name=dataset_name,
                        pool_type=pool_type,
                        error_ratio=error_ratio,
                        seed=seed,
                        sampling_mode=args.error_sampling_mode,
                        cnn_model=cnn_model,
                        assist_model=assist_model,
                        prototypes_by_layer=prototypes_by_layer,
                        dataset_cfg=cfg,
                        cache_dir=cache_dir,
                        greedy_cfg=greedy_cfg,
                        max_budget_ratio=max_budget,
                        budget_ratios=RQ2_BUDGET_RATIOS,
                        force_recompute=args.force_recompute,
                    )
                    run_rows.append(row)
                    ft_str = ', '.join(
                        f'{b:.0%}={int(_lookup_metric(row["fault_type_count_by_budget"], b))}'
                        for b in RQ2_BUDGET_RATIOS
                    )
                    gd_sum_str = ', '.join(
                        f'{b:.0%}={_lookup_metric(row["sum_intra_type_gd_by_budget"], b):.6f}'
                        for b in RQ2_BUDGET_RATIOS
                    )
                    print(f'  fault_type_count: {ft_str}')
                    print(f'  sum_intra_type_gd: {gd_sum_str}')

    ft_summary = aggregate_results(run_rows, 'fault_type_count_by_budget', RQ2_BUDGET_RATIOS)
    gd_sum_summary = aggregate_results(run_rows, 'sum_intra_type_gd_by_budget', RQ2_BUDGET_RATIOS)
    ft_table = format_results_table(ft_summary, 'fault_type_count', RQ2_BUDGET_RATIOS, args.seeds)
    gd_sum_table = format_results_table(
        gd_sum_summary, 'sum_intra_type_gd', RQ2_BUDGET_RATIOS, args.seeds,
    )

    print('\n=== RQ2 fault_type_count summary (mean over seeds) ===')
    print(ft_table)
    print('\n=== RQ2 sum_intra_type_gd summary (mean over seeds) ===')
    print(gd_sum_table)

    mode_tag = args.error_sampling_mode
    runs_path = out_dir / f'rq2_diversity_runs_{mode_tag}.json'
    ft_summary_path = out_dir / f'rq2_fault_type_count_summary_{mode_tag}.json'
    gd_sum_summary_path = out_dir / f'rq2_sum_intra_type_gd_summary_{mode_tag}.json'
    ft_table_path = out_dir / f'rq2_fault_type_count_summary_table_{mode_tag}.md'
    gd_sum_table_path = out_dir / f'rq2_sum_intra_type_gd_summary_table_{mode_tag}.md'

    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    ft_summary_path.write_text(json.dumps(ft_summary, indent=2, sort_keys=True), encoding='utf-8')
    gd_sum_summary_path.write_text(json.dumps(gd_sum_summary, indent=2, sort_keys=True), encoding='utf-8')
    ft_table_path.write_text(ft_table + '\n', encoding='utf-8')
    gd_sum_table_path.write_text(gd_sum_table + '\n', encoding='utf-8')

    print(f'\nSaved runs: {runs_path}')
    print(f'Saved fault_type_count summary: {ft_summary_path}')
    print(f'Saved sum_intra_type_gd summary: {gd_sum_summary_path}')
    print(f'Saved fault_type_count table: {ft_table_path}')
    print(f'Saved sum_intra_type_gd table: {gd_sum_table_path}')


if __name__ == '__main__':
    main()
