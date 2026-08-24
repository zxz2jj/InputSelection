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


from training_models.load_data import load_cifar10, load_fmnist, load_svhn
from pseudo_labelling import build_or_load_snorkel_result_from_risk_features
from risk_scoring import (
    DEFAULT_RISK_FEATURE_KEYS,
    build_or_load_class_prototypes_dict,
    build_or_load_mahalanobis_stats,
    build_or_load_risk_features,
    risk_scoring_function,
    required_risk_feature_keys,
)
from selection_method import (
    build_or_load_greedy_run,
    compute_flat_hidden_vectors_at_layer,
    compute_greedy_selection_curves,
)


RQ1_BUDGET_RATIOS = [0.01, 0.03, 0.05, 0.10]
RQ1_SEEDS = [0, 1, 2, 3, 4]
RQ1_POOL_TYPES = ['adversarial', 'transformation']
RQ1_ERROR_RATIOS = [0.10, 0.20]
ERROR_SAMPLING_MODES = ('random', 'high_conf')


def _budget_key(budget_ratio):
    return round(float(budget_ratio), 4)


def _trc_by_budget_dict(budget_ratios, curve):
    trc = np.asarray(curve['trc'], dtype=np.float64).reshape(-1)
    if len(trc) != len(budget_ratios):
        raise ValueError('curve TRC length mismatch with requested budget ratios')
    return {_budget_key(b): float(trc[i]) for i, b in enumerate(budget_ratios)}


def _lookup_trc(trc_by_budget, budget_ratio):
    return trc_by_budget[_budget_key(budget_ratio)]

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
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
        'assist_model_path': _REPO_ROOT / 'models' / 'vgg19_svhn' / 'tf_model.h5',
        'distance_layer_index': -4,
        'consistency_layer_indices': [-40, -23, -14, -4],
    },
}


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
    """Build storage/attributes for a shuffled mixed pool (sampled_data format)."""
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
            'data': pool_data[i],
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
    """RQ1-specific tag for snorkel/greedy caches."""
    ratio_pct = int(round(float(error_ratio) * 100))
    return f'rq1_{dataset_name}_{pool_type}_r{ratio_pct:02d}_s{int(seed)}_{sampling_mode}'


def explore_feature_cache_name(dataset_name, pool_type, error_ratio, seed, sampling_mode='random'):
    """Shared explore-phase feature cache (Ablation_risk_feature_exploration)."""
    ratio_pct = int(round(float(error_ratio) * 100))
    tag = f'explore_{dataset_name}_{pool_type}_r{ratio_pct:02d}_s{int(seed)}_{sampling_mode}'
    return f'all_risk_features_{tag}.npz'


def _as_label_vector(arr):
    out = np.asarray(arr)
    if out.ndim > 1:
        out = np.argmax(out, axis=-1)
    return out.reshape(-1).astype(np.int64, copy=False)


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
    class_means,
    class_inv_covs,
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

    greedy_order, _greedy_run = build_or_load_greedy_run(
        cache_dir=cache_dir,
        cache_name=f'greedy_run_{tag}.npz',
        attribute_records=sample_attributes,
        selection_ratio=float(max_budget_ratio),
        greedy_cfg=greedy_cfg,
        force_recompute=force_recompute,
    )

    curve = compute_greedy_selection_curves(
        sample_storage,
        greedy_order,
        None,
        budget_ratios=budget_ratios,
    )
    ratio_to_trc = _trc_by_budget_dict(budget_ratios, curve)
    return {
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_errors': int(curve['total_errors']),
        'trc_by_budget': ratio_to_trc,
    }


def aggregate_results(run_rows):
    groups = {}
    for row in run_rows:
        key = (row['dataset'], row['pool_type'], row['sampling_mode'], row['error_ratio'])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (dataset, pool_type, sampling_mode, error_ratio), rows in sorted(groups.items()):
        for budget in RQ1_BUDGET_RATIOS:
            vals = [_lookup_trc(r['trc_by_budget'], budget) for r in rows]
            arr = np.asarray(vals, dtype=np.float64)
            summary_rows.append({
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


def format_results_table(summary_rows, seeds):
    seed_headers = [f'seed{s}' for s in seeds]
    headers = [
        'dataset', 'pool_type', 'sampling_mode', 'error_ratio', 'budget',
        *seed_headers, 'mean', 'std',
    ]
    lines = [
        ' | '.join(headers),
        ' | '.join(['---'] * len(headers)),
    ]
    for row in summary_rows:
        seed_vals = row['seed_trc']
        cells = [
            row['dataset'],
            row['pool_type'],
            row['sampling_mode'],
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


def parse_args():
    parser = argparse.ArgumentParser(description='RQ1 greedy TRC evaluation on sampled pools.')
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
        '--error-sampling-mode',
        default='random',
        choices=list(ERROR_SAMPLING_MODES),
        help='sampled pool filename suffix (seed_N_<mode>.npz)',
    )
    parser.add_argument('--sampled-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--output-dir', default=str(_EXP_DIR / 'results' / 'rq1'))
    parser.add_argument('--greedy-alpha', type=float, default=0.5)
    parser.add_argument('--greedy-phi-mode', default='sqrt')
    parser.add_argument('--greedy-risk-gate-power', type=float, default=4.0)
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    greedy_cfg = {
        'alpha': float(args.greedy_alpha),
        'phi_mode': str(args.greedy_phi_mode),
        'risk_gate_power': float(args.greedy_risk_gate_power),
    }
    max_budget = max(RQ1_BUDGET_RATIOS)

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
                        f'Evaluating {dataset_name}/{pool_type}/'
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
                        class_means=class_means,
                        class_inv_covs=class_inv_covs,
                        dataset_cfg=cfg,
                        cache_dir=cache_dir,
                        greedy_cfg=greedy_cfg,
                        max_budget_ratio=max_budget,
                        budget_ratios=RQ1_BUDGET_RATIOS,
                        force_recompute=args.force_recompute,
                    )
                    run_rows.append(row)
                    trc_str = ', '.join(
                        f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                        for b in RQ1_BUDGET_RATIOS
                    )
                    print(f'  TRC: {trc_str}')

    summary_rows = aggregate_results(run_rows)
    table_text = format_results_table(summary_rows, args.seeds)
    print('\n=== RQ1 TRC summary (mean over seeds) ===')
    print(table_text)

    mode_tag = args.error_sampling_mode
    runs_path = out_dir / f'rq1_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
