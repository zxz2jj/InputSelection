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

RQ1_BUDGET_RATIOS = [0.01, 0.03, 0.05, 0.10]
RQ1_SEEDS = [0, 1, 2, 3, 4]
RQ1_POOL_TYPES = ['adversarial', 'transformation']
RQ1_ERROR_RATIOS = [0.10, 0.20]
RQ1_DATASETS = ['fmnist', 'cifar10', 'svhn']
ERROR_SAMPLING_MODES = ('random', 'high_conf')

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
    },
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
    },
}


def configure_tensorflow_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return []
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print(f'Warning: could not enable GPU memory growth: {exc}')
    return gpus


def _budget_key(budget_ratio):
    return round(float(budget_ratio), 4)


def _trc_by_budget_dict(budget_ratios, curve):
    trc = np.asarray(curve['trc'], dtype=np.float64).reshape(-1)
    if len(trc) != len(budget_ratios):
        raise ValueError('curve TRC length mismatch with requested budget ratios')
    return {_budget_key(b): float(trc[i]) for i, b in enumerate(budget_ratios)}


def _lookup_trc(trc_by_budget, budget_ratio):
    return trc_by_budget[_budget_key(budget_ratio)]


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


def build_pool_storage(pool):
    n = int(len(pool['clean_labels']))
    clean = np.asarray(pool['clean_labels'], dtype=np.int64).reshape(-1)
    pred = np.asarray(pool['predictions'], dtype=np.int64).reshape(-1)
    err_flag = np.asarray(pool['is_error'], dtype=bool).reshape(-1)
    if not (clean.shape[0] == pred.shape[0] == err_flag.shape[0] == n):
        raise ValueError('flat pool field length mismatch')

    storage_out = []
    for i in range(n):
        storage_out.append({
            'idx': i,
            'clean_label': int(clean[i]),
            'is_wrongly_predicted': bool(err_flag[i]),
            'prediction': int(pred[i]),
        })
    return storage_out


def _probs_from_model_output(raw_out):
    raw_out = tf.cast(tf.convert_to_tensor(raw_out), tf.float32)
    looks_like_probs = tf.reduce_all((raw_out >= 0.0) & (raw_out <= 1.0)) & tf.reduce_all(
        tf.abs(tf.reduce_sum(raw_out, axis=-1) - 1.0) < 1e-3,
    )
    return tf.cond(looks_like_probs, lambda: raw_out, lambda: tf.nn.softmax(raw_out, axis=-1))


def compute_deepgini_scores(data, model, batch_size=64):
    """DeepGini = 1 - sum_i p_i^2 over softmax probabilities."""
    scores = []
    n = len(data)
    for start in range(0, n, batch_size):
        batch = tf.convert_to_tensor(data[start:start + batch_size])
        raw = model(batch, training=False)
        probs = _probs_from_model_output(raw)
        gini = 1.0 - tf.reduce_sum(tf.square(probs), axis=-1)
        scores.append(np.asarray(gini.numpy(), dtype=np.float32).reshape(-1))
    out = np.concatenate(scores)
    if out.shape[0] != n:
        raise ValueError('deepgini score length mismatch with pool')
    return out


def build_or_load_deepgini_scores(
    *,
    cache_dir,
    cache_name,
    data,
    model,
    batch_size=64,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        scores = np.asarray(z['deepgini_score'], dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(data):
            raise ValueError(f'deepgini cache length mismatch: {cache_path}')
        print(f'Loaded DeepGini scores from {cache_path}')
        return scores

    scores = compute_deepgini_scores(data, model, batch_size=batch_size)
    np.savez_compressed(cache_path, deepgini_score=scores)
    print(f'Saved DeepGini scores to {cache_path}')
    return scores


def deepgini_selection_order(deepgini_scores):
    scores = np.asarray(deepgini_scores, dtype=np.float64).reshape(-1)
    n = scores.shape[0]
    order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
    return order.tolist()


def _error_mask_from_storage(storage_rows):
    n = len(storage_rows)
    err = np.zeros(n, dtype=bool)
    for row in storage_rows:
        err[int(row['idx'])] = bool(row['is_wrongly_predicted'])
    return err


def compute_selection_curves(storage_rows, selection_order, budget_ratios):
    """Same TRC definition as selection_method.compute_greedy_selection_curves."""
    err = _error_mask_from_storage(storage_rows)
    n_pool = len(storage_rows)
    total_errors = int(np.sum(err))
    chosen = [int(i) for i in selection_order]

    out_trc = []
    for r in budget_ratios:
        rr = float(r)
        if rr <= 0 or rr > 1:
            raise ValueError(f'budget ratio must be in (0, 1], got {rr}')
        k = int(np.ceil(rr * n_pool))
        k = max(1, min(k, n_pool))
        prefix = chosen[:k]
        discovered = int(np.sum(err[prefix]))
        denom = min(k, total_errors)
        trc = (discovered / denom) if denom > 0 else np.nan
        out_trc.append(trc)

    return {
        'total_errors': total_errors,
        'trc': out_trc,
    }


def evaluate_deepgini_pool(
    pool,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    seed,
    sampling_mode,
    model,
    cache_dir,
    budget_ratios,
    batch_size,
    force_recompute,
):
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)
    deepgini_scores = build_or_load_deepgini_scores(
        cache_dir=cache_dir,
        cache_name=f'deepgini_{tag}.npz',
        data=pool['data'],
        model=model,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    order = deepgini_selection_order(deepgini_scores)
    curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
    ratio_to_trc = _trc_by_budget_dict(budget_ratios, curve)
    return {
        'method': 'deepgini',
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
                'method': 'deepgini',
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
        'method', 'dataset', 'pool_type', 'sampling_mode', 'error_ratio', 'budget',
        *seed_headers, 'mean', 'std',
    ]
    lines = [
        ' | '.join(headers),
        ' | '.join(['---'] * len(headers)),
    ]
    for row in summary_rows:
        seed_vals = row['seed_trc']
        cells = [
            row['method'],
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
    parser = argparse.ArgumentParser(
        description='RQ1 DeepGini baseline TRC evaluation on sampled pools.',
    )
    parser.add_argument('--datasets', nargs='+', default=RQ1_DATASETS, choices=sorted(RQ1_DATASETS))
    parser.add_argument('--pool-types', nargs='+', default=RQ1_POOL_TYPES, choices=RQ1_POOL_TYPES)
    parser.add_argument('--error-ratios', nargs='+', type=float, default=RQ1_ERROR_RATIOS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument(
        '--error-sampling-mode',
        default='random',
        choices=list(ERROR_SAMPLING_MODES),
        help='sampled pool filename suffix (seed_N_<mode>.npz)',
    )
    parser.add_argument('--sampled-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--output-dir', default=str(_EXP_DIR / 'results' / 'rq1'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    configure_tensorflow_gpu()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIG[dataset_name]
        if not Path(cfg['model_path']).is_file():
            raise FileNotFoundError(f'missing model: {cfg["model_path"]}')

        print(f'=== dataset: {dataset_name} ===')
        model = tf.keras.models.load_model(cfg['model_path'], compile=False)
        cache_dir = _EXP_DIR / 'cache_files' / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)

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
                    n_pool = len(pool['clean_labels'])
                    n_errors = int(np.sum(pool['is_error']))
                    print(
                        f'Evaluating deepgini {dataset_name}/{pool_type}/'
                        f'{args.error_sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                        f'(n={n_pool}, errors={n_errors})',
                    )
                    row = evaluate_deepgini_pool(
                        pool,
                        dataset_name=dataset_name,
                        pool_type=pool_type,
                        error_ratio=error_ratio,
                        seed=seed,
                        sampling_mode=args.error_sampling_mode,
                        model=model,
                        cache_dir=cache_dir,
                        budget_ratios=RQ1_BUDGET_RATIOS,
                        batch_size=int(args.batch_size),
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
    print('\n=== RQ1 DeepGini baseline TRC summary (mean over seeds) ===')
    print(table_text)

    mode_tag = args.error_sampling_mode
    runs_path = out_dir / f'rq1_deepgini_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_deepgini_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_deepgini_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
