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
from tqdm import tqdm

tf.get_logger().setLevel('ERROR')
logging.getLogger().setLevel(logging.WARNING)

from risk_scoring import _probs_from_model_output_tensor
from training_models.load_data import load_cifar10, load_fmnist, load_svhn

RQ1_BUDGET_RATIOS = [0.01, 0.03, 0.05, 0.10]
RQ1_SEEDS = [0, 1, 2, 3, 4]
RQ1_POOL_TYPES = ['adversarial', 'transformation']
RQ1_ERROR_RATIOS = [0.10, 0.20]
RQ1_DATASETS = ['fmnist', 'cifar10', 'svhn']
ERROR_SAMPLING_MODES = ('random', 'high_conf')

MC_METHOD_NAME = 'mc'
# Ma et al. TOSEM 2021 used k=50 as cost/stability trade-off.
MC_N_PASSES = 50

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


def _budget_k(n_pool, budget_ratio):
    rr = float(budget_ratio)
    if rr <= 0 or rr > 1:
        raise ValueError(f'budget ratio must be in (0, 1], got {rr}')
    k = int(np.ceil(rr * int(n_pool)))
    return max(1, min(k, int(n_pool)))


def pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode='random'):
    ratio_pct = int(round(float(error_ratio) * 100))
    return f'rq1_{dataset_name}_{pool_type}_r{ratio_pct:02d}_s{int(seed)}_{sampling_mode}'


def mc_scores_cache_name(tag, n_passes):
    return f'mc_{tag}_n{int(n_passes)}_varratio.npz'


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


def selected_by_budget_from_order(rank_list, budget_ratios, n_pool):
    order = np.asarray(rank_list, dtype=np.int64).reshape(-1)
    if order.shape[0] != int(n_pool):
        raise ValueError('rank_list length mismatch with n_pool')
    out = {}
    for budget_ratio in budget_ratios:
        k = _budget_k(n_pool, budget_ratio)
        out[_budget_key(budget_ratio)] = order[:k].copy()
    return out


def _error_mask_from_storage(storage_rows):
    n = len(storage_rows)
    err = np.zeros(n, dtype=bool)
    for row in storage_rows:
        err[int(row['idx'])] = bool(row['is_wrongly_predicted'])
    return err


def compute_selection_curves(storage_rows, selection_order, budget_ratios):
    err = _error_mask_from_storage(storage_rows)
    n_pool = len(storage_rows)
    total_errors = int(np.sum(err))
    chosen = np.asarray(selection_order, dtype=np.int64).reshape(-1)

    out_trc = []
    for r in budget_ratios:
        k = _budget_k(n_pool, r)
        prefix = chosen[:k]
        discovered = int(np.sum(err[prefix]))
        denom = min(k, total_errors)
        trc = (discovered / denom) if denom > 0 else np.nan
        out_trc.append(trc)
    return {'total_errors': total_errors, 'trc': out_trc}


def _jsonable_selected_by_budget(selected_by_budget):
    out = {}
    for key, idx in selected_by_budget.items():
        out[_budget_key(key)] = [int(i) for i in np.asarray(idx).reshape(-1)]
    return out


def _is_dropout_layer(layer):
    return isinstance(layer, tf.keras.layers.Dropout)


def _iter_layers(layer):
    yield layer
    nested = getattr(layer, 'layers', None)
    if nested:
        for sub in nested:
            yield from _iter_layers(sub)


def _patch_dropout_always_on(model):
    """Keep Dropout stochastic while other layers (e.g. BN) stay in inference mode."""
    n_drop = 0
    for layer in _iter_layers(model):
        if not _is_dropout_layer(layer):
            continue
        original_call = layer.call

        def call_always_drop(inputs, training=None, _orig=original_call, **kwargs):
            return _orig(inputs, training=True, **kwargs)

        layer.call = call_always_drop
        n_drop += 1
    if n_drop == 0:
        raise ValueError('model has no Dropout layers; MC dropout cannot run')
    return n_drop


def build_mc_dropout_model(model):
    """Clone so the original RQ model is not mutated."""
    mc_model = tf.keras.models.clone_model(model)
    mc_model.set_weights(model.get_weights())
    n_drop = _patch_dropout_always_on(mc_model)
    print(f'  MC dropout clone: {n_drop} Dropout layer(s) enabled at inference')
    return mc_model


def variation_ratio_from_votes(votes, n_classes=None):
    """MC(x) = 1 - count(mode class) / N."""
    votes = np.asarray(votes, dtype=np.int64)
    if votes.ndim != 2:
        raise ValueError('votes must be (n_pool, n_passes)')
    n_pool, n_passes = votes.shape
    if n_passes == 0:
        raise ValueError('n_passes must be > 0')
    if n_classes is None:
        n_classes = int(votes.max()) + 1 if n_pool > 0 else 1
    n_classes = max(int(n_classes), int(votes.max()) + 1 if n_pool > 0 else 1)
    offsets = votes + (np.arange(n_pool, dtype=np.int64)[:, None] * n_classes)
    counts = np.bincount(offsets.ravel(), minlength=n_pool * n_classes)
    counts = counts.reshape(n_pool, n_classes)
    return (1.0 - counts.max(axis=1) / float(n_passes)).astype(np.float64, copy=False)


def mc_selection_order(scores):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = scores.shape[0]
    order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
    return order.astype(np.int64, copy=False)


def extract_mc_votes(data, mc_model, n_passes=MC_N_PASSES, batch_size=64):
    """
    N full-pool batched forwards (not per-sample loops).

    Cost is ~N times one DeepGini softmax pass over the pool.
    """
    n = int(len(data))
    n_passes = int(n_passes)
    bs = int(batch_size)
    if n_passes < 1:
        raise ValueError(f'n_passes must be >= 1, got {n_passes}')
    votes = np.empty((n, n_passes), dtype=np.int32)
    for t in tqdm(range(n_passes), desc=f'mc dropout passes n={n_passes}'):
        pred_chunks = []
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch = tf.convert_to_tensor(data[start:end])
            raw = mc_model(batch, training=False)
            probs = _probs_from_model_output_tensor(raw).numpy()
            pred = np.argmax(probs, axis=-1).astype(np.int32, copy=False).reshape(-1)
            pred_chunks.append(pred)
        row = np.concatenate(pred_chunks, axis=0)
        if row.shape[0] != n:
            raise ValueError('MC vote length mismatch with pool')
        votes[:, t] = row
    return votes


def build_or_load_mc_scores(
    *,
    cache_dir,
    cache_name,
    data,
    model,
    n_passes=MC_N_PASSES,
    batch_size=64,
    rng_seed=None,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n = int(len(data))
    n_passes = int(n_passes)

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('mc_score', 'sort_ix', 'n_passes', 'n_pool')
        missing = [key for key in required if key not in z.files]
        reasons = []
        if missing:
            reasons.append(f'missing keys {missing}')
        else:
            scores = np.asarray(z['mc_score'], dtype=np.float64).reshape(-1)
            order = np.asarray(z['sort_ix'], dtype=np.int64).reshape(-1)
            cached_n_passes = int(np.asarray(z['n_passes']).reshape(-1)[0])
            cached_n = int(np.asarray(z['n_pool']).reshape(-1)[0])
            if scores.shape[0] != n:
                reasons.append(f'n_score={scores.shape[0]} vs n={n}')
            if order.shape[0] != n:
                reasons.append(f'rank len={order.shape[0]} vs n={n}')
            if cached_n != n:
                reasons.append(f'n_pool={cached_n} vs {n}')
            if cached_n_passes != n_passes:
                reasons.append(f'n_passes={cached_n_passes} vs {n_passes}')
            if not reasons:
                print(f'Loaded MC scores from {cache_path}')
                return scores, order
        print(
            f'MC score cache mismatch ({"; ".join(reasons)}), '
            f'recomputing -> {cache_path}',
        )

    if rng_seed is not None:
        tf.random.set_seed(int(rng_seed) + 50_000)
    print(
        f'  MC variation-ratio n_pool={n} n_passes={n_passes} '
        f'(~{n_passes}x one softmax pass over the pool)',
    )
    mc_model = build_mc_dropout_model(model)
    votes = extract_mc_votes(
        data, mc_model, n_passes=n_passes, batch_size=batch_size,
    )
    n_classes = int(mc_model.output_shape[-1])
    scores = variation_ratio_from_votes(votes, n_classes=n_classes)
    order = mc_selection_order(scores)
    np.savez_compressed(
        cache_path,
        mc_score=scores.astype(np.float32, copy=False),
        sort_ix=np.asarray(order, dtype=np.int64),
        n_passes=np.int32(n_passes),
        n_pool=np.int32(n),
    )
    print(f'Saved MC scores to {cache_path}')
    return scores, order


def evaluate_mc_pool(
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
    n_passes=MC_N_PASSES,
    batch_size=64,
    force_recompute=False,
    return_selection=False,
):
    """
    RQ entry: MC dropout variation-ratio ranking, prefix TRC.

    Higher score = more disagreement across stochastic forwards.
    """
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)
    n_passes = int(n_passes)

    scores, order = build_or_load_mc_scores(
        cache_dir=cache_dir,
        cache_name=mc_scores_cache_name(tag, n_passes),
        data=pool['data'],
        model=model,
        n_passes=n_passes,
        batch_size=batch_size,
        rng_seed=int(seed),
        force_recompute=force_recompute,
    )
    selected_by_budget = selected_by_budget_from_order(order, budget_ratios, n_pool)
    curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
    ratio_to_trc = _trc_by_budget_dict(budget_ratios, curve)
    row = {
        'method': MC_METHOD_NAME,
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_errors': int(curve['total_errors']),
        'n_passes': int(n_passes),
        'uncertainty': 'variation_ratio',
        'trc_by_budget': ratio_to_trc,
    }
    if return_selection:
        row['selection_order'] = [int(i) for i in np.asarray(order).reshape(-1)]
        row['selected_by_budget'] = _jsonable_selected_by_budget(selected_by_budget)
        row['mc_score'] = [float(v) for v in np.asarray(scores, dtype=np.float64).reshape(-1)]
    return row


def aggregate_results(run_rows):
    groups = {}
    for row in run_rows:
        key = (row['dataset'], row['pool_type'], row['sampling_mode'], row['error_ratio'], row['method'])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (dataset, pool_type, sampling_mode, error_ratio, method), rows in sorted(groups.items()):
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
        description='MC dropout baseline TRC (variation ratio) on sampled pools.',
    )
    parser.add_argument('--datasets', nargs='+', default=RQ1_DATASETS, choices=sorted(RQ1_DATASETS))
    parser.add_argument('--pool-types', nargs='+', default=RQ1_POOL_TYPES, choices=RQ1_POOL_TYPES)
    parser.add_argument('--error-ratios', nargs='+', type=float, default=RQ1_ERROR_RATIOS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument(
        '--error-sampling-modes',
        nargs='+',
        default=['random'],
        choices=list(ERROR_SAMPLING_MODES),
    )
    parser.add_argument('--sampled-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--output-dir', default=str(_EXP_DIR / 'results' / 'rq1'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-passes', type=int, default=MC_N_PASSES)
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def _drop_heavy_fields(run_row):
    skip = {'selection_order', 'selected_by_budget', 'mc_score'}
    return {key: value for key, value in run_row.items() if key not in skip}


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
                        n_pool = len(pool['clean_labels'])
                        n_errors = int(np.sum(pool['is_error']))
                        print(
                            f'Evaluating mc {dataset_name}/{pool_type}/'
                            f'{sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                            f'(n={n_pool}, errors={n_errors}, n_passes={int(args.n_passes)})',
                        )
                        row = evaluate_mc_pool(
                            pool,
                            dataset_name=dataset_name,
                            pool_type=pool_type,
                            error_ratio=error_ratio,
                            seed=seed,
                            sampling_mode=sampling_mode,
                            model=model,
                            cache_dir=cache_dir,
                            budget_ratios=RQ1_BUDGET_RATIOS,
                            n_passes=int(args.n_passes),
                            batch_size=int(args.batch_size),
                            force_recompute=args.force_recompute,
                        )
                        run_rows.append(_drop_heavy_fields(row))
                        trc_str = ', '.join(
                            f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                            for b in RQ1_BUDGET_RATIOS
                        )
                        print(f'  {row["method"]} TRC: {trc_str}')

    summary_rows = aggregate_results(run_rows)
    table_text = format_results_table(summary_rows, args.seeds)
    print('\n=== MC dropout baseline TRC summary (variation ratio) ===')
    print(table_text)

    mode_tag = '_'.join(args.error_sampling_modes)
    runs_path = out_dir / f'rq1_mc_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_mc_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_mc_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
