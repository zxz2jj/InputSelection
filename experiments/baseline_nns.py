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
from sklearn.neighbors import NearestNeighbors
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

NNS_METHOD_NAME = 'nns'
# ISSTA 2023 NNS paper defaults (NNS_M + DeepGini).
NNS_K = 10
NNS_LAMBDA = 0.5

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        # conv2 + MaxPool: 12x12x20 = 2880 (same last encoder as SETS/NSS)
        'nns_feature_layer_index': 2,
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
        # last encoder ReLU (SETS/NSS -11): 2x2x512 = 2048
        'nns_feature_layer_index': -11,
    },
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
        # last residual ReLU before AvgPool (SETS/NSS -6): 4x4x512 = 8192
        'nns_feature_layer_index': -6,
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


def _float_tag(value):
    text = f'{float(value):.6f}'.rstrip('0').rstrip('.')
    return text.replace('-', 'm').replace('.', 'p')


def pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode='random'):
    ratio_pct = int(round(float(error_ratio) * 100))
    return f'rq1_{dataset_name}_{pool_type}_r{ratio_pct:02d}_s{int(seed)}_{sampling_mode}'


def nns_selection_cache_name(tag, k, lam, layer_index):
    return (
        f'nns_{tag}_sel_k{int(k)}_l{_float_tag(lam)}'
        f'_gini_layer{int(layer_index)}.npz'
    )


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


def extract_nns_probs(data, model, batch_size=64):
    chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc='nns softmax'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        raw = model(batch, training=False)
        probs = _probs_from_model_output_tensor(raw).numpy().astype(np.float32, copy=False)
        chunks.append(probs)
    out = np.concatenate(chunks, axis=0)
    if out.shape[0] != n:
        raise ValueError('NNS softmax length mismatch with pool')
    return out


def _nns_spatial_flatten(hid):
    """Keep spatial bins: reshape to (batch, H*W*C). Do not GAP."""
    hid = tf.cast(hid, tf.float32)
    b = tf.shape(hid)[0]
    return tf.reshape(hid, [b, -1])


def extract_nns_hidden(data, model, nns_feature_layer_index, batch_size=64):
    layer_index = int(nns_feature_layer_index)
    try:
        hidden_layer = model.layers[layer_index]
    except IndexError as exc:
        raise ValueError(
            f'nns_feature_layer_index={layer_index!r} out of range '
            f'(model has {len(model.layers)} layers)',
        ) from exc

    forward = tf.keras.Model(
        inputs=model.input,
        outputs=hidden_layer.output,
        name='nns_hidden_forward',
    )
    hid_chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc=f'nns hidden layer {layer_index}'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        hid = forward(batch, training=False)
        flat = _nns_spatial_flatten(hid).numpy().astype(np.float32, copy=False)
        hid_chunks.append(flat)
    features = np.concatenate(hid_chunks, axis=0)
    if features.shape[0] != n:
        raise ValueError('NNS hidden length mismatch with pool')
    return features


def knn_neighbor_indices(features, k=NNS_K):
    """k cosine nearest neighbors in the pool, excluding self."""
    feat = np.asarray(features, dtype=np.float64)
    if feat.ndim != 2:
        raise ValueError('features must be (n, dim)')
    n = int(feat.shape[0])
    k = int(k)
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)
    if k < 1:
        raise ValueError(f'NNS k must be >= 1, got {k}')
    k_use = min(k, max(n - 1, 0))
    if k_use == 0:
        return np.zeros((n, 0), dtype=np.int64)

    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / np.maximum(norms, 1e-12)
    n_query = min(k_use + 1, n)
    nn = NearestNeighbors(n_neighbors=n_query, metric='euclidean', algorithm='brute')
    nn.fit(feat)
    _, indices = nn.kneighbors(feat)
    arange = np.arange(n, dtype=np.int64)
    # Typical case: the nearest neighbor of i is i itself.
    if indices.shape[1] >= k_use + 1 and np.all(indices[:, 0] == arange):
        return indices[:, 1:1 + k_use].astype(np.int64, copy=False)

    out = np.empty((n, k_use), dtype=np.int64)
    for i in range(n):
        row = indices[i]
        others = row[row != i]
        if others.shape[0] < k_use:
            seen = set(others.tolist())
            seen.add(i)
            extra = [j for j in range(n) if j not in seen]
            others = np.concatenate(
                [others, np.asarray(extra[: k_use - others.shape[0]], dtype=np.int64)],
                axis=0,
            )
        out[i] = others[:k_use]
    return out


def smooth_probs(probs, neighbor_idx, lam=NNS_LAMBDA):
    p = np.asarray(probs, dtype=np.float64)
    idx = np.asarray(neighbor_idx, dtype=np.int64)
    if p.ndim != 2:
        raise ValueError('probs must be (n, num_classes)')
    if idx.shape[0] != p.shape[0]:
        raise ValueError('neighbor_idx length mismatch with probs')
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f'lambda must be in [0, 1], got {lam}')
    if idx.shape[1] == 0:
        return p.copy()
    p_bar = p[idx].mean(axis=1)
    return lam * p + (1.0 - lam) * p_bar


def deepgini_from_probs(probs):
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError('probs must be (n, num_classes)')
    return (1.0 - np.sum(np.square(p), axis=1)).astype(np.float64, copy=False)


def nns_selection_order(gini_scores):
    scores = np.asarray(gini_scores, dtype=np.float64).reshape(-1)
    n = scores.shape[0]
    order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
    return order.astype(np.int64, copy=False)


def build_or_load_nns_probs(
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
    n = int(len(data))

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        if 'probs' in z.files:
            probs = np.asarray(z['probs'], dtype=np.float64)
            if probs.ndim == 2 and probs.shape[0] == n:
                print(f'Loaded NNS softmax from {cache_path}')
                return probs
        print(f'NNS softmax cache mismatch, recomputing -> {cache_path}')

    probs = extract_nns_probs(data, model, batch_size=batch_size)
    np.savez_compressed(cache_path, probs=np.asarray(probs, dtype=np.float32))
    print(f'Saved NNS softmax to {cache_path}')
    return np.asarray(probs, dtype=np.float64)


def build_or_load_nns_features(
    *,
    cache_dir,
    cache_name,
    data,
    model,
    nns_feature_layer_index,
    batch_size=64,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n = int(len(data))
    layer_index = int(nns_feature_layer_index)

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        if 'feature' in z.files and 'layer_index' in z.files:
            feat = np.asarray(z['feature'], dtype=np.float32)
            cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
            if feat.ndim == 2 and feat.shape[0] == n and cached_layer == layer_index:
                print(f'Loaded NNS features from {cache_path}')
                return feat
        print(f'NNS feature cache mismatch, recomputing -> {cache_path}')

    feat = extract_nns_hidden(
        data, model, layer_index, batch_size=batch_size,
    )
    np.savez_compressed(
        cache_path,
        feature=feat,
        layer_index=np.int32(layer_index),
    )
    print(f'Saved NNS features to {cache_path}')
    return feat


def build_or_load_nns_selection(
    *,
    cache_dir,
    cache_name,
    probs,
    features,
    k,
    lam,
    layer_index,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n_pool = int(probs.shape[0])

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('sort_ix', 'k', 'lam', 'layer_index', 'n_pool')
        missing = [key for key in required if key not in z.files]
        reasons = []
        if missing:
            reasons.append(f'missing keys {missing}')
        else:
            order = np.asarray(z['sort_ix'], dtype=np.int64).reshape(-1)
            cached_k = int(np.asarray(z['k']).reshape(-1)[0])
            cached_lam = float(np.asarray(z['lam'], dtype=np.float64).reshape(-1)[0])
            cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
            cached_n = int(np.asarray(z['n_pool']).reshape(-1)[0])
            if order.shape[0] != n_pool:
                reasons.append(f'rank len={order.shape[0]} vs n={n_pool}')
            if cached_n != n_pool:
                reasons.append(f'n_pool={cached_n} vs {n_pool}')
            if cached_k != int(k):
                reasons.append(f'k={cached_k} vs {int(k)}')
            if cached_layer != int(layer_index):
                reasons.append(f'layer={cached_layer} vs {int(layer_index)}')
            if not np.isclose(cached_lam, float(lam), rtol=0.0, atol=1e-12):
                reasons.append(f'lam={cached_lam} vs {float(lam)}')
            if not reasons:
                print(f'Loaded NNS selection from {cache_path}')
                return order
        print(
            f'NNS selection cache mismatch ({"; ".join(reasons)}), '
            f'recomputing -> {cache_path}',
        )

    print(
        f'  NNS kNN+DeepGini n={n_pool} k={int(k)} lam={float(lam)} '
        f'layer={int(layer_index)}',
    )
    neighbor_idx = knn_neighbor_indices(features, k=k)
    p_hat = smooth_probs(probs, neighbor_idx, lam=lam)
    scores = deepgini_from_probs(p_hat)
    order = nns_selection_order(scores)
    np.savez_compressed(
        cache_path,
        sort_ix=np.asarray(order, dtype=np.int64),
        gini_score=scores.astype(np.float32, copy=False),
        k=np.int32(k),
        lam=np.float64(lam),
        layer_index=np.int32(layer_index),
        n_pool=np.int32(n_pool),
        uncertainty=np.asarray('gini'),
    )
    print(f'Saved NNS selection to {cache_path}')
    return order


def evaluate_nns_pool(
    pool,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    seed,
    sampling_mode,
    model,
    nns_feature_layer_index,
    cache_dir,
    budget_ratios,
    k=NNS_K,
    lam=NNS_LAMBDA,
    batch_size=64,
    force_recompute=False,
    return_selection=False,
):
    """
    RQ entry: NNS_M + DeepGini (kNN softmax smoothing, then Gini ranking).

    Returns a single result row (method='nns').
    """
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)
    layer_index = int(nns_feature_layer_index)

    probs = build_or_load_nns_probs(
        cache_dir=cache_dir,
        cache_name=f'nns_{tag}_probs.npz',
        data=pool['data'],
        model=model,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    features = build_or_load_nns_features(
        cache_dir=cache_dir,
        cache_name=f'nns_{tag}_feat_layer{layer_index}.npz',
        data=pool['data'],
        model=model,
        nns_feature_layer_index=layer_index,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    order = build_or_load_nns_selection(
        cache_dir=cache_dir,
        cache_name=nns_selection_cache_name(tag, k=k, lam=lam, layer_index=layer_index),
        probs=probs,
        features=features,
        k=k,
        lam=lam,
        layer_index=layer_index,
        force_recompute=force_recompute,
    )
    selected_by_budget = selected_by_budget_from_order(order, budget_ratios, n_pool)
    curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
    ratio_to_trc = _trc_by_budget_dict(budget_ratios, curve)
    row = {
        'method': NNS_METHOD_NAME,
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_errors': int(curve['total_errors']),
        'nns_feature_layer_index': int(layer_index),
        'k': int(k),
        'lam': float(lam),
        'uncertainty': 'gini',
        'trc_by_budget': ratio_to_trc,
    }
    if return_selection:
        row['selection_order'] = [int(i) for i in np.asarray(order).reshape(-1)]
        row['selected_by_budget'] = _jsonable_selected_by_budget(selected_by_budget)
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
        description='NNS baseline TRC (kNN softmax smoothing + DeepGini) on sampled pools.',
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
    parser.add_argument('--k', type=int, default=NNS_K)
    parser.add_argument('--lam', type=float, default=NNS_LAMBDA)
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def _drop_heavy_fields(run_row):
    skip = {'selection_order', 'selected_by_budget'}
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
        layer_index = int(cfg['nns_feature_layer_index'])

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
                            f'Evaluating nns {dataset_name}/{pool_type}/'
                            f'{sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                            f'(n={n_pool}, errors={n_errors}, k={int(args.k)}, lam={float(args.lam)})',
                        )
                        row = evaluate_nns_pool(
                            pool,
                            dataset_name=dataset_name,
                            pool_type=pool_type,
                            error_ratio=error_ratio,
                            seed=seed,
                            sampling_mode=sampling_mode,
                            model=model,
                            nns_feature_layer_index=layer_index,
                            cache_dir=cache_dir,
                            budget_ratios=RQ1_BUDGET_RATIOS,
                            k=int(args.k),
                            lam=float(args.lam),
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
    print('\n=== NNS baseline TRC summary (NNS_M + DeepGini) ===')
    print(table_text)

    mode_tag = '_'.join(args.error_sampling_modes)
    runs_path = out_dir / f'rq1_nns_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_nns_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_nns_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
