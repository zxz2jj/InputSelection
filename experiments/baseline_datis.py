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
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

tf.get_logger().setLevel('ERROR')
logging.getLogger().setLevel(logging.WARNING)

from risk_scoring import _hidden_to_flat_batch, _probs_from_model_output_tensor
from training_models.load_data import load_cifar10, load_fmnist, load_svhn

RQ1_BUDGET_RATIOS = [0.01, 0.03, 0.05, 0.10]
RQ1_SEEDS = [0, 1, 2, 3, 4]
RQ1_POOL_TYPES = ['adversarial', 'transformation']
RQ1_ERROR_RATIOS = [0.10, 0.20]
RQ1_DATASETS = ['fmnist', 'cifar10', 'svhn']
ERROR_SAMPLING_MODES = ('random', 'high_conf')

DATIS_K = 100
DATIS_T = 0.1
DATIS_SCORE_EPS = 1e-15

# Candidate shortlist = 3x budget, same rule as this repo's greedy shortlist.
DATIS_POOL_FACTOR = 3
# Official DATIS distance weight paired with pool_factor=3.
DATIS_DISTANCE_WEIGHT = 0.3

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        'support_layer_index': -4,
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
        'support_layer_index': -5,
    },
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
        'support_layer_index': -4,
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


def _as_label_vector(arr):
    out = np.asarray(arr)
    if out.ndim > 1:
        out = np.argmax(out, axis=-1)
    return out.reshape(-1).astype(np.int64, copy=False)


def _budget_k(n_pool, budget_ratio):
    rr = float(budget_ratio)
    if rr <= 0 or rr > 1:
        raise ValueError(f'budget ratio must be in (0, 1], got {rr}')
    k = int(np.ceil(rr * int(n_pool)))
    return max(1, min(k, int(n_pool)))


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


def train_support_cache_name(support_layer_index):
    return f'datis_train_support_layer{int(support_layer_index)}.npz'


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


def extract_support_features(data, model, support_layer_index, batch_size=64):
    """Last-hidden support vectors used by DATIS (flattened like this repo)."""
    layer_index = int(support_layer_index)
    try:
        hidden_layer = model.layers[layer_index]
    except IndexError as exc:
        raise ValueError(
            f'support_layer_index={layer_index!r} out of range '
            f'(model has {len(model.layers)} layers)',
        ) from exc

    forward = tf.keras.Model(
        inputs=model.input,
        outputs=hidden_layer.output,
        name='datis_support_forward',
    )
    chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc=f'datis support layer {layer_index}'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        hid = forward(batch, training=False)
        flat = _hidden_to_flat_batch(hid, hidden_layer).numpy().astype(np.float32, copy=False)
        chunks.append(flat)
    out = np.concatenate(chunks, axis=0)
    if out.shape[0] != n:
        raise ValueError('DATIS support feature length mismatch with inputs')
    return out


def extract_support_and_pred_classes(data, model, support_layer_index, batch_size=64):
    """Joint forward: support features + predicted class from model softmax."""
    layer_index = int(support_layer_index)
    try:
        hidden_layer = model.layers[layer_index]
    except IndexError as exc:
        raise ValueError(
            f'support_layer_index={layer_index!r} out of range '
            f'(model has {len(model.layers)} layers)',
        ) from exc

    forward = tf.keras.Model(
        inputs=model.input,
        outputs=[hidden_layer.output, model.output],
        name='datis_support_pred_forward',
    )
    hid_chunks = []
    pred_chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc=f'datis support+pred layer {layer_index}'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        hid, raw_out = forward(batch, training=False)
        flat = _hidden_to_flat_batch(hid, hidden_layer).numpy().astype(np.float32, copy=False)
        probs = _probs_from_model_output_tensor(raw_out).numpy()
        pred = np.argmax(probs, axis=-1).astype(np.int64, copy=False).reshape(-1)
        hid_chunks.append(flat)
        pred_chunks.append(pred)
    support = np.concatenate(hid_chunks, axis=0)
    pred_classes = np.concatenate(pred_chunks, axis=0)
    if support.shape[0] != n or pred_classes.shape[0] != n:
        raise ValueError('DATIS support/pred length mismatch with inputs')
    return support, pred_classes


def compute_datis_scores(
    pred_classes,
    train_support,
    y_train,
    test_support,
    num_classes,
    k=DATIS_K,
    T=DATIS_T,
):
    """
    DATIS stage-1 scores (higher = more likely to select).

    `pred_classes` may be a 1-D predicted-class vector or a 2-D softmax matrix
    (only argmax is used). Neighbor class probabilities come from training
    labels `y_train` and L2-normalized support features.
    """
    pred = _as_label_vector(pred_classes)
    y_tr = _as_label_vector(y_train)
    train_feat = np.asarray(train_support, dtype=np.float32)
    test_feat = np.asarray(test_support, dtype=np.float32)
    if train_feat.ndim != 2 or test_feat.ndim != 2:
        raise ValueError('train_support and test_support must be 2-D feature matrices')
    if train_feat.shape[0] != y_tr.shape[0]:
        raise ValueError('train_support / y_train length mismatch')
    if test_feat.shape[0] != pred.shape[0]:
        raise ValueError('test_support / pred_classes length mismatch')
    if train_feat.shape[1] != test_feat.shape[1]:
        raise ValueError('train/test support feature dim mismatch')

    n_train = int(train_feat.shape[0])
    n_test = int(test_feat.shape[0])
    n_classes = int(num_classes)
    if n_classes <= 1:
        raise ValueError(f'num_classes must be >= 2, got {n_classes}')
    if np.any(pred < 0) or np.any(pred >= n_classes):
        raise ValueError('pred_classes contains labels outside [0, num_classes)')
    if np.any(y_tr < 0) or np.any(y_tr >= n_classes):
        raise ValueError('y_train contains labels outside [0, num_classes)')

    k_use = min(int(k), n_train)
    if k_use < 1:
        raise ValueError('DATIS k must be >= 1')

    normalizer = Normalizer(norm='l2')
    train_norm = normalizer.transform(train_feat)
    test_norm = normalizer.transform(test_feat)

    nn = NearestNeighbors(n_neighbors=k_use, algorithm='auto', metric='euclidean')
    nn.fit(train_norm)
    neighbor_idx = nn.kneighbors(test_norm, n_neighbors=k_use, return_distance=False)

    neighbor_feat = train_norm[neighbor_idx]
    sq_dist = np.sum((test_norm[:, None, :] - neighbor_feat) ** 2, axis=-1)
    log_w = -sq_dist / float(T)
    log_w = log_w - np.max(log_w, axis=1, keepdims=True)
    weights = np.exp(log_w)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), DATIS_SCORE_EPS)

    neighbor_labels = y_tr[neighbor_idx]
    prob_knn = np.zeros((n_test, n_classes), dtype=np.float64)
    row_idx = np.arange(n_test, dtype=np.int64)[:, None]
    np.add.at(prob_knn, (row_idx, neighbor_labels), weights)

    knn_max = np.argmax(prob_knn, axis=1)
    tmp = prob_knn.copy()
    tmp[np.arange(n_test), knn_max] = -1.0
    knn_second = np.argmax(tmp, axis=1)

    agree = knn_max == pred
    numer = np.where(
        agree,
        prob_knn[np.arange(n_test), knn_second],
        prob_knn[np.arange(n_test), knn_max],
    )
    denom = prob_knn[np.arange(n_test), pred] + DATIS_SCORE_EPS
    return (numer / denom).astype(np.float64, copy=False)


def datis_selection_order(datis_scores):
    scores = np.asarray(datis_scores, dtype=np.float64).reshape(-1)
    n = scores.shape[0]
    order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
    return order.astype(np.int64, copy=False)


def datis_redundancy_elimination(budget_ratios, rank_list, test_support, n_pool=None):
    """
    DATIS stage-2: per-budget selected index arrays (not prefixes of one ranking).

    Returns dict[_budget_key(r)] -> np.ndarray of selected pool indices.
    """
    rank_list = np.asarray(rank_list, dtype=np.int64).reshape(-1)
    test = np.asarray(test_support, dtype=np.float32)
    if test.ndim != 2:
        raise ValueError('test_support must be a 2-D feature matrix')
    size = int(test.shape[0] if n_pool is None else n_pool)
    if rank_list.shape[0] != size or test.shape[0] != size:
        raise ValueError('rank_list / test_support length mismatch with n_pool')

    test_norm = Normalizer(norm='l2').transform(test)
    selected = {}

    for budget_ratio in budget_ratios:
        k = _budget_k(size, budget_ratio)
        pool_factor = int(DATIS_POOL_FACTOR)
        dist_weight = float(DATIS_DISTANCE_WEIGHT)
        tmp_k = int(np.ceil(float(budget_ratio) * size * pool_factor))
        tmp_k = max(k, min(tmp_k, size))
        cand_idx = rank_list[:tmp_k]
        cand_feat = test_norm[cand_idx]
        kn = min(k, 100, tmp_k)
        if kn < 1:
            kn = 1

        nn = NearestNeighbors(n_neighbors=kn, algorithm='auto', metric='euclidean')
        nn.fit(cand_feat)
        neigh_dist, _ = nn.kneighbors(cand_feat, n_neighbors=kn)
        mean_dist = np.mean(neigh_dist, axis=1)

        step1_weights = np.arange(tmp_k, 0, -1, dtype=np.float64)
        dist_order = np.argsort(-mean_dist, kind='mergesort')
        distance_weights = np.empty(tmp_k, dtype=np.float64)
        distance_weights[dist_order] = np.arange(tmp_k, 0, -1, dtype=np.float64)

        fused = (1.0 - dist_weight) * step1_weights + dist_weight * distance_weights
        keep = np.argsort(-fused, kind='mergesort')[:k]
        selected[_budget_key(budget_ratio)] = cand_idx[keep].astype(np.int64, copy=False)

    return selected


def selected_by_budget_from_order(rank_list, budget_ratios, n_pool):
    order = np.asarray(rank_list, dtype=np.int64).reshape(-1)
    if order.shape[0] != int(n_pool):
        raise ValueError('rank_list length mismatch with n_pool')
    out = {}
    for budget_ratio in budget_ratios:
        k = _budget_k(n_pool, budget_ratio)
        out[_budget_key(budget_ratio)] = order[:k].copy()
    return out


def build_or_load_train_support(
    *,
    cache_dir,
    support_layer_index,
    x_train,
    y_train,
    model,
    batch_size=64,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / train_support_cache_name(support_layer_index)
    y_tr = _as_label_vector(y_train)
    n_train = int(len(x_train))

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        support = np.asarray(z['support'], dtype=np.float32)
        cached_y = np.asarray(z['y_train'], dtype=np.int64).reshape(-1)
        cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
        if (
            support.shape[0] == n_train
            and cached_y.shape[0] == n_train
            and cached_layer == int(support_layer_index)
        ):
            print(f'Loaded DATIS train support from {cache_path}')
            return support, cached_y
        print(f'DATIS train support cache mismatch, recomputing -> {cache_path}')

    support = extract_support_features(
        x_train, model, support_layer_index, batch_size=batch_size,
    )
    np.savez_compressed(
        cache_path,
        support=support,
        y_train=y_tr,
        layer_index=np.int32(support_layer_index),
    )
    print(f'Saved DATIS train support to {cache_path}')
    return support, y_tr


def build_or_load_datis_pool_scores(
    *,
    cache_dir,
    cache_name,
    data,
    model,
    train_support,
    y_train,
    support_layer_index,
    num_classes,
    batch_size=64,
    force_recompute=False,
    k=DATIS_K,
    T=DATIS_T,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n = int(len(data))

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('support', 'pred_classes', 'datis_score', 'layer_index', 'k', 'T')
        missing = [key for key in required if key not in z.files]
        if missing:
            print(f'DATIS pool score cache missing keys {missing}, recomputing -> {cache_path}')
        else:
            support = np.asarray(z['support'], dtype=np.float32)
            pred_classes = np.asarray(z['pred_classes'], dtype=np.int64).reshape(-1)
            scores = np.asarray(z['datis_score'], dtype=np.float64).reshape(-1)
            cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
            cached_k = int(np.asarray(z['k']).reshape(-1)[0])
            cached_t = float(np.asarray(z['T']).reshape(-1)[0])
            reasons = []
            if support.shape[0] != n:
                reasons.append(f'n_support={support.shape[0]} vs n={n}')
            if pred_classes.shape[0] != n:
                reasons.append(f'n_pred={pred_classes.shape[0]} vs n={n}')
            if scores.shape[0] != n:
                reasons.append(f'n_score={scores.shape[0]} vs n={n}')
            if cached_layer != int(support_layer_index):
                reasons.append(f'layer={cached_layer} vs {int(support_layer_index)}')
            if cached_k != int(k):
                reasons.append(f'k={cached_k} vs {int(k)}')
            if not np.isclose(cached_t, float(T), rtol=0.0, atol=1e-5):
                reasons.append(f'T={cached_t} vs {float(T)}')
            if not reasons:
                print(f'Loaded DATIS pool scores from {cache_path}')
                return support, pred_classes, scores
            print(
                f'DATIS pool score cache mismatch ({"; ".join(reasons)}), '
                f'recomputing -> {cache_path}',
            )

    test_support, pred_classes = extract_support_and_pred_classes(
        data, model, support_layer_index, batch_size=batch_size,
    )
    scores = compute_datis_scores(
        pred_classes,
        train_support,
        y_train,
        test_support,
        num_classes=num_classes,
        k=k,
        T=T,
    )
    np.savez_compressed(
        cache_path,
        support=test_support,
        pred_classes=pred_classes,
        datis_score=scores.astype(np.float32, copy=False),
        layer_index=np.int32(support_layer_index),
        k=np.int32(k),
        T=np.float64(T),
    )
    print(f'Saved DATIS pool scores to {cache_path}')
    return test_support, pred_classes, scores


def _error_mask_from_storage(storage_rows):
    n = len(storage_rows)
    err = np.zeros(n, dtype=bool)
    for row in storage_rows:
        err[int(row['idx'])] = bool(row['is_wrongly_predicted'])
    return err


def compute_selection_curves(storage_rows, selection_order, budget_ratios):
    """Prefix TRC, same definition as selection_method.compute_greedy_selection_curves."""
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

    return {
        'total_errors': total_errors,
        'trc': out_trc,
    }


def compute_selection_curves_by_sets(storage_rows, selected_by_budget, budget_ratios):
    """TRC on an independent selected set per budget (DATIS stage-2)."""
    err = _error_mask_from_storage(storage_rows)
    n_pool = len(storage_rows)
    total_errors = int(np.sum(err))

    out_trc = []
    for r in budget_ratios:
        k = _budget_k(n_pool, r)
        chosen = np.asarray(selected_by_budget[_budget_key(r)], dtype=np.int64).reshape(-1)
        discovered = int(np.sum(err[chosen]))
        denom = min(k, total_errors)
        trc = (discovered / denom) if denom > 0 else np.nan
        out_trc.append(trc)

    return {
        'total_errors': total_errors,
        'trc': out_trc,
    }


def _jsonable_selected_by_budget(selected_by_budget):
    out = {}
    for key, idx in selected_by_budget.items():
        out[_budget_key(key)] = [int(i) for i in np.asarray(idx).reshape(-1)]
    return out


def evaluate_datis_pool(
    pool,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    seed,
    sampling_mode,
    model,
    x_train,
    y_train,
    support_layer_index,
    cache_dir,
    budget_ratios,
    apply_redundancy=True,
    batch_size=64,
    force_recompute=False,
    k=DATIS_K,
    T=DATIS_T,
    return_selection=False,
):
    """
    RQ entry: DATIS TRC on one sampled pool.

    apply_redundancy=False -> stage-1 ranking prefixes (RQ1 without diversity).
    apply_redundancy=True  -> official per-budget redundancy elimination (RQ2/RQ3).
    """
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)
    num_classes = int(model.output_shape[-1])

    train_support, train_labels = build_or_load_train_support(
        cache_dir=cache_dir,
        support_layer_index=support_layer_index,
        x_train=x_train,
        y_train=y_train,
        model=model,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    test_support, _pred_classes, datis_scores = build_or_load_datis_pool_scores(
        cache_dir=cache_dir,
        cache_name=f'datis_{tag}.npz',
        data=pool['data'],
        model=model,
        train_support=train_support,
        y_train=train_labels,
        support_layer_index=support_layer_index,
        num_classes=num_classes,
        batch_size=batch_size,
        force_recompute=force_recompute,
        k=k,
        T=T,
    )
    order = datis_selection_order(datis_scores)

    if apply_redundancy:
        selected_by_budget = datis_redundancy_elimination(
            budget_ratios, order, test_support, n_pool=n_pool,
        )
        curve = compute_selection_curves_by_sets(
            storage, selected_by_budget, budget_ratios=budget_ratios,
        )
        method_name = 'datis'
    else:
        selected_by_budget = selected_by_budget_from_order(order, budget_ratios, n_pool)
        curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
        method_name = 'datis_uncertainty'

    ratio_to_trc = _trc_by_budget_dict(budget_ratios, curve)
    row = {
        'method': method_name,
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_errors': int(curve['total_errors']),
        'apply_redundancy': bool(apply_redundancy),
        'support_layer_index': int(support_layer_index),
        'k': int(k),
        'T': float(T),
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
        description='DATIS baseline TRC evaluation on sampled pools.',
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
    parser.add_argument('--k', type=int, default=DATIS_K)
    parser.add_argument('--temperature', type=float, default=DATIS_T)
    parser.add_argument(
        '--apply-redundancy',
        action='store_true',
        help='use official stage-2 redundancy elimination (per-budget sets)',
    )
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def _drop_heavy_fields(run_row):
    """Keep JSON summaries small; RQ callers still get full dicts from evaluate_datis_pool."""
    skip = {'selection_order', 'selected_by_budget', 'pred_classes'}
    return {key: value for key, value in run_row.items() if key not in skip}


def main():
    args = parse_args()
    configure_tensorflow_gpu()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_redundancy = bool(args.apply_redundancy)
    method_tag = 'datis' if apply_redundancy else 'datis_uncertainty'

    run_rows = []
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIG[dataset_name]
        if not Path(cfg['model_path']).is_file():
            raise FileNotFoundError(f'missing model: {cfg["model_path"]}')

        print(f'=== dataset: {dataset_name} ===')
        model = tf.keras.models.load_model(cfg['model_path'], compile=False)
        x_train, y_train, _, _ = cfg['loader']()
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
                        f'Evaluating {method_tag} {dataset_name}/{pool_type}/'
                        f'{args.error_sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                        f'(n={n_pool}, errors={n_errors}, redundancy={apply_redundancy})',
                    )
                    row = evaluate_datis_pool(
                        pool,
                        dataset_name=dataset_name,
                        pool_type=pool_type,
                        error_ratio=error_ratio,
                        seed=seed,
                        sampling_mode=args.error_sampling_mode,
                        model=model,
                        x_train=x_train,
                        y_train=y_train,
                        support_layer_index=cfg['support_layer_index'],
                        cache_dir=cache_dir,
                        budget_ratios=RQ1_BUDGET_RATIOS,
                        apply_redundancy=apply_redundancy,
                        batch_size=int(args.batch_size),
                        force_recompute=args.force_recompute,
                        k=int(args.k),
                        T=float(args.temperature),
                    )
                    run_rows.append(_drop_heavy_fields(row))
                    trc_str = ', '.join(
                        f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                        for b in RQ1_BUDGET_RATIOS
                    )
                    print(f'  TRC: {trc_str}')

    summary_rows = aggregate_results(run_rows)
    table_text = format_results_table(summary_rows, args.seeds)
    print(f'\n=== DATIS baseline TRC summary ({method_tag}, mean over seeds) ===')
    print(table_text)

    mode_tag = args.error_sampling_mode
    runs_path = out_dir / f'rq1_{method_tag}_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_{method_tag}_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_{method_tag}_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
