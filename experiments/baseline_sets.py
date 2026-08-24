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

# Official SETS RQ2/3 defaults.
SETS_A = 3
SETS_UNCERTAINTY = 'maxp'
SETS_DIVERSITY = 'gd'
SETS_UNCERTAINTY_CHOICES = ('maxp', 'gini')
SETS_DIVERSITY_CHOICES = ('gd', 'std')

DATASET_CONFIG = {
    # GD uses a full spatial flatten (no GAP). dim must exceed max budget k.
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        # Flatten after pool: 12x12x20 = 2880 (> k_10% ~1164)
        'sets_feature_layer_index': -11,
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
        # Last conv ReLU before final MaxPool: 2x2x512 = 2048 (> k_10% ~1128)
        'sets_feature_layer_index': -11,
    },
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
        # Last residual ReLU before AveragePooling: 4x4x512 = 8192 (> k_10% ~3128)
        'sets_feature_layer_index': -6,
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


def minmax_normalize_feature_matrix(feature_rows, eps=1e-12):
    v = np.asarray(feature_rows, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError('feature_rows must be 2-D')
    vmin = np.min(v, axis=0)
    vmax = np.max(v, axis=0)
    span = vmax - vmin
    constant = span <= float(eps)
    span = np.where(constant, 1.0, span)
    out = (v - vmin) / span
    out[:, constant] = 0.0
    return out


def maxp_score(probs):
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError('probs must be (n, num_classes)')
    return (1.0 - np.max(p, axis=1)).astype(np.float64, copy=False)


def gini_score(probs):
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError('probs must be (n, num_classes)')
    return (1.0 - np.sum(np.square(p), axis=1)).astype(np.float64, copy=False)


def uncertainty_scores_from_probs(probs, uncertainty=SETS_UNCERTAINTY):
    name = str(uncertainty).lower()
    if name == 'maxp':
        return maxp_score(probs)
    if name == 'gini':
        return gini_score(probs)
    raise ValueError(f'unknown SETS uncertainty={uncertainty!r}')


def extract_sets_probs(data, model, batch_size=64):
    chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc='sets softmax'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        raw = model(batch, training=False)
        probs = _probs_from_model_output_tensor(raw).numpy().astype(np.float32, copy=False)
        chunks.append(probs)
    out = np.concatenate(chunks, axis=0)
    if out.shape[0] != n:
        raise ValueError('SETS softmax length mismatch with pool')
    return out


def _sets_spatial_flatten(hid):
    """Keep spatial bins: reshape to (batch, H*W*C). Do not GAP."""
    hid = tf.cast(hid, tf.float32)
    b = tf.shape(hid)[0]
    return tf.reshape(hid, [b, -1])


def extract_sets_hidden(data, model, sets_feature_layer_index, batch_size=64):
    layer_index = int(sets_feature_layer_index)
    try:
        hidden_layer = model.layers[layer_index]
    except IndexError as exc:
        raise ValueError(
            f'sets_feature_layer_index={layer_index!r} out of range '
            f'(model has {len(model.layers)} layers)',
        ) from exc

    forward = tf.keras.Model(
        inputs=model.input,
        outputs=hidden_layer.output,
        name='sets_hidden_forward',
    )
    hid_chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc=f'sets hidden layer {layer_index}'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        hid = forward(batch, training=False)
        flat = _sets_spatial_flatten(hid).numpy().astype(np.float32, copy=False)
        hid_chunks.append(flat)
    features = np.concatenate(hid_chunks, axis=0)
    if features.shape[0] != n:
        raise ValueError('SETS hidden length mismatch with pool')
    return features


def _gd_logdet(feature_rows):
    x = np.asarray(feature_rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return 0.0
    gram = np.matmul(x, x.T)
    _sign, logdet = np.linalg.slogdet(gram)
    return float(logdet)


def _std_l1(feature_rows):
    x = np.asarray(feature_rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(np.std(x, axis=0), 1))


def sets_chunked_select(
    k,
    uncertainty_scores,
    features,
    *,
    a=SETS_A,
    diversity=SETS_DIVERSITY,
):
    """
    Official SETS greedy: keep top-(a*k) by uncertainty, round-robin into k
    chunks, pick one from each chunk by u(i) * minmax(delta_diversity).
    """
    scores = np.asarray(uncertainty_scores, dtype=np.float64).reshape(-1)
    feat = np.asarray(features, dtype=np.float64)
    n = int(scores.shape[0])
    k = int(k)
    if k < 1 or k > n:
        raise ValueError(f'SETS budget k must be in [1, n], got k={k}, n={n}')
    if feat.shape[0] != n or feat.ndim != 2:
        raise ValueError('SETS features must be (n, dim)')
    div_name = str(diversity).lower()
    if div_name not in SETS_DIVERSITY_CHOICES:
        raise ValueError(f'unknown SETS diversity={diversity!r}')

    n_cand = int(np.ceil(float(a) * k))
    n_cand = max(k, min(n_cand, n))
    ranked = np.argsort(-scores, kind='mergesort')
    filtered = ranked[:n_cand]
    selected = []
    selected_feat = np.zeros((0, feat.shape[1]), dtype=np.float64)
    current_div = 0.0

    for chunk_i in range(k):
        chunk = filtered[chunk_i::k]
        if chunk.size == 0:
            continue
        deltas = np.empty(chunk.size, dtype=np.float64)
        new_divs = np.empty(chunk.size, dtype=np.float64)
        for j, idx in enumerate(chunk):
            trial = np.vstack([selected_feat, feat[int(idx)][None, :]])
            if div_name == 'gd':
                new_div = _gd_logdet(trial)
            else:
                new_div = _std_l1(trial)
            new_divs[j] = new_div
            deltas[j] = new_div - current_div
        dmin = float(np.min(deltas))
        dmax = float(np.max(deltas))
        if dmax - dmin > 0.0:
            norm_delta = (deltas - dmin) / (dmax - dmin + 0.5)
        else:
            norm_delta = np.zeros_like(deltas)
        objective = scores[chunk] * norm_delta
        best_j = int(np.argmax(objective))
        best_idx = int(chunk[best_j])
        selected.append(best_idx)
        selected_feat = np.vstack([selected_feat, feat[best_idx][None, :]])
        current_div = float(new_divs[best_j])

    if len(selected) < k:
        leftover = [int(i) for i in filtered if int(i) not in set(selected)]
        selected.extend(leftover[: k - len(selected)])
    return np.asarray(selected[:k], dtype=np.int64)


def maxp_selection_order(maxp_scores):
    scores = np.asarray(maxp_scores, dtype=np.float64).reshape(-1)
    n = scores.shape[0]
    order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
    return order.astype(np.int64, copy=False)


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


def compute_selection_curves_by_sets(storage_rows, selected_by_budget, budget_ratios):
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
    return {'total_errors': total_errors, 'trc': out_trc}


def _jsonable_selected_by_budget(selected_by_budget):
    out = {}
    for key, idx in selected_by_budget.items():
        out[_budget_key(key)] = [int(i) for i in np.asarray(idx).reshape(-1)]
    return out


def build_or_load_sets_probs(
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
        probs = np.asarray(z['probs'], dtype=np.float32)
        maxp = np.asarray(z['maxp_score'], dtype=np.float64).reshape(-1)
        gini = np.asarray(z['gini_score'], dtype=np.float64).reshape(-1)
        if probs.shape[0] == n and maxp.shape[0] == n and gini.shape[0] == n:
            print(f'Loaded SETS softmax/uncertainty from {cache_path}')
            return probs, maxp, gini
        print(f'SETS softmax cache mismatch, recomputing -> {cache_path}')

    probs = extract_sets_probs(data, model, batch_size=batch_size)
    maxp = maxp_score(probs)
    gini = gini_score(probs)
    np.savez_compressed(
        cache_path,
        probs=probs,
        maxp_score=maxp.astype(np.float32, copy=False),
        gini_score=gini.astype(np.float32, copy=False),
    )
    print(f'Saved SETS softmax/uncertainty to {cache_path}')
    return probs, maxp, gini


def build_or_load_sets_features(
    *,
    cache_dir,
    cache_name,
    data,
    model,
    sets_feature_layer_index,
    batch_size=64,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n = int(len(data))
    layer_index = int(sets_feature_layer_index)

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        feat = np.asarray(z['feature'], dtype=np.float32)
        feat_mm = np.asarray(z['feature_minmax'], dtype=np.float64)
        cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
        if feat.shape[0] == n and feat_mm.shape[0] == n and cached_layer == layer_index:
            print(f'Loaded SETS features from {cache_path}')
            return feat, feat_mm
        print(f'SETS feature cache mismatch, recomputing -> {cache_path}')

    feat = extract_sets_hidden(
        data, model, layer_index, batch_size=batch_size,
    )
    feat_mm = minmax_normalize_feature_matrix(feat)
    np.savez_compressed(
        cache_path,
        feature=feat,
        feature_minmax=feat_mm.astype(np.float32, copy=False),
        layer_index=np.int32(layer_index),
    )
    print(f'Saved SETS features to {cache_path}')
    return feat, feat_mm


def build_or_load_sets_selection(
    *,
    cache_dir,
    cache_name,
    uncertainty_scores,
    features,
    budget_ratios,
    n_pool,
    a,
    diversity,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    keys = [_budget_key(b) for b in budget_ratios]

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('a', 'diversity') + tuple(f'sel_{key:.4f}' for key in keys)
        missing = [key for key in required if key not in z.files]
        reasons = []
        selected = {}
        if missing:
            reasons.append(f'missing keys {missing}')
        else:
            cached_a = float(np.asarray(z['a']).reshape(-1)[0])
            cached_div = (
                str(z['diversity'].item())
                if hasattr(z['diversity'], 'item')
                else str(z['diversity'])
            )
            if not np.isclose(cached_a, float(a), rtol=0.0, atol=1e-5):
                reasons.append(f'a={cached_a} vs {float(a)}')
            if cached_div != str(diversity):
                reasons.append(f'diversity={cached_div!r} vs {str(diversity)!r}')
            for key in keys:
                arr_key = f'sel_{key:.4f}'
                selected[key] = np.asarray(z[arr_key], dtype=np.int64).reshape(-1)
                expect_k = _budget_k(n_pool, key)
                if selected[key].shape[0] != expect_k:
                    reasons.append(
                        f'{arr_key} len={selected[key].shape[0]} vs k={expect_k}',
                    )
        if not reasons:
            print(f'Loaded SETS selection from {cache_path}')
            return selected
        print(
            f'SETS selection cache mismatch ({"; ".join(reasons)}), '
            f'recomputing -> {cache_path}',
        )

    selected = {}
    for budget_ratio in budget_ratios:
        k = _budget_k(n_pool, budget_ratio)
        print(f'  SETS greedy budget={budget_ratio:.0%} k={k} a={a} diversity={diversity}')
        selected[_budget_key(budget_ratio)] = sets_chunked_select(
            k,
            uncertainty_scores,
            features,
            a=a,
            diversity=diversity,
        )
    save_dict = {
        'a': np.float64(a),
        'diversity': np.asarray(str(diversity)),
        'n_pool': np.int32(n_pool),
    }
    for key, idx in selected.items():
        save_dict[f'sel_{key:.4f}'] = np.asarray(idx, dtype=np.int64)
    np.savez_compressed(cache_path, **save_dict)
    print(f'Saved SETS selection to {cache_path}')
    return selected


def evaluate_sets_pool(
    pool,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    seed,
    sampling_mode,
    model,
    sets_feature_layer_index,
    cache_dir,
    budget_ratios,
    apply_diversity=True,
    uncertainty=SETS_UNCERTAINTY,
    diversity=SETS_DIVERSITY,
    a=SETS_A,
    batch_size=64,
    force_recompute=False,
    return_selection=False,
):
    """
    RQ entry for SETS.

    apply_diversity=False -> rank by uncertainty only (RQ1 method name = uncertainty,
    e.g. 'maxp').
    apply_diversity=True  -> official chunked greedy (method name 'sets').
    """
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)
    unc_name = str(uncertainty).lower()
    div_name = str(diversity).lower()

    _probs, maxp, gini = build_or_load_sets_probs(
        cache_dir=cache_dir,
        cache_name=f'sets_{tag}_probs.npz',
        data=pool['data'],
        model=model,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    scores = maxp if unc_name == 'maxp' else gini if unc_name == 'gini' else uncertainty_scores_from_probs(
        _probs, uncertainty=unc_name,
    )

    if apply_diversity:
        _feat, feat_mm = build_or_load_sets_features(
            cache_dir=cache_dir,
            cache_name=f'sets_{tag}_feat_layer{int(sets_feature_layer_index)}.npz',
            data=pool['data'],
            model=model,
            sets_feature_layer_index=sets_feature_layer_index,
            batch_size=batch_size,
            force_recompute=force_recompute,
        )
        selected_by_budget = build_or_load_sets_selection(
            cache_dir=cache_dir,
            cache_name=(
                f'sets_{tag}_sel_a{int(a)}_{unc_name}_{div_name}'
                f'_layer{int(sets_feature_layer_index)}.npz'
            ),
            uncertainty_scores=scores,
            features=feat_mm,
            budget_ratios=budget_ratios,
            n_pool=n_pool,
            a=a,
            diversity=div_name,
            force_recompute=force_recompute,
        )
        curve = compute_selection_curves_by_sets(
            storage, selected_by_budget, budget_ratios=budget_ratios,
        )
        method_name = 'sets'
        order = None
    else:
        order = maxp_selection_order(scores)
        selected_by_budget = selected_by_budget_from_order(order, budget_ratios, n_pool)
        curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
        method_name = unc_name

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
        'apply_diversity': bool(apply_diversity),
        'uncertainty': unc_name,
        'diversity': div_name if apply_diversity else None,
        'a': int(a) if apply_diversity else None,
        'sets_feature_layer_index': int(sets_feature_layer_index),
        'trc_by_budget': ratio_to_trc,
    }
    if return_selection:
        if order is not None:
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
    parser = argparse.ArgumentParser(description='SETS baseline TRC evaluation on sampled pools.')
    parser.add_argument('--datasets', nargs='+', default=RQ1_DATASETS, choices=sorted(RQ1_DATASETS))
    parser.add_argument('--pool-types', nargs='+', default=RQ1_POOL_TYPES, choices=RQ1_POOL_TYPES)
    parser.add_argument('--error-ratios', nargs='+', type=float, default=RQ1_ERROR_RATIOS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument(
        '--error-sampling-mode',
        default='random',
        choices=list(ERROR_SAMPLING_MODES),
    )
    parser.add_argument('--sampled-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--output-dir', default=str(_EXP_DIR / 'results' / 'rq1'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--uncertainty', default=SETS_UNCERTAINTY, choices=list(SETS_UNCERTAINTY_CHOICES))
    parser.add_argument('--diversity', default=SETS_DIVERSITY, choices=list(SETS_DIVERSITY_CHOICES))
    parser.add_argument('--a', type=int, default=SETS_A)
    parser.add_argument(
        '--apply-diversity',
        action='store_true',
        help='run official chunked greedy SETS (default: uncertainty ranking only)',
    )
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    configure_tensorflow_gpu()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_diversity = bool(args.apply_diversity)
    method_tag = 'sets' if apply_diversity else str(args.uncertainty)

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
                        f'Evaluating {method_tag} {dataset_name}/{pool_type}/'
                        f'{args.error_sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                        f'(n={n_pool}, errors={n_errors}, diversity={apply_diversity})',
                    )
                    row = evaluate_sets_pool(
                        pool,
                        dataset_name=dataset_name,
                        pool_type=pool_type,
                        error_ratio=error_ratio,
                        seed=seed,
                        sampling_mode=args.error_sampling_mode,
                        model=model,
                        sets_feature_layer_index=cfg['sets_feature_layer_index'],
                        cache_dir=cache_dir,
                        budget_ratios=RQ1_BUDGET_RATIOS,
                        apply_diversity=apply_diversity,
                        uncertainty=str(args.uncertainty),
                        diversity=str(args.diversity),
                        a=int(args.a),
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
    print(f'\n=== SETS baseline TRC summary ({method_tag}, mean over seeds) ===')
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
