import argparse
import itertools
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

ATS_METHOD_NAME = 'ats'
# Official ATS/ats_config.py + demo.py th.
ATS_TH = 0.001
ATS_LINEAR_RATIO = 0.005
ATS_UP_BOUNDARY = 0.99
ATS_BOUNDARY = 0.0
ATS_ROUND_NUM = 5
ATS_S_UP_SHUFFLE_SEED = 0

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


def _float_tag(value):
    text = f'{float(value):.6f}'.rstrip('0').rstrip('.')
    return text.replace('-', 'm').replace('.', 'p')


def pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode='random'):
    ratio_pct = int(round(float(error_ratio) * 100))
    return f'rq1_{dataset_name}_{pool_type}_r{ratio_pct:02d}_s{int(seed)}_{sampling_mode}'


def ats_selection_cache_name(tag, th=ATS_TH):
    return f'ats_{tag}_sel_th{_float_tag(th)}.npz'


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


def extract_ats_probs(data, model, batch_size=64):
    chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc='ats softmax'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        raw = model(batch, training=False)
        probs = _probs_from_model_output_tensor(raw).numpy().astype(np.float32, copy=False)
        chunks.append(probs)
    out = np.concatenate(chunks, axis=0)
    if out.shape[0] != n:
        raise ValueError('ATS softmax length mismatch with pool')
    return out


def pq_pairs(n_classes, pred_class):
    others = [j for j in range(int(n_classes)) if j != int(pred_class)]
    return list(itertools.combinations(others, 2))


def project_onto_ipq(probs, pred_class, p, q):
    """Project softmax onto the affine hull of (e_i, e_p, e_q).

    Official ProjectExtendStep.get_projection_point assigns A_dot[p] twice
    (the second write was meant for A_dot[q]). This follows the paper formula.
    """
    a = np.asarray(probs, dtype=np.float64)
    i = int(pred_class)
    out = np.zeros_like(a, dtype=np.float64)
    ai = a[:, i]
    ap = a[:, p]
    aq = a[:, q]
    out[:, i] = (2.0 / 3.0) * ai - (1.0 / 3.0) * ap - (1.0 / 3.0) * aq + (1.0 / 3.0)
    out[:, p] = (2.0 / 3.0) * ap - (1.0 / 3.0) * aq - (1.0 / 3.0) * ai + (1.0 / 3.0)
    out[:, q] = (2.0 / 3.0) * aq - (1.0 / 3.0) * ap - (1.0 / 3.0) * ai + (1.0 / 3.0)
    return out


def extend_to_boundary(proj, pred_class, boundary=ATS_BOUNDARY):
    """Ray from predicted-class vertex through the projected point, onto the opposite face."""
    i = int(pred_class)
    ext = np.array(proj, dtype=np.float64, copy=True)
    dist = 1.0 - ext[:, i]
    valid = dist > 1e-12
    scale = np.zeros(ext.shape[0], dtype=np.float64)
    scale[valid] = (1.0 - float(boundary)) / dist[valid]
    ext *= scale[:, None]
    ext[:, i] = float(boundary)
    ext[~valid] = 0.0
    return ext, dist


def fault_pattern_intervals(
    probs,
    pred_class,
    n_classes,
    *,
    linear_ratio=ATS_LINEAR_RATIO,
    up_boundary=ATS_UP_BOUNDARY,
    boundary=ATS_BOUNDARY,
    round_num=ATS_ROUND_NUM,
):
    """One [a, b] interval per (p, q) pair. Width L = linear_ratio * (up_boundary - p'_i)."""
    pmat = np.asarray(probs, dtype=np.float64)
    if pmat.ndim != 2:
        raise ValueError('probs must be (n_mid, n_classes)')
    m = int(pmat.shape[0])
    pairs = pq_pairs(n_classes, pred_class)
    n_pairs = len(pairs)
    if m == 0 or n_pairs == 0:
        return np.zeros((m, n_pairs, 2), dtype=np.float64), np.zeros(m, dtype=np.float64)

    intervals = np.zeros((m, n_pairs, 2), dtype=np.float64)
    hi = 1.0 - float(boundary)
    for k, (p, q) in enumerate(pairs):
        proj = project_onto_ipq(pmat, pred_class, p, q)
        radius_d = float(up_boundary) - proj[:, int(pred_class)]
        ext, ext_d = extend_to_boundary(proj, pred_class, boundary=boundary)
        center = ext[:, p]
        cov_l = float(linear_ratio) * radius_d
        left = np.round(center - cov_l, int(round_num))
        right = np.round(center + cov_l, int(round_num))
        left = np.clip(left, float(boundary), hi)
        right = np.clip(right, float(boundary), hi)
        invalid = (radius_d < 0.0) | (center < 0.0) | (ext_d <= 1e-12)
        left = np.where(invalid, 0.0, left)
        right = np.where(invalid, 0.0, right)
        swap = left > right
        left = np.where(swap, right, left)
        intervals[:, k, 0] = left
        intervals[:, k, 1] = right

    lengths = np.maximum(0.0, intervals[:, :, 1] - intervals[:, :, 0]).sum(axis=1)
    return intervals, lengths


def _merge_interval(segs, left, right):
    if right <= left:
        return segs
    out = []
    na, nb = float(left), float(right)
    placed = False
    for start, end in segs:
        if end < na:
            out.append((start, end))
        elif start > nb:
            if not placed:
                out.append((na, nb))
                placed = True
            out.append((start, end))
        else:
            na = min(na, start)
            nb = max(nb, end)
    if not placed:
        out.append((na, nb))
    return out


def _coverage_deltas(intervals, idx, merged_per_pair):
    if idx.size == 0:
        return np.zeros(0, dtype=np.float64)
    left = intervals[idx, :, 0]
    right = intervals[idx, :, 1]
    added = np.maximum(right - left, 0.0)
    for pair_i, segs in enumerate(merged_per_pair):
        if not segs:
            continue
        a_p = left[:, pair_i]
        b_p = right[:, pair_i]
        for start, end in segs:
            overlap = np.minimum(b_p, end) - np.maximum(a_p, start)
            added[:, pair_i] -= np.maximum(overlap, 0.0)
    return np.maximum(added, 0.0).sum(axis=1)


def greedy_pattern_select(intervals, lengths, th=ATS_TH):
    """Select S_mid samples by incremental union coverage. Returns (selected, deltas, leftover)."""
    m = int(intervals.shape[0])
    n_pairs = int(intervals.shape[1])
    if m == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, np.zeros(0, dtype=np.float64), empty

    remaining = np.ones(m, dtype=bool)
    selected_mask = np.zeros(m, dtype=bool)
    merged = [[] for _ in range(n_pairs)]
    selected = []
    selected_delta = []
    th = float(th)

    while True:
        idx = np.flatnonzero(remaining)
        if idx.size == 0:
            break
        deltas = _coverage_deltas(intervals, idx, merged)
        dead = deltas <= th
        remaining[idx[dead]] = False
        live = ~dead
        if not np.any(live):
            break
        live_idx = idx[live]
        live_delta = deltas[live]
        best = int(np.argmax(live_delta))
        pick = int(live_idx[best])
        delta = float(live_delta[best])
        selected.append(pick)
        selected_delta.append(delta)
        selected_mask[pick] = True
        remaining[pick] = False
        for pair_i in range(n_pairs):
            merged[pair_i] = _merge_interval(
                merged[pair_i],
                intervals[pick, pair_i, 0],
                intervals[pick, pair_i, 1],
            )

    leftover = np.flatnonzero(~selected_mask)
    if leftover.size > 0:
        leftover = leftover[np.argsort(-np.asarray(lengths)[leftover], kind='stable')]
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(selected_delta, dtype=np.float64),
        leftover.astype(np.int64, copy=False),
    )


def _sort_by_length_desc(indices, lengths):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    if indices.shape[0] == 0:
        return indices, lengths
    order = np.argsort(-lengths, kind='stable')
    return indices[order], lengths[order]


def ats_priority_sequence(
    probs,
    *,
    th=ATS_TH,
    linear_ratio=ATS_LINEAR_RATIO,
    up_boundary=ATS_UP_BOUNDARY,
    boundary=ATS_BOUNDARY,
    round_num=ATS_ROUND_NUM,
    shuffle_seed=ATS_S_UP_SHUFFLE_SEED,
):
    """Official ATS priority sequence: greedy-by-Δ, leftover-by-length, then shuffled S_up."""
    pmat = np.asarray(probs, dtype=np.float64)
    if pmat.ndim != 2:
        raise ValueError('probs must be (n_pool, n_classes)')
    n_pool, n_classes = int(pmat.shape[0]), int(pmat.shape[1])
    pred = np.argmax(pmat, axis=1).astype(np.int64, copy=False)

    selected_idx = []
    selected_len = []
    leftover_idx = []
    leftover_len = []
    up_idx = []
    low_idx = []

    for class_i in tqdm(range(n_classes), desc='ATS class greedy'):
        class_mask = pred == class_i
        class_ix = np.flatnonzero(class_mask)
        if class_ix.size == 0:
            continue
        p_i = pmat[class_ix, class_i]
        mid_rel = (p_i >= float(boundary)) & (p_i < float(up_boundary))
        up_rel = p_i >= float(up_boundary)
        low_rel = p_i < float(boundary)
        up_idx.extend(class_ix[up_rel].tolist())
        low_idx.extend(class_ix[low_rel].tolist())

        mid_ix = class_ix[mid_rel]
        if mid_ix.size == 0:
            continue
        intervals, lengths = fault_pattern_intervals(
            pmat[mid_ix],
            class_i,
            n_classes,
            linear_ratio=linear_ratio,
            up_boundary=up_boundary,
            boundary=boundary,
            round_num=round_num,
        )
        sel_rel, sel_delta, left_rel = greedy_pattern_select(intervals, lengths, th=th)
        if sel_rel.size:
            selected_idx.append(mid_ix[sel_rel])
            selected_len.append(sel_delta)
        if left_rel.size:
            leftover_idx.append(mid_ix[left_rel])
            leftover_len.append(lengths[left_rel])

    if selected_idx:
        sel_all = np.concatenate(selected_idx, axis=0)
        sel_all_len = np.concatenate(selected_len, axis=0)
        sel_all, _ = _sort_by_length_desc(sel_all, sel_all_len)
    else:
        sel_all = np.zeros(0, dtype=np.int64)

    if leftover_idx:
        left_all = np.concatenate(leftover_idx, axis=0)
        left_all_len = np.concatenate(leftover_len, axis=0)
        left_all, _ = _sort_by_length_desc(left_all, left_all_len)
    else:
        left_all = np.zeros(0, dtype=np.int64)

    up_all = np.asarray(up_idx, dtype=np.int64)
    if up_all.size:
        rng = np.random.RandomState(int(shuffle_seed))
        up_all = up_all[rng.permutation(up_all.shape[0])]
    low_all = np.asarray(low_idx, dtype=np.int64)

    parts = [arr for arr in (sel_all, left_all, up_all, low_all) if arr.size]
    if not parts:
        raise ValueError('ATS produced an empty priority sequence')
    order = np.concatenate(parts, axis=0)
    if order.shape[0] != n_pool or np.unique(order).shape[0] != n_pool:
        raise ValueError(
            f'ATS rank is not a permutation of the pool '
            f'(n={n_pool}, rank={order.shape[0]}, unique={np.unique(order).shape[0]})',
        )
    return order


def build_or_load_ats_probs(
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
                print(f'Loaded ATS softmax from {cache_path}')
                return probs
        print(f'ATS softmax cache mismatch, recomputing -> {cache_path}')

    probs = extract_ats_probs(data, model, batch_size=batch_size)
    np.savez_compressed(cache_path, probs=np.asarray(probs, dtype=np.float32))
    print(f'Saved ATS softmax to {cache_path}')
    return np.asarray(probs, dtype=np.float64)


def build_or_load_ats_selection(
    *,
    cache_dir,
    cache_name,
    probs,
    th,
    linear_ratio,
    up_boundary,
    boundary,
    round_num,
    shuffle_seed,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n_pool = int(probs.shape[0])

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('sort_ix', 'th', 'linear_ratio', 'up_boundary', 'n_pool')
        missing = [key for key in required if key not in z.files]
        reasons = []
        if missing:
            reasons.append(f'missing keys {missing}')
        else:
            order = np.asarray(z['sort_ix'], dtype=np.int64).reshape(-1)
            cached_th = float(np.asarray(z['th'], dtype=np.float64).reshape(-1)[0])
            cached_ratio = float(np.asarray(z['linear_ratio'], dtype=np.float64).reshape(-1)[0])
            cached_up = float(np.asarray(z['up_boundary'], dtype=np.float64).reshape(-1)[0])
            cached_n = int(np.asarray(z['n_pool']).reshape(-1)[0])
            if order.shape[0] != n_pool:
                reasons.append(f'rank len={order.shape[0]} vs n={n_pool}')
            if cached_n != n_pool:
                reasons.append(f'n_pool={cached_n} vs {n_pool}')
            if not np.isclose(cached_th, float(th), rtol=0.0, atol=1e-12):
                reasons.append(f'th={cached_th} vs {float(th)}')
            if not np.isclose(cached_ratio, float(linear_ratio), rtol=0.0, atol=1e-12):
                reasons.append(f'linear_ratio={cached_ratio} vs {float(linear_ratio)}')
            if not np.isclose(cached_up, float(up_boundary), rtol=0.0, atol=1e-12):
                reasons.append(f'up_boundary={cached_up} vs {float(up_boundary)}')
            if not reasons:
                print(f'Loaded ATS selection from {cache_path}')
                return order
        print(
            f'ATS selection cache mismatch ({"; ".join(reasons)}), '
            f'recomputing -> {cache_path}',
        )

    print(
        f'  ATS greedy n={n_pool} classes={probs.shape[1]} '
        f'th={float(th)} ratio={float(linear_ratio)} up={float(up_boundary)}',
    )
    order = ats_priority_sequence(
        probs,
        th=th,
        linear_ratio=linear_ratio,
        up_boundary=up_boundary,
        boundary=boundary,
        round_num=round_num,
        shuffle_seed=shuffle_seed,
    )
    np.savez_compressed(
        cache_path,
        sort_ix=np.asarray(order, dtype=np.int64),
        th=np.float64(th),
        linear_ratio=np.float64(linear_ratio),
        up_boundary=np.float64(up_boundary),
        boundary=np.float64(boundary),
        round_num=np.int32(round_num),
        shuffle_seed=np.int32(shuffle_seed),
        n_pool=np.int32(n_pool),
        projection_fixed=np.asarray('paper_A_dot_q'),
    )
    print(f'Saved ATS selection to {cache_path}')
    return order


def evaluate_ats_pool(
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
    th=ATS_TH,
    linear_ratio=ATS_LINEAR_RATIO,
    up_boundary=ATS_UP_BOUNDARY,
    boundary=ATS_BOUNDARY,
    round_num=ATS_ROUND_NUM,
    shuffle_seed=ATS_S_UP_SHUFFLE_SEED,
    batch_size=64,
    force_recompute=False,
    return_selection=False,
):
    """
    RQ entry: official ATS greedy priority sequence, prefix TRC.

    Returns a single result row (method='ats').
    """
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)

    probs = build_or_load_ats_probs(
        cache_dir=cache_dir,
        cache_name=f'ats_{tag}_probs.npz',
        data=pool['data'],
        model=model,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    order = build_or_load_ats_selection(
        cache_dir=cache_dir,
        cache_name=ats_selection_cache_name(tag, th=th),
        probs=probs,
        th=th,
        linear_ratio=linear_ratio,
        up_boundary=up_boundary,
        boundary=boundary,
        round_num=round_num,
        shuffle_seed=shuffle_seed,
        force_recompute=force_recompute,
    )
    selected_by_budget = selected_by_budget_from_order(order, budget_ratios, n_pool)
    curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
    ratio_to_trc = _trc_by_budget_dict(budget_ratios, curve)
    row = {
        'method': ATS_METHOD_NAME,
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_errors': int(curve['total_errors']),
        'th': float(th),
        'linear_ratio': float(linear_ratio),
        'up_boundary': float(up_boundary),
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
        description='ATS baseline TRC (official greedy coverage) on sampled pools.',
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
    parser.add_argument('--th', type=float, default=ATS_TH)
    parser.add_argument('--linear-ratio', type=float, default=ATS_LINEAR_RATIO)
    parser.add_argument('--up-boundary', type=float, default=ATS_UP_BOUNDARY)
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
                            f'Evaluating ats {dataset_name}/{pool_type}/'
                            f'{sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                            f'(n={n_pool}, errors={n_errors})',
                        )
                        row = evaluate_ats_pool(
                            pool,
                            dataset_name=dataset_name,
                            pool_type=pool_type,
                            error_ratio=error_ratio,
                            seed=seed,
                            sampling_mode=sampling_mode,
                            model=model,
                            cache_dir=cache_dir,
                            budget_ratios=RQ1_BUDGET_RATIOS,
                            th=float(args.th),
                            linear_ratio=float(args.linear_ratio),
                            up_boundary=float(args.up_boundary),
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
    print('\n=== ATS baseline TRC summary (greedy coverage) ===')
    print(table_text)

    mode_tag = '_'.join(args.error_sampling_modes)
    runs_path = out_dir / f'rq1_ats_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_ats_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_ats_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
