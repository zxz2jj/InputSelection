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

MCP_METHOD_NAME = 'mcp'
MCP_PRIORITY_METHOD_NAME = 'mcp_priority'
# Official select_from_firstsec_dic first-pass filter.
MCP_RATIO_MIN = 0.1
MCP_RATIO_EPS = 1e-12

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


def mcp_selection_cache_name(tag, ratio_min, apply_clustering):
    mode = 'cluster' if apply_clustering else 'priority'
    return f'mcp_{tag}_sel_{mode}_r{_float_tag(ratio_min)}.npz'


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


def compute_selection_curves_by_sets(storage_rows, selected_by_budget, budget_ratios):
    """TRC on an independent selected set per budget (full MCP clustering)."""
    err = _error_mask_from_storage(storage_rows)
    n_pool = len(storage_rows)
    total_errors = int(np.sum(err))

    out_trc = []
    for r in budget_ratios:
        k = _budget_k(n_pool, r)
        chosen = np.asarray(selected_by_budget[_budget_key(r)], dtype=np.int64).reshape(-1)
        if chosen.shape[0] != k:
            raise ValueError(
                f'MCP selected size {chosen.shape[0]} != budget k={k} for ratio={r}',
            )
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


def extract_mcp_probs(data, model, batch_size=64):
    chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc='mcp softmax'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        raw = model(batch, training=False)
        probs = _probs_from_model_output_tensor(raw).numpy().astype(np.float32, copy=False)
        chunks.append(probs)
    out = np.concatenate(chunks, axis=0)
    if out.shape[0] != n:
        raise ValueError('MCP softmax length mismatch with pool')
    return out


def mcp_boundary_stats(probs, ratio_eps=MCP_RATIO_EPS):
    """c1 = predicted class, c2 = second class, ratio = p_c2 / p_c1."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError('probs must be (n, num_classes)')
    n, n_classes = p.shape
    c1 = np.argmax(p, axis=1).astype(np.int64, copy=False)
    tmp = p.copy()
    tmp[np.arange(n), c1] = -np.inf
    c2 = np.argmax(tmp, axis=1).astype(np.int64, copy=False)
    p1 = p[np.arange(n), c1]
    p2 = p[np.arange(n), c2]
    ratio = (p2 / np.maximum(p1, float(ratio_eps))).astype(np.float64, copy=False)
    return c1, c2, ratio, int(n_classes)


def mcp_priority_order(ratio):
    scores = np.asarray(ratio, dtype=np.float64).reshape(-1)
    n = scores.shape[0]
    order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
    return order.astype(np.int64, copy=False)


def _n_nonempty(dicratio):
    return sum(1 for row in dicratio if len(row) > 0)


def mcp_cluster_select_k(c1, c2, ratio, k, n_classes, ratio_min=MCP_RATIO_MIN):
    """
    Official MCP select_from_firstsec_dic for one budget k.

    Phase 1: while remaining >= nonempty buckets, take the current max-ratio
    sample from each nonempty bucket if ratio >= ratio_min.
    Phase 2: fill the rest from per-bucket maxima (no ratio_min), highest first.

    Fixes official padding with pool index 0, and the r<ratio_min infinite loop.
    """
    c1 = np.asarray(c1, dtype=np.int64).reshape(-1)
    c2 = np.asarray(c2, dtype=np.int64).reshape(-1)
    ratio = np.asarray(ratio, dtype=np.float64).reshape(-1)
    n = int(c1.shape[0])
    if c2.shape[0] != n or ratio.shape[0] != n:
        raise ValueError('c1/c2/ratio length mismatch')
    n_classes = int(n_classes)
    k = max(0, min(int(k), n))
    if k == 0:
        return np.zeros((0,), dtype=np.int64)

    n_buckets = n_classes * n_classes
    dicratio = [[] for _ in range(n_buckets)]
    dicindex = [[] for _ in range(n_buckets)]
    for i in range(n):
        b = int(c1[i]) * n_classes + int(c2[i])
        if b < 0 or b >= n_buckets:
            raise ValueError(f'class pair out of range: {(int(c1[i]), int(c2[i]))}')
        dicratio[b].append(float(ratio[i]))
        dicindex[b].append(int(i))

    selected = []
    selected_set = set()
    ratio_min = float(ratio_min)

    while (k - len(selected)) >= _n_nonempty(dicratio) and _n_nonempty(dicratio) > 0:
        picked = 0
        for b in range(n_buckets):
            if not dicratio[b]:
                continue
            tmp = max(dicratio[b])
            j = dicratio[b].index(tmp)
            if tmp >= ratio_min:
                idx = dicindex[b][j]
                selected.append(idx)
                selected_set.add(idx)
                dicratio[b].pop(j)
                dicindex[b].pop(j)
                picked += 1
        if picked == 0:
            break

    need = k - len(selected)
    if need > 0:
        cand_r = []
        cand_i = []
        for b in range(n_buckets):
            if not dicratio[b]:
                continue
            tmp = max(dicratio[b])
            j = dicratio[b].index(tmp)
            cand_r.append(tmp)
            cand_i.append(dicindex[b][j])
        if cand_i:
            max_tmp = [0.0] * need
            max_idx = [-1] * need
            for r_val, idx in zip(cand_r, cand_i):
                min_slot = min(max_tmp)
                if r_val > min_slot:
                    pos = max_tmp.index(min_slot)
                    max_tmp[pos] = r_val
                    max_idx[pos] = idx
            for idx in max_idx:
                if idx >= 0 and idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)

        leftover = [i for i in range(n) if i not in selected_set]
        if leftover and len(selected) < k:
            leftover = np.asarray(leftover, dtype=np.int64)
            left_ratio = ratio[leftover]
            take_order = np.lexsort((leftover, -left_ratio))
            for idx in leftover[take_order]:
                if len(selected) >= k:
                    break
                selected.append(int(idx))
                selected_set.add(int(idx))

    if len(selected) != k:
        raise RuntimeError(f'MCP selected {len(selected)} != requested k={k}')
    return np.asarray(selected, dtype=np.int64)


def mcp_cluster_select_by_budget(c1, c2, ratio, budget_ratios, n_pool, n_classes, ratio_min=MCP_RATIO_MIN):
    out = {}
    for budget_ratio in budget_ratios:
        k = _budget_k(n_pool, budget_ratio)
        out[_budget_key(budget_ratio)] = mcp_cluster_select_k(
            c1, c2, ratio, k=k, n_classes=n_classes, ratio_min=ratio_min,
        )
    return out


def build_or_load_mcp_probs(
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
                print(f'Loaded MCP softmax from {cache_path}')
                return probs
        print(f'MCP softmax cache mismatch, recomputing -> {cache_path}')

    probs = extract_mcp_probs(data, model, batch_size=batch_size)
    np.savez_compressed(cache_path, probs=np.asarray(probs, dtype=np.float32))
    print(f'Saved MCP softmax to {cache_path}')
    return np.asarray(probs, dtype=np.float64)


def build_or_load_mcp_selection(
    *,
    cache_dir,
    cache_name,
    probs,
    budget_ratios,
    apply_clustering,
    ratio_min=MCP_RATIO_MIN,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n_pool = int(probs.shape[0])
    c1, c2, ratio, n_classes = mcp_boundary_stats(probs)
    budget_keys = [_budget_key(b) for b in budget_ratios]

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('n_pool', 'n_classes', 'ratio_min', 'apply_clustering', 'c1', 'c2', 'ratio')
        missing = [key for key in required if key not in z.files]
        reasons = []
        if missing:
            reasons.append(f'missing keys {missing}')
        else:
            cached_n = int(np.asarray(z['n_pool']).reshape(-1)[0])
            cached_c = int(np.asarray(z['n_classes']).reshape(-1)[0])
            cached_rmin = float(np.asarray(z['ratio_min'], dtype=np.float64).reshape(-1)[0])
            cached_cluster = bool(np.asarray(z['apply_clustering']).reshape(-1)[0])
            if cached_n != n_pool:
                reasons.append(f'n_pool={cached_n} vs {n_pool}')
            if cached_c != n_classes:
                reasons.append(f'n_classes={cached_c} vs {n_classes}')
            if cached_cluster != bool(apply_clustering):
                reasons.append(f'cluster={cached_cluster} vs {bool(apply_clustering)}')
            if not np.isclose(cached_rmin, float(ratio_min), rtol=0.0, atol=1e-12):
                reasons.append(f'ratio_min={cached_rmin} vs {float(ratio_min)}')
            selected_by_budget = {}
            if not reasons:
                for b, key in zip(budget_ratios, budget_keys):
                    sel_name = f'sel_{key}'
                    if sel_name not in z.files:
                        reasons.append(f'missing {sel_name}')
                        break
                    chosen = np.asarray(z[sel_name], dtype=np.int64).reshape(-1)
                    k = _budget_k(n_pool, b)
                    if chosen.shape[0] != k:
                        reasons.append(f'{sel_name} len={chosen.shape[0]} vs k={k}')
                        break
                    selected_by_budget[key] = chosen
            if not reasons:
                print(f'Loaded MCP selection from {cache_path}')
                return selected_by_budget, c1, c2, ratio
        print(
            f'MCP selection cache mismatch ({"; ".join(reasons)}), '
            f'recomputing -> {cache_path}',
        )

    if apply_clustering:
        print(
            f'  MCP clustering n={n_pool} classes={n_classes} '
            f'ratio_min={float(ratio_min)} budgets={list(budget_ratios)}',
        )
        selected_by_budget = mcp_cluster_select_by_budget(
            c1, c2, ratio, budget_ratios, n_pool, n_classes, ratio_min=ratio_min,
        )
    else:
        order = mcp_priority_order(ratio)
        selected_by_budget = selected_by_budget_from_order(order, budget_ratios, n_pool)

    payload = {
        'n_pool': np.int32(n_pool),
        'n_classes': np.int32(n_classes),
        'ratio_min': np.float64(ratio_min),
        'apply_clustering': np.int32(1 if apply_clustering else 0),
        'c1': c1.astype(np.int64, copy=False),
        'c2': c2.astype(np.int64, copy=False),
        'ratio': ratio.astype(np.float32, copy=False),
    }
    for key, idx in selected_by_budget.items():
        payload[f'sel_{key}'] = np.asarray(idx, dtype=np.int64)
    np.savez_compressed(cache_path, **payload)
    print(f'Saved MCP selection to {cache_path}')
    return selected_by_budget, c1, c2, ratio


def evaluate_mcp_pool(
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
    apply_clustering=True,
    ratio_min=MCP_RATIO_MIN,
    batch_size=64,
    force_recompute=False,
    return_selection=False,
):
    """
    RQ entry: MCP on one sampled pool.

    apply_clustering=True  -> official per-budget clustering + prioritization (RQ1).
    apply_clustering=False -> rank by p_c2/p_c1 prefixes only.
    """
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)

    probs = build_or_load_mcp_probs(
        cache_dir=cache_dir,
        cache_name=f'mcp_{tag}_probs.npz',
        data=pool['data'],
        model=model,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    selected_by_budget, c1, c2, ratio = build_or_load_mcp_selection(
        cache_dir=cache_dir,
        cache_name=mcp_selection_cache_name(tag, ratio_min, apply_clustering),
        probs=probs,
        budget_ratios=budget_ratios,
        apply_clustering=apply_clustering,
        ratio_min=ratio_min,
        force_recompute=force_recompute,
    )

    if apply_clustering:
        curve = compute_selection_curves_by_sets(
            storage, selected_by_budget, budget_ratios=budget_ratios,
        )
        method_name = MCP_METHOD_NAME
    else:
        order = mcp_priority_order(ratio)
        curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
        method_name = MCP_PRIORITY_METHOD_NAME

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
        'apply_clustering': bool(apply_clustering),
        'ratio_min': float(ratio_min),
        'trc_by_budget': ratio_to_trc,
    }
    if return_selection:
        row['selected_by_budget'] = _jsonable_selected_by_budget(selected_by_budget)
        row['c1'] = [int(v) for v in np.asarray(c1).reshape(-1)]
        row['c2'] = [int(v) for v in np.asarray(c2).reshape(-1)]
        row['ratio'] = [float(v) for v in np.asarray(ratio, dtype=np.float64).reshape(-1)]
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
        description='MCP baseline TRC (multiple-boundary clustering + prioritization) on sampled pools.',
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
    parser.add_argument('--ratio-min', type=float, default=MCP_RATIO_MIN)
    parser.add_argument(
        '--no-clustering',
        action='store_true',
        help='rank by p_c2/p_c1 only (no per-budget clustering)',
    )
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def _drop_heavy_fields(run_row):
    skip = {'selected_by_budget', 'c1', 'c2', 'ratio'}
    return {key: value for key, value in run_row.items() if key not in skip}


def main():
    args = parse_args()
    configure_tensorflow_gpu()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_clustering = not bool(args.no_clustering)

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
                            f'Evaluating mcp {dataset_name}/{pool_type}/'
                            f'{sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                            f'(n={n_pool}, errors={n_errors}, clustering={apply_clustering})',
                        )
                        row = evaluate_mcp_pool(
                            pool,
                            dataset_name=dataset_name,
                            pool_type=pool_type,
                            error_ratio=error_ratio,
                            seed=seed,
                            sampling_mode=sampling_mode,
                            model=model,
                            cache_dir=cache_dir,
                            budget_ratios=RQ1_BUDGET_RATIOS,
                            apply_clustering=apply_clustering,
                            ratio_min=float(args.ratio_min),
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
    print('\n=== MCP baseline TRC summary (clustering + prioritization) ===')
    print(table_text)

    mode_tag = '_'.join(args.error_sampling_modes)
    runs_path = out_dir / f'rq1_mcp_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_mcp_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_mcp_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
