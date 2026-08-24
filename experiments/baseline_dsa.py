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

from risk_scoring import _hidden_to_flat_batch, _probs_from_model_output_tensor
from training_models.load_data import load_cifar10, load_fmnist, load_svhn

RQ1_BUDGET_RATIOS = [0.01, 0.03, 0.05, 0.10]
RQ1_SEEDS = [0, 1, 2, 3, 4]
RQ1_POOL_TYPES = ['adversarial', 'transformation']
RQ1_ERROR_RATIOS = [0.10, 0.20]
RQ1_DATASETS = ['fmnist', 'cifar10', 'svhn']
ERROR_SAMPLING_MODES = ('random', 'high_conf')

DSA_METHOD_NAME = 'dsa'
# ICSE 2019 SADL: dist_b near 0 would explode the ratio.
DSA_DIST_EPS = 1e-12

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        # last-hidden (same index as DATIS)
        'dsa_feature_layer_index': -4,
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
        'dsa_feature_layer_index': -5,
    },
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
        'dsa_feature_layer_index': -4,
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


def train_ats_cache_name(layer_index):
    return f'dsa_train_ats_layer{int(layer_index)}.npz'


def dsa_scores_cache_name(tag, layer_index):
    return f'dsa_{tag}_scores_layer{int(layer_index)}.npz'


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


def extract_ats_and_pred_classes(data, model, dsa_feature_layer_index, batch_size=64):
    """Last-hidden activation traces + predicted class (argmax softmax)."""
    layer_index = int(dsa_feature_layer_index)
    try:
        hidden_layer = model.layers[layer_index]
    except IndexError as exc:
        raise ValueError(
            f'dsa_feature_layer_index={layer_index!r} out of range '
            f'(model has {len(model.layers)} layers)',
        ) from exc

    forward = tf.keras.Model(
        inputs=model.input,
        outputs=[hidden_layer.output, model.output],
        name='dsa_ats_pred_forward',
    )
    hid_chunks = []
    pred_chunks = []
    n = len(data)
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc=f'dsa ats+pred layer {layer_index}'):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        hid, raw_out = forward(batch, training=False)
        flat = _hidden_to_flat_batch(hid, hidden_layer).numpy().astype(np.float32, copy=False)
        probs = _probs_from_model_output_tensor(raw_out).numpy()
        pred = np.argmax(probs, axis=-1).astype(np.int64, copy=False).reshape(-1)
        hid_chunks.append(flat)
        pred_chunks.append(pred)
    ats = np.concatenate(hid_chunks, axis=0)
    pred_classes = np.concatenate(pred_chunks, axis=0)
    if ats.shape[0] != n or pred_classes.shape[0] != n:
        raise ValueError('DSA AT/pred length mismatch with inputs')
    return ats, pred_classes


def compute_dsa_scores(pool_ats, pool_pred, train_ats, train_pred, dist_eps=DSA_DIST_EPS):
    """
    Official SADL DSA: for x with predicted class c_x,
      x_a = nearest same-class train AT of x (Euclidean)
      x_b = nearest other-class train AT of x_a (not of x)
      DSA(x) = dist(x, x_a) / dist(x_a, x_b)
    """
    pool_ats = np.asarray(pool_ats, dtype=np.float64)
    train_ats = np.asarray(train_ats, dtype=np.float64)
    pool_pred = np.asarray(pool_pred, dtype=np.int64).reshape(-1)
    train_pred = np.asarray(train_pred, dtype=np.int64).reshape(-1)
    if pool_ats.ndim != 2 or train_ats.ndim != 2:
        raise ValueError('activation traces must be (n, dim)')
    n_pool = int(pool_ats.shape[0])
    if pool_pred.shape[0] != n_pool:
        raise ValueError('pool_pred length mismatch with pool_ats')
    if train_pred.shape[0] != train_ats.shape[0]:
        raise ValueError('train_pred length mismatch with train_ats')
    if pool_ats.shape[1] != train_ats.shape[1]:
        raise ValueError('pool/train AT dim mismatch')

    scores = np.full(n_pool, np.nan, dtype=np.float64)
    eps = float(dist_eps)
    labels = np.unique(pool_pred)
    for c in labels:
        mask = pool_pred == int(c)
        query = pool_ats[mask]
        if query.shape[0] == 0:
            continue
        same_idx = np.flatnonzero(train_pred == int(c))
        other_idx = np.flatnonzero(train_pred != int(c))
        if same_idx.size == 0:
            scores[mask] = np.inf
            continue

        nn_same = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
        nn_same.fit(train_ats[same_idx])
        dist_a, ind_local = nn_same.kneighbors(query, n_neighbors=1, return_distance=True)
        dist_a = dist_a.reshape(-1)
        xa = train_ats[same_idx[ind_local.reshape(-1)]]

        if other_idx.size == 0:
            dist_b = np.full(xa.shape[0], eps, dtype=np.float64)
        else:
            nn_other = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
            nn_other.fit(train_ats[other_idx])
            dist_b, _ = nn_other.kneighbors(xa, n_neighbors=1, return_distance=True)
            dist_b = dist_b.reshape(-1)
        scores[mask] = dist_a / np.maximum(dist_b, eps)
    return scores


def dsa_selection_order(dsa_scores):
    scores = np.asarray(dsa_scores, dtype=np.float64).reshape(-1)
    n = scores.shape[0]
    # Higher DSA first; NaN last. +inf ranks first.
    rank_key = np.nan_to_num(
        scores,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.finfo(np.float64).max,
    )
    order = np.lexsort((np.arange(n, dtype=np.int64), -rank_key))
    return order.astype(np.int64, copy=False)


def build_or_load_train_ats(
    *,
    cache_dir,
    dsa_feature_layer_index,
    x_train,
    model,
    batch_size=64,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    layer_index = int(dsa_feature_layer_index)
    cache_path = cache_dir / train_ats_cache_name(layer_index)
    n_train = int(len(x_train))

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('ats', 'pred_classes', 'layer_index', 'n_train')
        missing = [key for key in required if key not in z.files]
        reasons = []
        if missing:
            reasons.append(f'missing keys {missing}')
        else:
            ats = np.asarray(z['ats'], dtype=np.float32)
            pred = np.asarray(z['pred_classes'], dtype=np.int64).reshape(-1)
            cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
            cached_n = int(np.asarray(z['n_train']).reshape(-1)[0])
            if ats.ndim != 2 or ats.shape[0] != n_train:
                reasons.append(f'n_ats={ats.shape[0]} vs n={n_train}')
            if pred.shape[0] != n_train:
                reasons.append(f'n_pred={pred.shape[0]} vs n={n_train}')
            if cached_n != n_train:
                reasons.append(f'n_train={cached_n} vs {n_train}')
            if cached_layer != layer_index:
                reasons.append(f'layer={cached_layer} vs {layer_index}')
            if not reasons:
                print(f'Loaded DSA train ATs from {cache_path}')
                return ats, pred
        print(
            f'DSA train AT cache mismatch ({"; ".join(reasons)}), '
            f'recomputing -> {cache_path}',
        )

    ats, pred = extract_ats_and_pred_classes(
        x_train, model, layer_index, batch_size=batch_size,
    )
    np.savez_compressed(
        cache_path,
        ats=ats,
        pred_classes=pred.astype(np.int64, copy=False),
        layer_index=np.int32(layer_index),
        n_train=np.int32(n_train),
    )
    print(f'Saved DSA train ATs to {cache_path}')
    return ats, pred


def build_or_load_dsa_scores(
    *,
    cache_dir,
    cache_name,
    data,
    model,
    train_ats,
    train_pred,
    dsa_feature_layer_index,
    batch_size=64,
    force_recompute=False,
    dist_eps=DSA_DIST_EPS,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n = int(len(data))
    layer_index = int(dsa_feature_layer_index)

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('dsa_score', 'sort_ix', 'pred_classes', 'layer_index', 'n_pool')
        missing = [key for key in required if key not in z.files]
        reasons = []
        if missing:
            reasons.append(f'missing keys {missing}')
        else:
            scores = np.asarray(z['dsa_score'], dtype=np.float64).reshape(-1)
            order = np.asarray(z['sort_ix'], dtype=np.int64).reshape(-1)
            pred = np.asarray(z['pred_classes'], dtype=np.int64).reshape(-1)
            cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
            cached_n = int(np.asarray(z['n_pool']).reshape(-1)[0])
            if scores.shape[0] != n:
                reasons.append(f'n_score={scores.shape[0]} vs n={n}')
            if order.shape[0] != n:
                reasons.append(f'rank len={order.shape[0]} vs n={n}')
            if pred.shape[0] != n:
                reasons.append(f'n_pred={pred.shape[0]} vs n={n}')
            if cached_n != n:
                reasons.append(f'n_pool={cached_n} vs {n}')
            if cached_layer != layer_index:
                reasons.append(f'layer={cached_layer} vs {layer_index}')
            if not reasons:
                print(f'Loaded DSA scores from {cache_path}')
                return scores, order, pred
        print(
            f'DSA score cache mismatch ({"; ".join(reasons)}), '
            f'recomputing -> {cache_path}',
        )

    pool_ats, pool_pred = extract_ats_and_pred_classes(
        data, model, layer_index, batch_size=batch_size,
    )
    print(f'  DSA scoring n_pool={n} n_train={int(train_ats.shape[0])} layer={layer_index}')
    scores = compute_dsa_scores(
        pool_ats, pool_pred, train_ats, train_pred, dist_eps=dist_eps,
    )
    order = dsa_selection_order(scores)
    np.savez_compressed(
        cache_path,
        dsa_score=scores.astype(np.float32, copy=False),
        sort_ix=np.asarray(order, dtype=np.int64),
        pred_classes=pool_pred.astype(np.int64, copy=False),
        layer_index=np.int32(layer_index),
        n_pool=np.int32(n),
    )
    print(f'Saved DSA scores to {cache_path}')
    return scores, order, pool_pred


def evaluate_dsa_pool(
    pool,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    seed,
    sampling_mode,
    model,
    x_train,
    dsa_feature_layer_index,
    cache_dir,
    budget_ratios,
    batch_size=64,
    force_recompute=False,
    dist_eps=DSA_DIST_EPS,
    return_selection=False,
):
    """
    RQ entry: DSA ranking (higher surprise first), prefix TRC.

    Class buckets use model predicted labels for train and pool (not ground truth).
    """
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)
    layer_index = int(dsa_feature_layer_index)

    train_ats, train_pred = build_or_load_train_ats(
        cache_dir=cache_dir,
        dsa_feature_layer_index=layer_index,
        x_train=x_train,
        model=model,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    scores, order, _pool_pred = build_or_load_dsa_scores(
        cache_dir=cache_dir,
        cache_name=dsa_scores_cache_name(tag, layer_index),
        data=pool['data'],
        model=model,
        train_ats=train_ats,
        train_pred=train_pred,
        dsa_feature_layer_index=layer_index,
        batch_size=batch_size,
        force_recompute=force_recompute,
        dist_eps=dist_eps,
    )
    selected_by_budget = selected_by_budget_from_order(order, budget_ratios, n_pool)
    curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
    ratio_to_trc = _trc_by_budget_dict(budget_ratios, curve)
    row = {
        'method': DSA_METHOD_NAME,
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_errors': int(curve['total_errors']),
        'dsa_feature_layer_index': int(layer_index),
        'trc_by_budget': ratio_to_trc,
    }
    if return_selection:
        row['selection_order'] = [int(i) for i in np.asarray(order).reshape(-1)]
        row['selected_by_budget'] = _jsonable_selected_by_budget(selected_by_budget)
        row['dsa_score'] = [float(v) for v in np.asarray(scores, dtype=np.float64).reshape(-1)]
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
        description='DSA baseline TRC (surprise adequacy, last-hidden AT) on sampled pools.',
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
    parser.add_argument('--force-recompute', action='store_true')
    return parser.parse_args()


def _drop_heavy_fields(run_row):
    skip = {'selection_order', 'selected_by_budget', 'dsa_score'}
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
        layer_index = int(cfg['dsa_feature_layer_index'])
        x_train, _y_train, _x_test, _y_test = cfg['loader']()

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
                            f'Evaluating dsa {dataset_name}/{pool_type}/'
                            f'{sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                            f'(n={n_pool}, errors={n_errors}, layer={layer_index})',
                        )
                        row = evaluate_dsa_pool(
                            pool,
                            dataset_name=dataset_name,
                            pool_type=pool_type,
                            error_ratio=error_ratio,
                            seed=seed,
                            sampling_mode=sampling_mode,
                            model=model,
                            x_train=x_train,
                            dsa_feature_layer_index=layer_index,
                            cache_dir=cache_dir,
                            budget_ratios=RQ1_BUDGET_RATIOS,
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
    print('\n=== DSA baseline TRC summary (surprise adequacy) ===')
    print(table_text)

    mode_tag = '_'.join(args.error_sampling_modes)
    runs_path = out_dir / f'rq1_dsa_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_dsa_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_dsa_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
