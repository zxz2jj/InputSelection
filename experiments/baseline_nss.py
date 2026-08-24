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

from training_models.load_data import load_cifar10, load_fmnist, load_svhn

RQ1_BUDGET_RATIOS = [0.01, 0.03, 0.05, 0.10]
RQ1_SEEDS = [0, 1, 2, 3, 4]
RQ1_POOL_TYPES = ['adversarial', 'transformation']
RQ1_ERROR_RATIOS = [0.10, 0.20]
RQ1_DATASETS = ['fmnist', 'cifar10', 'svhn']
ERROR_SAMPLING_MODES = ('random', 'high_conf')

NSS_SENSITIVE_FRACTION = 0.10
NSS_IDENTIFIER_SEED = 0
NSS_METHOD_NAME = 'nss'

# Official NSS benign_params / paper Table III. Each image uses exactly one op.
NSS_BENIGN_PARAMS = {
    'shift_x': (0.05, 0.15),
    'shift_y': (0.05, 0.15),
    'rotate': (5.0, 25.0),
    'scale': (0.8, 1.2),
    'shear': (15.0, 30.0),
    'contrast': (0.5, 1.5),
    'brightness': (0.5, 1.5),
}

# 3x3 Gaussian used by official blur_mode='easy' (approx. cv2 ksize=3).
_GAUSS_KERNEL_3 = np.array(
    [[1.0, 2.0, 1.0],
     [2.0, 4.0, 2.0],
     [1.0, 2.0, 1.0]],
    dtype=np.float32,
)
_GAUSS_KERNEL_3 /= float(_GAUSS_KERNEL_3.sum())

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        # conv2 + MaxPool: 12x12x20 = 2880
        'nss_feature_layer_index': 2,
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
        # last encoder ReLU (same as SETS -11): 2x2x512 = 2048
        'nss_feature_layer_index': -11,
    },
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
        # last residual ReLU before AvgPool (same as SETS -6): 4x4x512 = 8192
        'nss_feature_layer_index': -6,
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


def identifier_cache_name(dataset_name, layer_index, fraction, identifier_seed):
    frac_tag = f'{float(fraction):.2f}'.replace('.', 'p')
    return (
        f'nss_{dataset_name}_sens_l1_layer{int(layer_index)}'
        f'_f{frac_tag}_idseed{int(identifier_seed)}.npz'
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


def nss_selection_order(scores):
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    return np.argsort(-s, kind='mergesort')


def _sensitive_count(n_neurons, fraction=NSS_SENSITIVE_FRACTION):
    n = int(n_neurons)
    k = int(np.floor(float(fraction) * n))
    return max(1, min(k, n))


def top_sensitive_indices(neuron_scores, fraction=NSS_SENSITIVE_FRACTION):
    scores = np.asarray(neuron_scores, dtype=np.float64).reshape(-1)
    k = _sensitive_count(scores.shape[0], fraction)
    return np.argsort(scores, kind='mergesort')[-k:]


def _spatial_flatten(hid):
    hid = tf.cast(hid, tf.float32)
    b = tf.shape(hid)[0]
    return tf.reshape(hid, [b, -1])


def _make_feature_forward(model, layer_index):
    layer_index = int(layer_index)
    try:
        hidden_layer = model.layers[layer_index]
    except IndexError as exc:
        raise ValueError(
            f'nss_feature_layer_index={layer_index!r} out of range '
            f'(model has {len(model.layers)} layers)',
        ) from exc
    return tf.keras.Model(
        inputs=model.input,
        outputs=hidden_layer.output,
        name='nss_feature_forward',
    )


def extract_nss_features(data, model, nss_feature_layer_index, batch_size=64, desc='nss features'):
    forward = _make_feature_forward(model, nss_feature_layer_index)
    chunks = []
    n = int(len(data))
    bs = int(batch_size)
    for start in tqdm(range(0, n, bs), desc=desc):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(data[start:end])
        hid = forward(batch, training=False)
        flat = _spatial_flatten(hid).numpy().astype(np.float32, copy=False)
        chunks.append(flat)
    out = np.concatenate(chunks, axis=0)
    if out.shape[0] != n:
        raise ValueError('NSS feature length mismatch with inputs')
    return out


def _apply_affine(img, *, theta=0.0, tx=0.0, ty=0.0, shear=0.0, zx=1.0, zy=1.0):
    try:
        from tensorflow.keras.preprocessing.image import apply_affine_transform
    except ImportError:
        from keras.preprocessing.image import apply_affine_transform
    return apply_affine_transform(
        img,
        theta=float(theta),
        tx=float(tx),
        ty=float(ty),
        shear=float(shear),
        zx=float(zx),
        zy=float(zy),
        fill_mode='nearest',
    )


def _gaussian_blur3(img):
    x = np.asarray(img, dtype=np.float32)
    kernel = _GAUSS_KERNEL_3
    pad = 1
    if x.ndim != 3:
        raise ValueError('expected HWC image for blur')
    padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    out = np.empty_like(x)
    for c in range(x.shape[2]):
        ch = padded[:, :, c]
        acc = (
            kernel[0, 0] * ch[0:-2, 0:-2] + kernel[0, 1] * ch[0:-2, 1:-1] + kernel[0, 2] * ch[0:-2, 2:]
            + kernel[1, 0] * ch[1:-1, 0:-2] + kernel[1, 1] * ch[1:-1, 1:-1] + kernel[1, 2] * ch[1:-1, 2:]
            + kernel[2, 0] * ch[2:, 0:-2] + kernel[2, 1] * ch[2:, 1:-1] + kernel[2, 2] * ch[2:, 2:]
        )
        out[:, :, c] = acc
    return out


def _mutate_one(img, rng, params=NSS_BENIGN_PARAMS):
    x = np.asarray(img, dtype=np.float32)
    h, w = int(x.shape[0]), int(x.shape[1])
    op = int(rng.integers(0, 7))
    if op == 0:
        sx = float(rng.uniform(*params['shift_x']))
        sy = float(rng.uniform(*params['shift_y']))
        tx = float(rng.choice(np.array([-1.0, 1.0])) * sx * w)
        ty = float(rng.choice(np.array([-1.0, 1.0])) * sy * h)
        x = _apply_affine(x, tx=tx, ty=ty)
    elif op == 1:
        angle = float(rng.uniform(*params['rotate']))
        x = _apply_affine(x, theta=angle)
    elif op == 2:
        scale = float(rng.uniform(*params['scale']))
        x = _apply_affine(x, zx=scale, zy=scale)
    elif op == 3:
        shear = float(rng.uniform(*params['shear']))
        x = _apply_affine(x, shear=shear)
    elif op == 4:
        x = _gaussian_blur3(x)
    elif op == 5:
        factor = float(rng.uniform(*params['contrast']))
        mean = np.mean(x, axis=(0, 1), keepdims=True)
        x = (x - mean) * factor + mean
    else:
        factor = float(rng.uniform(*params['brightness']))
        x = x * factor
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def benign_mutate_images(images, rng):
    images = np.asarray(images, dtype=np.float32)
    out = np.empty_like(images)
    for i in range(int(images.shape[0])):
        out[i] = _mutate_one(images[i], rng)
    return out


def neuron_sensitivity_l1(clean_feat, mutated_feat):
    delta = np.abs(mutated_feat - clean_feat)
    return np.sum(delta, axis=0)


def sample_scores_l1(clean_feat, mutated_feat, snidx):
    delta = np.abs(mutated_feat[:, snidx] - clean_feat[:, snidx])
    return np.sum(delta, axis=1)


def detect_sensitive_neurons(
    x_test,
    model,
    nss_feature_layer_index,
    *,
    identifier_seed=NSS_IDENTIFIER_SEED,
    sensitive_fraction=NSS_SENSITIVE_FRACTION,
    batch_size=64,
):
    """Identifier on original test: accumulate L1 |N(x')-N(x)|, take top-k%."""
    rng = np.random.default_rng(int(identifier_seed))
    forward = _make_feature_forward(model, nss_feature_layer_index)
    n = int(len(x_test))
    bs = int(batch_size)
    acc_l1 = None
    n_neurons = None

    for start in tqdm(range(0, n, bs), desc=f'nss identifier layer {int(nss_feature_layer_index)}'):
        end = min(start + bs, n)
        clean_batch = np.asarray(x_test[start:end], dtype=np.float32)
        mutated_batch = benign_mutate_images(clean_batch, rng)
        clean_feat = _spatial_flatten(
            forward(tf.convert_to_tensor(clean_batch), training=False),
        ).numpy().astype(np.float32, copy=False)
        mutated_feat = _spatial_flatten(
            forward(tf.convert_to_tensor(mutated_batch), training=False),
        ).numpy().astype(np.float32, copy=False)
        if acc_l1 is None:
            n_neurons = int(clean_feat.shape[1])
            acc_l1 = np.zeros(n_neurons, dtype=np.float64)
        acc_l1 += neuron_sensitivity_l1(clean_feat, mutated_feat)

    if acc_l1 is None:
        raise ValueError('empty x_test for NSS identifier')

    snidx = top_sensitive_indices(acc_l1, sensitive_fraction)
    print(
        f'NSS SNIdx (L1) layer={int(nss_feature_layer_index)} '
        f'n_neurons={n_neurons} k={int(snidx.shape[0])}',
    )
    return {
        'sens_idx': snidx.astype(np.int64, copy=False),
        'n_neurons': int(n_neurons),
        'n_test': n,
    }


def build_or_load_sensitive_neurons(
    *,
    cache_dir,
    dataset_name,
    x_test,
    model,
    nss_feature_layer_index,
    identifier_seed=NSS_IDENTIFIER_SEED,
    sensitive_fraction=NSS_SENSITIVE_FRACTION,
    batch_size=64,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / identifier_cache_name(
        dataset_name, nss_feature_layer_index, sensitive_fraction, identifier_seed,
    )
    n_test = int(len(x_test))

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('sens_idx', 'n_neurons', 'layer_index', 'fraction', 'identifier_seed', 'n_test')
        missing = [key for key in required if key not in z.files]
        if missing:
            print(f'NSS SNIdx cache missing keys {missing}, recomputing -> {cache_path}')
        else:
            snidx = np.asarray(z['sens_idx'], dtype=np.int64).reshape(-1)
            n_neurons = int(np.asarray(z['n_neurons']).reshape(-1)[0])
            cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
            cached_frac = float(np.asarray(z['fraction']).reshape(-1)[0])
            cached_id_seed = int(np.asarray(z['identifier_seed']).reshape(-1)[0])
            cached_n_test = int(np.asarray(z['n_test']).reshape(-1)[0])
            expect_k = _sensitive_count(n_neurons, sensitive_fraction)
            reasons = []
            if cached_layer != int(nss_feature_layer_index):
                reasons.append(f'layer={cached_layer} vs {int(nss_feature_layer_index)}')
            if cached_id_seed != int(identifier_seed):
                reasons.append(f'id_seed={cached_id_seed} vs {int(identifier_seed)}')
            if cached_n_test != n_test:
                reasons.append(f'n_test={cached_n_test} vs {n_test}')
            if not np.isclose(cached_frac, float(sensitive_fraction), rtol=0.0, atol=1e-5):
                reasons.append(f'fraction={cached_frac} vs {float(sensitive_fraction)}')
            if snidx.shape[0] != expect_k:
                reasons.append(f'k={snidx.shape[0]} vs {expect_k}')
            if not reasons:
                print(f'Loaded NSS SNIdx from {cache_path}')
                return {
                    'sens_idx': snidx,
                    'n_neurons': n_neurons,
                    'n_test': n_test,
                }
            print(
                f'NSS SNIdx cache mismatch ({"; ".join(reasons)}), '
                f'recomputing -> {cache_path}',
            )

    detected = detect_sensitive_neurons(
        x_test,
        model,
        nss_feature_layer_index,
        identifier_seed=identifier_seed,
        sensitive_fraction=sensitive_fraction,
        batch_size=batch_size,
    )
    np.savez_compressed(
        cache_path,
        sens_idx=detected['sens_idx'],
        n_neurons=np.int32(detected['n_neurons']),
        n_test=np.int32(detected['n_test']),
        layer_index=np.int32(nss_feature_layer_index),
        fraction=np.float64(sensitive_fraction),
        identifier_seed=np.int32(identifier_seed),
    )
    print(f'Saved NSS SNIdx to {cache_path}')
    return detected


def compute_nss_pool_scores(
    data,
    model,
    nss_feature_layer_index,
    sens_idx,
    *,
    mutation_seed,
    batch_size=64,
):
    rng = np.random.default_rng(int(mutation_seed))
    forward = _make_feature_forward(model, nss_feature_layer_index)
    n = int(len(data))
    bs = int(batch_size)
    scores = np.empty(n, dtype=np.float64)
    snidx = np.asarray(sens_idx, dtype=np.int64).reshape(-1)

    for start in tqdm(range(0, n, bs), desc=f'nss pool scores layer {int(nss_feature_layer_index)}'):
        end = min(start + bs, n)
        clean_batch = np.asarray(data[start:end], dtype=np.float32)
        mutated_batch = benign_mutate_images(clean_batch, rng)
        clean_feat = _spatial_flatten(
            forward(tf.convert_to_tensor(clean_batch), training=False),
        ).numpy().astype(np.float32, copy=False)
        mutated_feat = _spatial_flatten(
            forward(tf.convert_to_tensor(mutated_batch), training=False),
        ).numpy().astype(np.float32, copy=False)
        scores[start:end] = sample_scores_l1(clean_feat, mutated_feat, snidx)
    return scores


def build_or_load_nss_pool_scores(
    *,
    cache_dir,
    cache_name,
    data,
    model,
    nss_feature_layer_index,
    sens_idx,
    mutation_seed,
    batch_size=64,
    force_recompute=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    n = int(len(data))
    snidx = np.asarray(sens_idx, dtype=np.int64).reshape(-1)

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        required = ('tnss_l1', 'sens_idx', 'layer_index', 'mutation_seed')
        missing = [key for key in required if key not in z.files]
        if missing:
            print(f'NSS pool score cache missing keys {missing}, recomputing -> {cache_path}')
        else:
            scores = np.asarray(z['tnss_l1'], dtype=np.float64).reshape(-1)
            cached_idx = np.asarray(z['sens_idx'], dtype=np.int64).reshape(-1)
            cached_layer = int(np.asarray(z['layer_index']).reshape(-1)[0])
            cached_mut = int(np.asarray(z['mutation_seed']).reshape(-1)[0])
            reasons = []
            if scores.shape[0] != n:
                reasons.append(f'n_score={scores.shape[0]} vs n={n}')
            if cached_layer != int(nss_feature_layer_index):
                reasons.append(f'layer={cached_layer} vs {int(nss_feature_layer_index)}')
            if cached_mut != int(mutation_seed):
                reasons.append(f'mutation_seed={cached_mut} vs {int(mutation_seed)}')
            if cached_idx.shape[0] != snidx.shape[0] or not np.array_equal(cached_idx, snidx):
                reasons.append('sens_idx mismatch')
            if not reasons:
                print(f'Loaded NSS pool scores from {cache_path}')
                return scores
            print(
                f'NSS pool score cache mismatch ({"; ".join(reasons)}), '
                f'recomputing -> {cache_path}',
            )

    scores = compute_nss_pool_scores(
        data,
        model,
        nss_feature_layer_index,
        snidx,
        mutation_seed=mutation_seed,
        batch_size=batch_size,
    )
    np.savez_compressed(
        cache_path,
        tnss_l1=scores.astype(np.float32, copy=False),
        sens_idx=snidx,
        layer_index=np.int32(nss_feature_layer_index),
        mutation_seed=np.int32(mutation_seed),
    )
    print(f'Saved NSS pool scores to {cache_path}')
    return scores


def evaluate_nss_pool(
    pool,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    seed,
    sampling_mode,
    model,
    x_test,
    nss_feature_layer_index,
    cache_dir,
    budget_ratios,
    sensitive_fraction=NSS_SENSITIVE_FRACTION,
    identifier_seed=NSS_IDENTIFIER_SEED,
    mutation_seed=None,
    batch_size=64,
    force_recompute=False,
    return_selection=False,
    snidx_pack=None,
):
    """
    RQ entry: paper TNSScore (L1 sum on last-encoder SNIdx) prefix TRC.

    Returns a single result row (method='nss').
    """
    storage = build_pool_storage(pool)
    n_pool = len(storage)
    mut_seed = int(seed) if mutation_seed is None else int(mutation_seed)
    tag = pool_cache_tag(dataset_name, pool_type, error_ratio, seed, sampling_mode)

    if snidx_pack is None:
        snidx_pack = build_or_load_sensitive_neurons(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            x_test=x_test,
            model=model,
            nss_feature_layer_index=nss_feature_layer_index,
            identifier_seed=identifier_seed,
            sensitive_fraction=sensitive_fraction,
            batch_size=batch_size,
            force_recompute=force_recompute,
        )
    sens_idx = np.asarray(snidx_pack['sens_idx'], dtype=np.int64).reshape(-1)

    scores = build_or_load_nss_pool_scores(
        cache_dir=cache_dir,
        cache_name=f'nss_{tag}_l1_scores.npz',
        data=pool['data'],
        model=model,
        nss_feature_layer_index=nss_feature_layer_index,
        sens_idx=sens_idx,
        mutation_seed=mut_seed,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    order = nss_selection_order(scores)
    selected_by_budget = selected_by_budget_from_order(order, budget_ratios, n_pool)
    curve = compute_selection_curves(storage, order, budget_ratios=budget_ratios)
    ratio_to_trc = _trc_by_budget_dict(budget_ratios, curve)
    row = {
        'method': NSS_METHOD_NAME,
        'dataset': dataset_name,
        'pool_type': pool_type,
        'sampling_mode': sampling_mode,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'num_total': int(n_pool),
        'total_errors': int(curve['total_errors']),
        'nss_feature_layer_index': int(nss_feature_layer_index),
        'sensitive_fraction': float(sensitive_fraction),
        'n_neurons': int(snidx_pack['n_neurons']),
        'n_sensitive': int(sens_idx.shape[0]),
        'identifier_seed': int(identifier_seed),
        'mutation_seed': int(mut_seed),
        'score_mode': 'l1',
        'trc_by_budget': ratio_to_trc,
    }
    if return_selection:
        row['selection_order'] = [int(i) for i in np.asarray(order).reshape(-1)]
        row['selected_by_budget'] = _jsonable_selected_by_budget(selected_by_budget)
        row['sens_idx'] = [int(i) for i in sens_idx]
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
        description='NSS baseline TRC (paper L1 TNSScore) on sampled pools.',
    )
    parser.add_argument('--datasets', nargs='+', default=RQ1_DATASETS, choices=sorted(RQ1_DATASETS))
    parser.add_argument('--pool-types', nargs='+', default=RQ1_POOL_TYPES, choices=RQ1_POOL_TYPES)
    parser.add_argument('--error-ratios', nargs='+', type=float, default=RQ1_ERROR_RATIOS)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(RQ1_SEEDS))
    parser.add_argument(
        '--error-sampling-modes',
        nargs='+',
        default=list(ERROR_SAMPLING_MODES),
        choices=list(ERROR_SAMPLING_MODES),
        help='error sampling modes to evaluate',
    )
    parser.add_argument('--sampled-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--output-dir', default=str(_EXP_DIR / 'results' / 'rq1'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sensitive-fraction', type=float, default=NSS_SENSITIVE_FRACTION)
    parser.add_argument('--identifier-seed', type=int, default=NSS_IDENTIFIER_SEED)
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
        _x_train, _y_train, x_test, _y_test = cfg['loader']()
        layer_index = int(cfg['nss_feature_layer_index'])
        cache_dir = _EXP_DIR / 'cache_files' / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f'NSS identifier on original test n={len(x_test)} '
            f'layer={layer_index} fraction={float(args.sensitive_fraction):.2f} '
            f'id_seed={int(args.identifier_seed)} score=L1',
        )
        snidx_pack = build_or_load_sensitive_neurons(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            x_test=x_test,
            model=model,
            nss_feature_layer_index=layer_index,
            identifier_seed=int(args.identifier_seed),
            sensitive_fraction=float(args.sensitive_fraction),
            batch_size=int(args.batch_size),
            force_recompute=args.force_recompute,
        )

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
                            f'Evaluating nss {dataset_name}/{pool_type}/'
                            f'{sampling_mode}/ratio={error_ratio:.2f}/seed={seed} '
                            f'(n={n_pool}, errors={n_errors}, mutation_seed={int(seed)})',
                        )
                        row = evaluate_nss_pool(
                            pool,
                            dataset_name=dataset_name,
                            pool_type=pool_type,
                            error_ratio=error_ratio,
                            seed=seed,
                            sampling_mode=sampling_mode,
                            model=model,
                            x_test=x_test,
                            nss_feature_layer_index=layer_index,
                            cache_dir=cache_dir,
                            budget_ratios=RQ1_BUDGET_RATIOS,
                            sensitive_fraction=float(args.sensitive_fraction),
                            identifier_seed=int(args.identifier_seed),
                            mutation_seed=int(seed),
                            batch_size=int(args.batch_size),
                            force_recompute=args.force_recompute,
                            snidx_pack=snidx_pack,
                        )
                        run_rows.append(row)
                        trc_str = ', '.join(
                            f'{b:.0%}={_lookup_trc(row["trc_by_budget"], b):.4f}'
                            for b in RQ1_BUDGET_RATIOS
                        )
                        print(f'  {row["method"]} TRC: {trc_str}')

    summary_rows = aggregate_results(run_rows)
    table_text = format_results_table(summary_rows, args.seeds)
    print('\n=== NSS baseline TRC summary (paper L1 TNSScore) ===')
    print(table_text)

    mode_tag = '_'.join(args.error_sampling_modes)
    runs_path = out_dir / f'rq1_nss_trc_runs_{mode_tag}.json'
    summary_path = out_dir / f'rq1_nss_trc_summary_{mode_tag}.json'
    table_path = out_dir / f'rq1_nss_trc_summary_table_{mode_tag}.md'
    runs_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True), encoding='utf-8')
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding='utf-8')
    table_path.write_text(table_text + '\n', encoding='utf-8')
    print(f'\nSaved runs: {runs_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved table: {table_path}')


if __name__ == '__main__':
    main()
