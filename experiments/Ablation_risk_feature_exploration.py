import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent
_SEL_DIR = _REPO_ROOT / 'selection_method'
for _p in (_REPO_ROOT, _SEL_DIR, _EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

tf.get_logger().setLevel('ERROR')
logging.getLogger().setLevel(logging.WARNING)

from training_models.load_data import load_cifar10, load_fmnist, load_svhn
from risk_scoring import (
    _batch_pred_class_is_min_distance,
    _hidden_to_flat_batch,
    build_or_load_class_prototypes_dict,
    compute_trc_by_budget,
    image_data_augmentation,
    keras_logits_fn,
    risk_scoring_function,
)

CACHE_ROOT = _EXP_DIR / 'cache_files'

EXPLORE_SAMPLING_MODES = ('random', 'high_conf')
EXPLORE_POOL_TYPES = ('adversarial', 'transformation')
EXPLORE_ERROR_RATIOS = (0.10, 0.20)
EXPLORE_SEEDS = (0,)
EXPLORE_DATASETS = ('fmnist', 'cifar10', 'svhn')

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        'distance_layer_index': -4,
        'consistency_layer_indices': [-10, -8, -6, -4],
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
        'distance_layer_index': -5,
        'consistency_layer_indices': [-19, -15, -11, -5],
    },
    'svhn': {
        'loader': load_svhn,
        'model_path': _REPO_ROOT / 'models' / 'resnet18_svhn' / 'tf_model.h5',
        'distance_layer_index': -4,  
        'consistency_layer_indices': [-40, -23, -14, -4],
    },
}

# Continuous features (higher value => higher risk).
CONTINUOUS_RISK_FEATURE_KEYS = (
    'prediction_entropy',
    'energy_score',
    'top12_margin',
    'deepgini',
    'max_softmax_prob',
    'neg_logit_l2_norm',
    'stability_class_change_rate',
    'stability_max_prob_variance',
    'stability_mean_kl',
    'stability_entropy_variance',
    'stability_pred_prob_drop',
    'hidden_aug_variance',
    'dist_pred_class_prototype',
    'dist_ratio_pred_to_nearest_other_prototype',
    'hidden_distance_margin_risk',
    'dist_nearest_non_pred_proto_risk',
    'dist_layer_inconsistency',
    'cosine_dist_pred_proto',
    'mahalanobis_dist_pred_class',
)

# --- Feature combination search (pure risk-ranking TRC; no pseudo-labels) ---
SEARCH_BUDGET_RATIOS = (0.01, 0.03, 0.05, 0.10)
SEARCH_BUDGET_WEIGHTS = {0.01: 2.0, 0.03: 1.5, 0.05: 1.0, 0.10: 1.0}
SEARCH_LAMBDA_HIGH = 0.3
SEARCH_EPS_STOP = 0.002
SEARCH_DELTA_PRUNE = 0.002
SEARCH_DELTA_FAMILY = 0.01
SEARCH_RANDOM_FLOOR_EPS = 0.03
SEARCH_INIT_KEYS = ('deepgini',)
SEARCH_DEV_SEEDS = (0, 1, 2)
SEARCH_TEST_SEEDS = (3, 4)

FEATURE_FAMILY = {
    'prediction_entropy': 'softmax',
    'deepgini': 'softmax',
    'max_softmax_prob': 'softmax',
    'top12_margin': 'softmax',
    'energy_score': 'logit',
    'neg_logit_l2_norm': 'logit',
    'dist_pred_class_prototype': 'geo',
    'dist_ratio_pred_to_nearest_other_prototype': 'geo',
    'hidden_distance_margin_risk': 'geo',
    'cosine_dist_pred_proto': 'geo',
    'stability_max_prob_variance': 'stab_var',
    'stability_entropy_variance': 'stab_var',
    'stability_class_change_rate': 'stab_flip',
    'stability_mean_kl': 'stab_kl',
    'stability_pred_prob_drop': 'stab_drop',
    'mahalanobis_dist_pred_class': 'mahal',
    'dist_layer_inconsistency': 'layer',
    'hidden_aug_variance': 'aug_var',
    'dist_nearest_non_pred_proto_risk': 'near_other',
}


def _as_label_vector(arr):
    out = np.asarray(arr)
    if out.ndim > 1:
        out = np.argmax(out, axis=-1)
    return out.reshape(-1).astype(np.int64, copy=False)


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
    return f'explore_{dataset_name}_{pool_type}_r{ratio_pct:02d}_s{int(seed)}_{sampling_mode}'


def _batch_proto_geometry_features(hid_flat, pred, prototypes, class_means, class_inv_covs):
    """L2/cosine/mahalanobis geometry features; all oriented higher => higher risk."""
    hid_flat = np.asarray(hid_flat, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    B = hid_flat.shape[0]
    C = prototypes.shape[0]
    valid_proto = ~np.isnan(prototypes).any(axis=1)
    eps = 1e-8

    dist_pred_proto = np.full(B, np.nan, dtype=np.float64)
    dist_ratio = np.full(B, np.nan, dtype=np.float64)
    hidden_distance_margin_risk = np.full(B, np.nan, dtype=np.float64)
    dist_nearest_non_pred_proto_risk = np.full(B, np.nan, dtype=np.float64)
    cosine_dist_pred_proto = np.full(B, np.nan, dtype=np.float64)
    mahalanobis_dist_pred_class = np.full(B, np.nan, dtype=np.float64)

    for i in range(B):
        h = hid_flat[i]
        p = int(pred[i])
        if p < 0 or p >= C or not valid_proto[p]:
            continue

        d_all = np.linalg.norm(prototypes - h, axis=1)
        d_p = float(d_all[p])
        dist_pred_proto[i] = d_p

        mask = valid_proto & (np.arange(C, dtype=np.int64) != p)
        if np.any(mask):
            d_min_other = float(np.min(d_all[mask]))
            dist_ratio[i] = d_p / (d_min_other + eps)
            hidden_distance_margin_risk[i] = d_p - d_min_other
            dist_nearest_non_pred_proto_risk[i] = -d_min_other

        proto_p = prototypes[p]
        norm_h = np.linalg.norm(h)
        norm_p = np.linalg.norm(proto_p)
        if norm_h > eps and norm_p > eps:
            cosine_sim = float(np.dot(h, proto_p) / (norm_h * norm_p))
            cosine_dist_pred_proto[i] = 1.0 - cosine_sim

        if class_means is not None and class_inv_covs is not None and p in class_means:
            mu = class_means[p]
            inv_cov = class_inv_covs[p]
            diff = h - mu
            quad = float(diff @ inv_cov @ diff)
            if quad >= 0.0 and np.isfinite(quad):
                mahalanobis_dist_pred_class[i] = np.sqrt(quad)

    return {
        'dist_pred_class_prototype': dist_pred_proto.astype(np.float32),
        'dist_ratio_pred_to_nearest_other_prototype': dist_ratio.astype(np.float32),
        'hidden_distance_margin_risk': hidden_distance_margin_risk.astype(np.float32),
        'dist_nearest_non_pred_proto_risk': dist_nearest_non_pred_proto_risk.astype(np.float32),
        'cosine_dist_pred_proto': cosine_dist_pred_proto.astype(np.float32),
        'mahalanobis_dist_pred_class': mahalanobis_dist_pred_class.astype(np.float32),
    }


def build_or_load_mahalanobis_stats(
    model,
    train_data,
    train_labels,
    layer_index,
    dataset_name,
    *,
    batch_size=64,
    cache_root=CACHE_ROOT,
    force_recompute=False,
    cov_eps=1e-4,
):
    layer_index = int(layer_index)
    cache_dir = Path(cache_root) / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'mahalanobis_stats_layer_{layer_index}.npz'

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        num_classes = int(z['num_classes'])
        class_means = {}
        class_inv_covs = {}
        for c in range(num_classes):
            key_m = f'mean_{c}'
            key_i = f'inv_cov_{c}'
            if key_m in z.files and key_i in z.files:
                class_means[c] = np.asarray(z[key_m], dtype=np.float64)
                class_inv_covs[c] = np.asarray(z[key_i], dtype=np.float64)
        print(f'Loaded Mahalanobis stats from {cache_path} ({len(class_means)} classes)')
        return class_means, class_inv_covs

    y_flat = _as_label_vector(train_labels)
    num_classes = int(y_flat.max()) + 1
    hidden_layer = model.layers[layer_index]
    forward = tf.keras.Model(
        inputs=model.input,
        outputs=hidden_layer.output,
        name='mahalanobis_hidden_forward',
    )

    by_class = {c: [] for c in range(num_classes)}
    for start in tqdm(range(0, len(train_data), batch_size), desc=f'mahalanobis stats layer {layer_index}'):
        end = min(start + batch_size, len(train_data))
        bx = train_data[start:end]
        by = y_flat[start:end]
        bt = tf.convert_to_tensor(bx)
        hid = forward(bt, training=False)
        logits_raw = model(bt, training=False)
        if isinstance(logits_raw, (list, tuple)):
            logits_raw = logits_raw[0]
        probs = tf.nn.softmax(tf.cast(logits_raw, tf.float32), axis=-1)
        pred = tf.argmax(probs, axis=-1).numpy()
        hid_flat = _hidden_to_flat_batch(hid, hidden_layer).numpy().astype(np.float64, copy=False)
        correct = pred == by
        for i in range(hid_flat.shape[0]):
            if not correct[i]:
                continue
            c = int(by[i])
            by_class[c].append(hid_flat[i])

    class_means = {}
    class_inv_covs = {}
    save_dict = {'num_classes': np.int32(num_classes), 'layer_index': np.int32(layer_index)}
    dim = None
    for c in range(num_classes):
        samples = by_class[c]
        if len(samples) < 2:
            continue
        arr = np.stack(samples, axis=0)
        dim = arr.shape[1]
        mu = arr.mean(axis=0)
        cov = np.cov(arr, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        cov = cov + cov_eps * np.eye(cov.shape[0], dtype=np.float64)
        inv_cov = np.linalg.inv(cov)
        class_means[c] = mu
        class_inv_covs[c] = inv_cov
        save_dict[f'mean_{c}'] = mu.astype(np.float32)
        save_dict[f'inv_cov_{c}'] = inv_cov.astype(np.float32)

    if dim is not None:
        save_dict['hidden_dim'] = np.int32(dim)
    np.savez_compressed(cache_path, **save_dict)
    print(f'Saved Mahalanobis stats to {cache_path} ({len(class_means)} classes with stats)')
    return class_means, class_inv_covs


def get_all_risk_features(
    data_without_labelling,
    model,
    prototypes_by_layer_map,
    distance_feature_layer_index,
    consistency_feature_layer_indices,
    class_means,
    class_inv_covs,
    logits_layer_index=-2,
    batch_size=16,
    num_augmentations=5,
    augment_repeats_per_transform=3,
):
    if augment_repeats_per_transform is None or int(augment_repeats_per_transform) < 1:
        raise ValueError('augment_repeats_per_transform must be >= 1')
    augment_repeats_per_transform = int(augment_repeats_per_transform)

    logit_fn = keras_logits_fn(model, logits_layer_index=logits_layer_index)
    consistency_feature_layer_indices = [int(i) for i in consistency_feature_layer_indices]
    distance_feature_layer_index = int(distance_feature_layer_index)
    hidden_layer_indices = list(
        dict.fromkeys([distance_feature_layer_index] + consistency_feature_layer_indices),
    )

    hidden_layers = {}
    hidden_prototypes = {}
    for idx in hidden_layer_indices:
        if idx not in prototypes_by_layer_map:
            raise ValueError(f'prototypes_by_layer_map missing layer_index={idx}')
        hidden_layers[idx] = model.layers[idx]
        hidden_prototypes[idx] = np.asarray(prototypes_by_layer_map[idx], dtype=np.float32)

    logits_layer = model.layers[logits_layer_index]
    base_forward = tf.keras.Model(
        inputs=model.input,
        outputs=[hidden_layers[idx].output for idx in hidden_layer_indices] + [logits_layer.output],
        name='explore_risk_base_forward',
    )
    aug_hidden_forward = tf.keras.Model(
        inputs=model.input,
        outputs=[hidden_layers[idx].output for idx in consistency_feature_layer_indices],
        name='explore_risk_aug_hidden_forward',
    )

    collectors = {k: [] for k in CONTINUOUS_RISK_FEATURE_KEYS}
    pred_classes_list = []

    eps = tf.constant(1e-8, tf.float32)
    n_aug = int(num_augmentations) * augment_repeats_per_transform

    for start in tqdm(range(0, len(data_without_labelling), batch_size), desc='all risk features'):
        batch = data_without_labelling[start:start + batch_size]
        batch_tensor = tf.convert_to_tensor(batch)

        forward_outs = base_forward(batch_tensor, training=False)
        hid_outs = forward_outs[:-1]
        logits = tf.cast(forward_outs[-1], tf.float32)
        probs = tf.nn.softmax(logits, axis=-1)

        pred_classes = tf.argmax(probs, axis=-1)
        pred_np = pred_classes.numpy()

        logit_l2 = tf.sqrt(tf.reduce_sum(tf.square(logits), axis=-1))
        max_prob = tf.reduce_max(probs, axis=-1)
        top2_vals, _ = tf.nn.top_k(probs, k=2)
        deepgini = 1.0 - tf.reduce_sum(tf.square(probs), axis=-1)

        hid_flat_by_layer = {}
        for idx, hid in zip(hidden_layer_indices, hid_outs):
            hid_flat_by_layer[idx] = _hidden_to_flat_batch(hid, hidden_layers[idx]).numpy()

        geom = _batch_proto_geometry_features(
            hid_flat_by_layer[distance_feature_layer_index],
            pred_np,
            hidden_prototypes[distance_feature_layer_index],
            class_means,
            class_inv_covs,
        )

        layer_flags = []
        for idx in consistency_feature_layer_indices:
            layer_flags.append(
                _batch_pred_class_is_min_distance(
                    hid_flat_by_layer[idx],
                    pred_np,
                    hidden_prototypes[idx],
                ),
            )
        layer_flags = np.stack(layer_flags, axis=1)
        dist_layer_inconsistency = 1.0 - np.mean(layer_flags, axis=1)

        aug_prob_list = []
        aug_entropy_list = []
        aug_max_prob_list = []
        aug_hidden_by_layer = {idx: [] for idx in consistency_feature_layer_indices}

        for tid in range(1, int(num_augmentations) + 1):
            for _ in range(augment_repeats_per_transform):
                x_aug = image_data_augmentation(batch_tensor, transform_id=tid)
                lg = tf.cast(logit_fn(x_aug), tf.float32)
                aug_p = tf.nn.softmax(lg, axis=-1)
                aug_prob_list.append(aug_p)
                aug_entropy_list.append(-tf.reduce_sum(tf.math.xlogy(aug_p, aug_p), axis=-1))
                aug_max_prob_list.append(tf.reduce_max(aug_p, axis=-1))
                aug_hid_outs = aug_hidden_forward(x_aug, training=False)
                for layer_idx, hid in zip(consistency_feature_layer_indices, aug_hid_outs):
                    flat = _hidden_to_flat_batch(hid, hidden_layers[layer_idx]).numpy()
                    aug_hidden_by_layer[layer_idx].append(flat)

        aug_probs = tf.stack(aug_prob_list, axis=1)
        aug_entropies = tf.stack(aug_entropy_list, axis=1)
        aug_max_probs = tf.stack(aug_max_prob_list, axis=1)

        aug_preds = tf.argmax(aug_probs, axis=-1)
        pred_orig = tf.expand_dims(pred_classes, 1)
        stability_flip = tf.reduce_mean(tf.cast(tf.not_equal(pred_orig, aug_preds), tf.float32), axis=1)
        stability_maxvar = tf.math.reduce_variance(tf.reduce_max(aug_probs, axis=-1), axis=1)

        p_orig = probs[:, tf.newaxis, :]
        log_p = tf.math.log(aug_probs + eps)
        log_q = tf.math.log(p_orig + eps)
        stability_mkl = tf.reduce_mean(tf.reduce_sum(aug_probs * (log_p - log_q), axis=-1), axis=1)

        stability_entropy_variance = tf.math.reduce_variance(aug_entropies, axis=1)
        stability_pred_prob_drop = max_prob - tf.reduce_mean(aug_max_probs, axis=1)

        layer_variances = []
        for layer_idx in consistency_feature_layer_indices:
            stacked = np.stack(aug_hidden_by_layer[layer_idx], axis=1)
            layer_variances.append(np.mean(np.var(stacked, axis=1), axis=1))
        hidden_aug_variance = np.mean(np.stack(layer_variances, axis=1), axis=1).astype(np.float32)

        batch_out = {
            'prediction_entropy': np.asarray(
                (-tf.reduce_sum(tf.math.xlogy(probs, probs), axis=-1)).numpy(), dtype=np.float32,
            ).reshape(-1),
            'energy_score': np.asarray((-tf.math.reduce_logsumexp(logits, axis=-1)).numpy(), dtype=np.float32).reshape(-1),
            'top12_margin': np.asarray((1.0 - (top2_vals[:, 0] - top2_vals[:, 1])).numpy(), dtype=np.float32).reshape(-1),
            'deepgini': np.asarray(deepgini.numpy(), dtype=np.float32).reshape(-1),
            'max_softmax_prob': np.asarray((1.0 - max_prob).numpy(), dtype=np.float32).reshape(-1),
            'neg_logit_l2_norm': np.asarray((-logit_l2).numpy(), dtype=np.float32).reshape(-1),
            'stability_class_change_rate': np.asarray(stability_flip.numpy(), dtype=np.float32).reshape(-1),
            'stability_max_prob_variance': np.asarray(stability_maxvar.numpy(), dtype=np.float32).reshape(-1),
            'stability_mean_kl': np.asarray(stability_mkl.numpy(), dtype=np.float32).reshape(-1),
            'stability_entropy_variance': np.asarray(stability_entropy_variance.numpy(), dtype=np.float32).reshape(-1),
            'stability_pred_prob_drop': np.asarray(stability_pred_prob_drop.numpy(), dtype=np.float32).reshape(-1),
            'hidden_aug_variance': hidden_aug_variance.reshape(-1),
            'dist_layer_inconsistency': dist_layer_inconsistency.astype(np.float32).reshape(-1),
            **{k: geom[k].reshape(-1) for k in geom},
        }

        for key in CONTINUOUS_RISK_FEATURE_KEYS:
            collectors[key].append(batch_out[key])
        pred_classes_list.append(pred_np.reshape(-1))

    out = {k: np.concatenate(collectors[k]) for k in CONTINUOUS_RISK_FEATURE_KEYS}
    out['pred_classes'] = np.concatenate(pred_classes_list).astype(np.int64)
    return out


def build_or_load_all_risk_features(
    cache_root,
    cache_name,
    data,
    model,
    prototypes_by_layer_map,
    distance_feature_layer_index,
    consistency_feature_layer_indices,
    class_means,
    class_inv_covs,
    logits_layer_index=-2,
    batch_size=16,
    num_augmentations=5,
    augment_repeats_per_transform=3,
    force_recompute=False,
):
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        out = {k: np.asarray(z[k]) for k in z.files}
        print(f'Loaded all risk features from {cache_path}')
        return out

    out = get_all_risk_features(
        data_without_labelling=data,
        model=model,
        prototypes_by_layer_map=prototypes_by_layer_map,
        distance_feature_layer_index=distance_feature_layer_index,
        consistency_feature_layer_indices=consistency_feature_layer_indices,
        class_means=class_means,
        class_inv_covs=class_inv_covs,
        logits_layer_index=logits_layer_index,
        batch_size=batch_size,
        num_augmentations=num_augmentations,
        augment_repeats_per_transform=augment_repeats_per_transform,
    )
    np.savez_compressed(cache_path, **out)
    print(f'Saved all risk features to {cache_path}')
    return out


def cohens_d(error_vals, correct_vals):
    err = np.asarray(error_vals, dtype=np.float64).reshape(-1)
    cor = np.asarray(correct_vals, dtype=np.float64).reshape(-1)
    err = err[np.isfinite(err)]
    cor = cor[np.isfinite(cor)]
    if err.size < 2 or cor.size < 2:
        return np.nan
    mean_e, mean_c = float(err.mean()), float(cor.mean())
    var_e, var_c = float(err.var(ddof=1)), float(cor.var(ddof=1))
    n_e, n_c = err.size, cor.size
    pooled = np.sqrt(((n_e - 1) * var_e + (n_c - 1) * var_c) / (n_e + n_c - 2))
    if pooled <= 0.0:
        return np.nan
    return (mean_e - mean_c) / pooled


def compute_univariate_metrics(feature_values, is_error):
    """ metrics for one feature on one pool (higher feature => higher risk)."""
    f = np.asarray(feature_values, dtype=np.float64).reshape(-1)
    y = np.asarray(is_error, dtype=bool).reshape(-1)
    if f.shape[0] != y.shape[0]:
        raise ValueError('feature_values and is_error length mismatch')

    mask = np.isfinite(f)
    f = f[mask]
    y = y[mask]
    n = f.size
    n_error = int(np.sum(y))
    n_correct = int(n - n_error)
    prevalence = (n_error / n) if n > 0 else np.nan

    out = {
        'n': n,
        'n_error': n_error,
        'n_correct': n_correct,
        'prevalence': prevalence,
        'spearman_rho': np.nan,
        'spearman_p': np.nan,
        'auroc': np.nan,
        'auprc': np.nan,
        'cohens_d': np.nan,
        'direction_ok': False,
    }
    if n < 3 or n_error == 0 or n_correct == 0:
        return out

    y_int = y.astype(np.int32)
    err_vals = f[y]
    cor_vals = f[~y]

    rho, p_val = spearmanr(f, y_int)
    out['spearman_rho'] = float(rho)
    out['spearman_p'] = float(p_val)
    out['cohens_d'] = float(cohens_d(err_vals, cor_vals))

    try:
        out['auroc'] = float(roc_auc_score(y_int, f))
    except ValueError:
        out['auroc'] = np.nan

    try:
        out['auprc'] = float(average_precision_score(y_int, f))
    except ValueError:
        out['auprc'] = np.nan

    out['direction_ok'] = bool(
        np.isfinite(out['spearman_rho'])
        and out['spearman_rho'] > 0
        and np.isfinite(out['auroc'])
        and out['auroc'] >= 0.5,
    )
    return out


def _fmt(x, digits=4):
    if x is None or not np.isfinite(x):
        return 'nan'
    return f'{float(x):.{digits}f}'


def _mean_finite(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def _aggregate_metric_rows(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, items in sorted(groups.items()):
        entry = {group_keys[i]: key[i] for i in range(len(group_keys))}
        entry['n_pools'] = len(items)
        for metric in ('spearman_rho', 'auroc', 'auprc', 'cohens_d'):
            entry[f'mean_{metric}'] = _mean_finite([it[metric] for it in items])
        entry['direction_ok_rate'] = _mean_finite([1.0 if it['direction_ok'] else 0.0 for it in items])
        summary.append(entry)
    return summary


def _md_cell(value):
    """Escape characters that break GFM tables / unintended emphasis."""
    text = str(value).replace('\n', ' ').replace('|', '\\|')
    return text


def _markdown_table(headers, rows):
    safe_headers = [_md_cell(h) for h in headers]
    lines = [
        '| ' + ' | '.join(safe_headers) + ' |',
        '| ' + ' | '.join(['---'] * len(safe_headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(_md_cell(c) for c in row) + ' |')
    return '\n'.join(lines)


def analyze_pool_features(feature_map, is_error):
    rows = []
    for feature_key in CONTINUOUS_RISK_FEATURE_KEYS:
        if feature_key not in feature_map:
            continue
        metrics = compute_univariate_metrics(feature_map[feature_key], is_error)
        rows.append({'feature': feature_key, **metrics})
    return rows


def iter_cached_pools(
    feature_cache_dir,
    sampled_root,
    datasets,
    pool_types,
    error_ratios,
    seeds,
    sampling_modes,
):
    """Yield one dict per cached pool (features aligned with sampled pool labels)."""
    feature_cache_dir = Path(feature_cache_dir)
    sampled_root = Path(sampled_root)
    missing = []

    for dataset_name in datasets:
        for pool_type in pool_types:
            for error_ratio in error_ratios:
                for sampling_mode in sampling_modes:
                    for seed in seeds:
                        tag = pool_cache_tag(
                            dataset_name, pool_type, error_ratio, seed, sampling_mode,
                        )
                        feat_path = feature_cache_dir / dataset_name / f'all_risk_features_{tag}.npz'
                        if not feat_path.is_file():
                            missing.append(str(feat_path))
                            continue
                        try:
                            pool = load_sampled_pool(
                                sampled_root, dataset_name, pool_type,
                                error_ratio, seed, sampling_mode,
                            )
                        except FileNotFoundError as exc:
                            missing.append(str(exc))
                            continue
                        z = np.load(feat_path)
                        feature_map = {k: np.asarray(z[k]) for k in z.files}
                        n_pool = len(pool['is_error'])
                        first_key = next(iter(feature_map))
                        if np.asarray(feature_map[first_key]).reshape(-1).shape[0] != n_pool:
                            raise ValueError(
                                f'feature/label length mismatch for pool {tag}: '
                                f'features={np.asarray(feature_map[first_key]).reshape(-1).shape[0]}, '
                                f'pool={n_pool}',
                            )
                        yield {
                            'dataset': dataset_name,
                            'tag': tag,
                            'pool_type': pool_type,
                            'error_ratio': float(error_ratio),
                            'seed': int(seed),
                            'sampling_mode': sampling_mode,
                            'feature_map': feature_map,
                            'is_error': np.asarray(pool['is_error'], dtype=bool).reshape(-1),
                        }

    if missing:
        print(f'Warning: skipped {len(missing)} missing pool/feature entries')


def error_only_feature_matrix(
    feature_map,
    is_error,
    feature_keys=CONTINUOUS_RISK_FEATURE_KEYS,
):
    """Feature matrix restricted to wrongly-predicted (error) samples in one pool."""
    mask = np.asarray(is_error, dtype=bool).reshape(-1)
    n = mask.shape[0]
    cols = []
    for key in feature_keys:
        if key not in feature_map:
            raise KeyError(f'missing feature {key!r}')
        col = np.asarray(feature_map[key], dtype=np.float64).reshape(-1)
        if col.shape[0] != n:
            raise ValueError(f'feature length mismatch for {key!r}')
        cols.append(col[mask])
    if not cols:
        return np.empty((0, len(feature_keys)), dtype=np.float64)
    return np.column_stack(cols)


def spearman_correlation_matrix(feature_matrix):
    """Pairwise Spearman rho; NaN pairs use samples finite in both columns."""
    x = np.asarray(feature_matrix, dtype=np.float64)
    n, d = x.shape
    corr = np.eye(d, dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            mask = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
            if int(np.sum(mask)) < 3:
                rho = np.nan
            else:
                rho, _ = spearmanr(x[mask, i], x[mask, j])
            corr[i, j] = rho
            corr[j, i] = rho
    return corr


def hierarchical_feature_clusters(corr, min_abs_corr=0.85, linkage_method='average'):
    """
    Cluster features by Spearman correlation.
    Distance = 1 - |rho|; cut so clusters have |rho| >= min_abs_corr within group (approx.).
    """
    corr = np.asarray(corr, dtype=np.float64)
    d = corr.shape[0]
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    dist = 0.5 * (dist + dist.T)
    condensed = squareform(dist, checks=False)
    z_link = linkage(condensed, method=linkage_method)
    max_dist = max(1.0 - float(min_abs_corr), 1e-6)
    cluster_ids = fcluster(z_link, t=max_dist, criterion='distance')
    return z_link, cluster_ids.astype(np.int64)


def _short_feature_label(name, max_len=28):
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + '...'


def plot_correlation_heatmap(corr, feature_names, out_path, title=None):
    """Heatmap with fixed axis order equal to feature_names (no cluster reordering)."""
    names = [_short_feature_label(n) for n in feature_names]
    n = len(names)
    fig_size = max(8.0, 0.45 * n + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(
        np.asarray(corr, dtype=np.float64),
        vmin=-1.0,
        vmax=1.0,
        cmap='RdBu_r',
        aspect='auto',
    )
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman rho')
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _cluster_members_table(cluster_ids, feature_names):
    clusters = {}
    for feat, cid in zip(feature_names, cluster_ids):
        clusters.setdefault(int(cid), []).append(feat)
    rows = []
    for cid in sorted(clusters):
        members = sorted(clusters[cid])
        rows.append([cid, len(members), ', '.join(members)])
    return rows


def _high_correlation_pair_rows(corr, feature_names, min_abs_corr=0.85):
    d = len(feature_names)
    rows = []
    for i in range(d):
        for j in range(i + 1, d):
            rho = corr[i, j]
            if not np.isfinite(rho) or abs(rho) < min_abs_corr:
                continue
            rows.append([feature_names[i], feature_names[j], _fmt(rho)])
    rows.sort(key=lambda r: abs(float(r[2])), reverse=True)
    return rows


def _correlation_matrix_table(corr, feature_names):
    headers = ['feature'] + [_short_feature_label(n, 18) for n in feature_names]
    rows = []
    for i, name in enumerate(feature_names):
        rows.append([_short_feature_label(name, 24)] + [_fmt(corr[i, j]) for j in range(len(feature_names))])
    return headers, rows


def _write_correlation_report_md(
    report_path,
    *,
    title,
    intro_lines,
    corr,
    feature_names,
    cluster_ids,
    min_abs_corr,
    heatmap_name,
):
    cluster_rows = _cluster_members_table(cluster_ids, feature_names)
    pair_rows = _high_correlation_pair_rows(corr, feature_names, min_abs_corr=min_abs_corr)
    mat_headers, mat_rows = _correlation_matrix_table(corr, feature_names)

    parts = [f'# {title}', ''] + intro_lines + [
        f'- Cluster cut: **|Spearman rho| >= {min_abs_corr}** (linkage distance <= {1 - min_abs_corr:.2f})',
        f'- Heatmap: `{heatmap_name}`',
        '',
        '## Hierarchical clusters',
        '',
        _markdown_table(['cluster_id', 'size', 'members'], cluster_rows),
        '',
        f'## High-correlation pairs (|rho| >= {min_abs_corr})',
        '',
    ]
    if pair_rows:
        parts.append(_markdown_table(['feature_a', 'feature_b', 'spearman_rho'], pair_rows))
    else:
        parts.append('_No pairs at or above threshold._')
    parts.extend([
        '',
        '## Full Spearman correlation matrix',
        '',
        _markdown_table(mat_headers, mat_rows),
        '',
    ])
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(parts), encoding='utf-8')
    return cluster_rows, pair_rows


def run_correlation_analysis(
    feature_cache_dir,
    sampled_root,
    datasets,
    pool_types,
    error_ratios,
    seeds,
    sampling_modes,
    cache_root=CACHE_ROOT,
    min_abs_corr=0.85,
    linkage_method='average',
    min_errors=3,
):
    """
    Write correlation outputs under cache_files/{dataset}/correlation/:
      per_pool/{tag}/correlation_heatmap.png|correlation_report.md|...
      mean_correlation_heatmap.png
      mean_correlation_report.md
      correlation_summary.md
      correlation_meta.json
    """
    cache_root = Path(cache_root)
    feature_names = list(CONTINUOUS_RISK_FEATURE_KEYS)

    # Group pools by dataset first so each dataset writes to its own directory.
    pools_by_dataset = {ds: [] for ds in datasets}
    skipped = []
    for pool in iter_cached_pools(
        feature_cache_dir,
        sampled_root,
        datasets,
        pool_types,
        error_ratios,
        seeds,
        sampling_modes,
    ):
        n_errors = int(np.sum(pool['is_error']))
        if n_errors < int(min_errors):
            skipped.append({
                'tag': pool['tag'],
                'n_errors': n_errors,
                'reason': f'n_errors < {min_errors}',
            })
            continue
        pools_by_dataset[pool['dataset']].append(pool)

    all_meta = []
    for dataset_name in datasets:
        pool_list = pools_by_dataset.get(dataset_name) or []
        if not pool_list:
            print(f'[{dataset_name}] no pools for correlation, skip')
            continue

        dataset_out = cache_root / dataset_name / 'correlation'
        per_pool_dir = dataset_out / 'per_pool'
        per_pool_dir.mkdir(parents=True, exist_ok=True)

        corr_list = []
        pool_meta_rows = []
        for pool in pool_list:
            matrix = error_only_feature_matrix(pool['feature_map'], pool['is_error'])
            n_errors = int(matrix.shape[0])
            corr = spearman_correlation_matrix(matrix)
            _z_link, cluster_ids = hierarchical_feature_clusters(
                corr, min_abs_corr=min_abs_corr, linkage_method=linkage_method,
            )

            pool_out = per_pool_dir / pool['tag']
            pool_out.mkdir(parents=True, exist_ok=True)
            heatmap_path = pool_out / 'correlation_heatmap.png'
            plot_correlation_heatmap(
                corr,
                feature_names,
                heatmap_path,
                title=(
                    f'{pool["tag"]}: errors-only Spearman ({n_errors} errors)'
                ),
            )

            np.savez_compressed(
                pool_out / 'spearman_correlation.npz',
                feature_names=np.array(feature_names, dtype=object),
                spearman_corr=corr.astype(np.float32),
                cluster_ids=cluster_ids.astype(np.int32),
                n_errors=np.int32(n_errors),
                min_abs_corr=np.float32(min_abs_corr),
                dataset=np.array(pool['dataset']),
                pool_type=np.array(pool['pool_type']),
                error_ratio=np.float32(pool['error_ratio']),
                seed=np.int32(pool['seed']),
                sampling_mode=np.array(pool['sampling_mode']),
            )

            intro = [
                f'- Dataset: **{pool["dataset"]}**',
                f'- Pool type: **{pool["pool_type"]}**',
                f'- Sampling mode: **{pool["sampling_mode"]}**',
                f'- Error ratio: **{pool["error_ratio"]:.2f}**',
                f'- Seed: **{pool["seed"]}**',
                f'- Error samples: **{n_errors}** (errors only)',
                f'- Features: **{len(feature_names)}**',
            ]
            _write_correlation_report_md(
                pool_out / 'correlation_report.md',
                title=f'{pool["tag"]}: error-sample feature correlation',
                intro_lines=intro,
                corr=corr,
                feature_names=feature_names,
                cluster_ids=cluster_ids,
                min_abs_corr=min_abs_corr,
                heatmap_name='correlation_heatmap.png',
            )

            corr_list.append(corr)
            pool_meta_rows.append({
                'dataset': pool['dataset'],
                'tag': pool['tag'],
                'pool_type': pool['pool_type'],
                'sampling_mode': pool['sampling_mode'],
                'error_ratio': pool['error_ratio'],
                'seed': pool['seed'],
                'n_errors': n_errors,
                'n_clusters': int(len(set(int(c) for c in cluster_ids))),
                'report': str((pool_out / 'correlation_report.md').relative_to(dataset_out)),
                'heatmap': str((pool_out / 'correlation_heatmap.png').relative_to(dataset_out)),
            })
            print(f'[{pool["tag"]}] errors={n_errors} -> {pool_out}')

        corr_stack = np.stack(corr_list, axis=0)
        corr_mean = np.nanmean(corr_stack, axis=0)
        np.fill_diagonal(corr_mean, 1.0)

        _z_link_mean, cluster_ids_mean = hierarchical_feature_clusters(
            corr_mean, min_abs_corr=min_abs_corr, linkage_method=linkage_method,
        )
        summary_heatmap = dataset_out / 'mean_correlation_heatmap.png'
        plot_correlation_heatmap(
            corr_mean,
            feature_names,
            summary_heatmap,
            title=f'{dataset_name}: mean Spearman over {len(corr_list)} pools (errors only)',
        )

        np.savez_compressed(
            dataset_out / 'mean_spearman_correlation.npz',
            feature_names=np.array(feature_names, dtype=object),
            spearman_corr_mean=corr_mean.astype(np.float32),
            cluster_ids=cluster_ids_mean.astype(np.int32),
            n_pools=np.int32(len(corr_list)),
            min_abs_corr=np.float32(min_abs_corr),
        )

        summary_report = dataset_out / 'mean_correlation_report.md'
        intro = [
            f'- Dataset: **{dataset_name}**',
            f'- Averaged pools: **{len(corr_list)}** (element-wise mean of Spearman matrices)',
            f'- Sample subset: **errors only** in each pool',
            f'- Features: **{len(feature_names)}**',
        ]
        cluster_rows, _pair_rows = _write_correlation_report_md(
            summary_report,
            title=f'{dataset_name}: mean error-sample feature correlation',
            intro_lines=intro,
            corr=corr_mean,
            feature_names=feature_names,
            cluster_ids=cluster_ids_mean,
            min_abs_corr=min_abs_corr,
            heatmap_name=summary_heatmap.name,
        )

        dataset_skipped = [s for s in skipped if s['tag'].startswith(f'explore_{dataset_name}_')]
        summary_lines = [
            f'# {dataset_name}: Risk Feature Correlation (errors only, per pool)',
            '',
            'Each pool: Spearman rho on **is_error** samples only; hierarchical clustering '
            f'(average linkage, cut |rho| >= {min_abs_corr}).',
            'Summary: element-wise **mean** of this dataset\'s per-pool correlation matrices.',
            '',
            f'- Pools analyzed: **{len(pool_meta_rows)}**',
            f'- Pools skipped (too few errors): **{len(dataset_skipped)}**',
            f'- Mean heatmap: [`{summary_heatmap.name}`]({summary_heatmap.name})',
            f'- Mean report: [`{summary_report.name}`]({summary_report.name})',
            '',
            '## Per-pool reports',
            '',
            '| pool_type | sampling_mode | error_ratio | seed | n_errors | report |',
            '| --- | --- | --- | --- | --- | --- |',
        ]
        for row in sorted(
            pool_meta_rows,
            key=lambda r: (r['pool_type'], r['sampling_mode'], r['error_ratio'], r['seed']),
        ):
            summary_lines.append(
                f'| {row["pool_type"]} | {row["sampling_mode"]} | '
                f'{row["error_ratio"]:.2f} | {row["seed"]} | {row["n_errors"]} | '
                f'`{row["report"]}` |',
            )
        summary_lines.extend([
            '',
            '## Mean-matrix clusters',
            '',
            _markdown_table(['cluster_id', 'size', 'members'], cluster_rows),
            '',
        ])

        meta = {
            'dataset': dataset_name,
            'min_abs_corr': float(min_abs_corr),
            'linkage_method': linkage_method,
            'min_errors': int(min_errors),
            'n_pools_analyzed': len(pool_meta_rows),
            'n_pools_skipped': len(dataset_skipped),
            'n_clusters_mean': int(len(set(int(c) for c in cluster_ids_mean))),
            'pools': pool_meta_rows,
            'skipped': dataset_skipped,
            'mean_heatmap': summary_heatmap.name,
            'mean_report': summary_report.name,
        }
        meta_path = dataset_out / 'correlation_meta.json'
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        summary_lines.extend(['', f'Metadata: `{meta_path.name}`', ''])
        summary_path = dataset_out / 'correlation_summary.md'
        summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
        print(f'[{dataset_name}] summary mean over {len(corr_list)} pools -> {dataset_out}')
        all_meta.append(meta)

    if not all_meta:
        raise RuntimeError(
            'No pools produced correlation output. Run --phase compute first, '
            f'or lower --corr-min-errors (current {min_errors}).',
        )
    return all_meta


def run_univariate_analysis(
    feature_cache_dir,
    sampled_root,
    datasets,
    pool_types,
    error_ratios,
    seeds,
    sampling_modes,
    output_dir=CACHE_ROOT,
):
    feature_cache_dir = Path(feature_cache_dir)
    sampled_root = Path(sampled_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    missing = []

    for dataset_name in datasets:
        for pool_type in pool_types:
            for error_ratio in error_ratios:
                for sampling_mode in sampling_modes:
                    for seed in seeds:
                        tag = pool_cache_tag(
                            dataset_name, pool_type, error_ratio, seed, sampling_mode,
                        )
                        feat_path = feature_cache_dir / dataset_name / f'all_risk_features_{tag}.npz'
                        if not feat_path.is_file():
                            missing.append(str(feat_path))
                            continue

                        try:
                            pool = load_sampled_pool(
                                sampled_root, dataset_name, pool_type,
                                error_ratio, seed, sampling_mode,
                            )
                        except FileNotFoundError as exc:
                            missing.append(str(exc))
                            continue

                        z = np.load(feat_path)
                        feature_map = {k: np.asarray(z[k]) for k in z.files}
                        is_error = pool['is_error']

                        for item in analyze_pool_features(feature_map, is_error):
                            detail_rows.append({
                                'dataset': dataset_name,
                                'pool_type': pool_type,
                                'sampling_mode': sampling_mode,
                                'error_ratio': float(error_ratio),
                                'seed': int(seed),
                                **item,
                            })

    if not detail_rows:
        raise RuntimeError(
            'No feature caches found for analysis. Run --phase compute first.\n'
            + 'Missing examples:\n  ' + '\n  '.join(missing[:10]),
        )

    summary_rows = _aggregate_metric_rows(
        detail_rows,
        group_keys=('dataset', 'pool_type', 'sampling_mode', 'error_ratio', 'feature'),
    )

    json_path = output_dir / 'univariate_detail.json'
    summary_json_path = output_dir / 'univariate_summary.json'
    json_path.write_text(json.dumps(detail_rows, indent=2), encoding='utf-8')
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding='utf-8')

    md_parts = [
        '# Univariate Risk Feature Analysis',
        '',
        'Metrics: Spearman ρ vs `is_error`, AUROC, AUPRC (baseline ≈ error prevalence), '
        "Cohen's d (error vs correct). All features oriented **higher => higher risk**.",
        '',
        f'- Pools analyzed: **{len({(r["dataset"], r["pool_type"], r["sampling_mode"], r["error_ratio"], r["seed"]) for r in detail_rows})}**',
        f'- Detail JSON: `{json_path.relative_to(_EXP_DIR)}`',
        '',
    ]

    for sampling_mode in sampling_modes:
        md_parts.append(f'## Sampling mode: `{sampling_mode}`')
        md_parts.append('')
        for dataset_name in datasets:
            for pool_type in pool_types:
                section_rows = [
                    r for r in summary_rows
                    if r['dataset'] == dataset_name
                    and r['pool_type'] == pool_type
                    and r['sampling_mode'] == sampling_mode
                ]
                if not section_rows:
                    continue

                md_parts.append(f'### {dataset_name} / {pool_type}')
                md_parts.append('')

                for error_ratio in error_ratios:
                    ratio_rows = [r for r in section_rows if abs(r['error_ratio'] - error_ratio) < 1e-9]
                    if not ratio_rows:
                        continue
                    ratio_rows = sorted(ratio_rows, key=lambda r: r['feature'])
                    md_parts.append(f'#### error_ratio = {error_ratio:.2f} (mean over seeds)')
                    md_parts.append('')
                    table_rows = []
                    for r in ratio_rows:
                        prev_rows = [
                            d for d in detail_rows
                            if d['dataset'] == dataset_name
                            and d['pool_type'] == pool_type
                            and d['sampling_mode'] == sampling_mode
                            and abs(d['error_ratio'] - error_ratio) < 1e-9
                            and d['feature'] == r['feature']
                        ]
                        mean_prev = _mean_finite([d['prevalence'] for d in prev_rows])
                        table_rows.append([
                            r['feature'],
                            _fmt(r['mean_spearman_rho']),
                            _fmt(r['mean_auroc']),
                            _fmt(r['mean_auprc']),
                            _fmt(mean_prev),
                            _fmt(r['mean_cohens_d']),
                            _fmt(r['direction_ok_rate'], 2),
                            int(r['n_pools']),
                        ])
                    md_parts.append(_markdown_table(
                        [
                            'feature', 'spearman_rho', 'auroc', 'auprc',
                            'auprc_baseline', 'cohens_d', 'direction_ok_rate', 'n_seeds',
                        ],
                        table_rows,
                    ))
                    md_parts.append('')

    md_path = output_dir / 'univariate_summary.md'
    md_path.write_text('\n'.join(md_parts), encoding='utf-8')

    if missing:
        miss_path = output_dir / 'univariate_missing_caches.txt'
        miss_path.write_text('\n'.join(missing), encoding='utf-8')
        print(f'Warning: {len(missing)} missing pool/feature entries (see {miss_path})')

    print(f'Wrote univariate detail JSON: {json_path}')
    print(f'Wrote univariate summary JSON: {summary_json_path}')
    print(f'Wrote univariate markdown table: {md_path}')
    return detail_rows, summary_rows, md_path


def run_feature_compute(
    sampled_root,
    datasets,
    pool_types,
    error_ratios,
    seeds,
    sampling_modes,
    cache_root=CACHE_ROOT,
    batch_size=16,
    force_recompute=False,
):
    sampled_root = Path(sampled_root)
    cache_root = Path(cache_root)

    for dataset_name in datasets:
        cfg = DATASET_CONFIG[dataset_name]
        model_path = Path(cfg['model_path'])
        if not model_path.is_file():
            raise FileNotFoundError(f'missing model: {model_path}')

        print(f'=== compute features: {dataset_name} ===')
        model = tf.keras.models.load_model(model_path, compile=False)
        loader = cfg['loader']
        x_train, y_train, _, _ = loader()
        y_train = _as_label_vector(y_train)

        layer_indices = list(
            dict.fromkeys([cfg['distance_layer_index']] + list(cfg['consistency_layer_indices'])),
        )
        prototypes = build_or_load_class_prototypes_dict(
            model,
            train_data=x_train,
            train_labels=y_train,
            layer_indices=layer_indices,
            dataset_name=dataset_name,
            batch_size=64,
            force_recompute=force_recompute,
        )
        class_means, class_inv_covs = build_or_load_mahalanobis_stats(
            model,
            x_train,
            y_train,
            cfg['distance_layer_index'],
            dataset_name,
            cache_root=cache_root,
            force_recompute=force_recompute,
        )

        dataset_cache = cache_root / dataset_name
        dataset_cache.mkdir(parents=True, exist_ok=True)

        for pool_type in pool_types:
            for error_ratio in error_ratios:
                for sampling_mode in sampling_modes:
                    for seed in seeds:
                        try:
                            pool = load_sampled_pool(
                                sampled_root, dataset_name, pool_type,
                                error_ratio, seed, sampling_mode,
                            )
                        except FileNotFoundError as exc:
                            print(f'  skip missing pool: {exc}')
                            continue

                        tag = pool_cache_tag(
                            dataset_name, pool_type, error_ratio, seed, sampling_mode,
                        )
                        print(
                            f'  pool {tag}: n={len(pool["data"])}, '
                            f'errors={int(np.sum(pool["is_error"]))}',
                        )
                        build_or_load_all_risk_features(
                            cache_root=dataset_cache,
                            cache_name=f'all_risk_features_{tag}.npz',
                            data=pool['data'],
                            model=model,
                            prototypes_by_layer_map=prototypes,
                            distance_feature_layer_index=cfg['distance_layer_index'],
                            consistency_feature_layer_indices=cfg['consistency_layer_indices'],
                            class_means=class_means,
                            class_inv_covs=class_inv_covs,
                            batch_size=batch_size,
                            force_recompute=force_recompute,
                        )


def _budget_key(b):
    return f'{float(b):.2f}'


def _trc_dict_from_result(trc_out, budgets=SEARCH_BUDGET_RATIOS):
    ratios = [float(x) for x in np.asarray(trc_out['budget_ratio']).reshape(-1)]
    vals = [float(x) for x in np.asarray(trc_out['trc']).reshape(-1)]
    by_ratio = {round(r, 4): v for r, v in zip(ratios, vals)}
    return {float(b): float(by_ratio.get(round(float(b), 4), np.nan)) for b in budgets}


def pool_trc_for_feature_keys(feature_map, is_error, feature_keys, budgets=SEARCH_BUDGET_RATIOS):
    keys = [k for k in feature_keys if k in feature_map]
    if not keys:
        return {float(b): np.nan for b in budgets}
    scored = risk_scoring_function(feature_map, feature_keys=keys)
    return _trc_dict_from_result(
        compute_trc_by_budget(scored['risk_score'], is_error, list(budgets)),
        budgets=budgets,
    )


def evaluate_feature_subset(
    pools,
    feature_keys,
    *,
    budgets=SEARCH_BUDGET_RATIOS,
    budget_weights=None,
    lambda_high=SEARCH_LAMBDA_HIGH,
):
    """Pure risk-ranking TRC aggregation + primary score J(S).

    Primary: weighted mean TRC on random pools.
    Secondary: lambda * weighted mean TRC on high_conf pools.
    """
    if budget_weights is None:
        budget_weights = SEARCH_BUDGET_WEIGHTS
    keys = list(feature_keys)
    high_lists = {float(b): [] for b in budgets}
    all_lists = {float(b): [] for b in budgets}
    random_lists = {float(b): [] for b in budgets}
    per_pool = []

    for pool in pools:
        trc_map = pool_trc_for_feature_keys(
            pool['feature_map'], pool['is_error'], keys, budgets=budgets,
        )
        per_pool.append({
            'dataset': pool['dataset'],
            'pool_type': pool['pool_type'],
            'error_ratio': pool['error_ratio'],
            'seed': pool['seed'],
            'sampling_mode': pool['sampling_mode'],
            'trc': { _budget_key(b): trc_map[float(b)] for b in budgets },
        })
        for b in budgets:
            bf = float(b)
            all_lists[bf].append(trc_map[bf])
            if pool['sampling_mode'] == 'high_conf':
                high_lists[bf].append(trc_map[bf])
            elif pool['sampling_mode'] == 'random':
                random_lists[bf].append(trc_map[bf])

    trc_high = {float(b): _mean_finite(high_lists[float(b)]) for b in budgets}
    trc_all = {float(b): _mean_finite(all_lists[float(b)]) for b in budgets}
    trc_random = {float(b): _mean_finite(random_lists[float(b)]) for b in budgets}

    j_random = 0.0
    j_high = 0.0
    for b in budgets:
        bf = float(b)
        w = float(budget_weights[bf])
        j_random += w * (trc_random[bf] if np.isfinite(trc_random[bf]) else 0.0)
        j_high += w * (trc_high[bf] if np.isfinite(trc_high[bf]) else 0.0)
    j_score = j_random + float(lambda_high) * j_high

    return {
        'feature_keys': keys,
        'J': float(j_score),
        'J_random_term': float(j_random),
        'J_high_term': float(j_high),
        'trc_high_conf': trc_high,
        'trc_all': trc_all,
        'trc_random': trc_random,
        'n_pools': len(pools),
        'n_high_conf': sum(1 for p in pools if p['sampling_mode'] == 'high_conf'),
        'n_random': sum(1 for p in pools if p['sampling_mode'] == 'random'),
        'per_pool': per_pool,
    }


def evaluate_feature_subset_global(
    pools_by_dataset,
    feature_keys,
    *,
    budgets=SEARCH_BUDGET_RATIOS,
    budget_weights=None,
    lambda_high=SEARCH_LAMBDA_HIGH,
    dataset_order=None,
):
    """Global J from direct pool-level mean (all Dev pools concatenated).

    Also returns per-dataset metrics for reporting and the random floor check.
    """
    if budget_weights is None:
        budget_weights = SEARCH_BUDGET_WEIGHTS
    if not pools_by_dataset:
        raise ValueError('pools_by_dataset is empty')
    if dataset_order is None:
        dataset_order = sorted(pools_by_dataset.keys())
    else:
        dataset_order = [d for d in dataset_order if d in pools_by_dataset]
        if not dataset_order:
            raise ValueError('dataset_order empty after filtering')

    keys = list(feature_keys)
    per_dataset = {}
    all_pools = []
    for ds in dataset_order:
        pools = pools_by_dataset[ds]
        if not pools:
            raise ValueError(f'no pools for dataset={ds}')
        per_dataset[ds] = evaluate_feature_subset(
            pools, keys,
            budgets=budgets,
            budget_weights=budget_weights,
            lambda_high=lambda_high,
        )
        all_pools.extend(pools)

    # Direct mean over all pools (no per-dataset re-weighting).
    overall = evaluate_feature_subset(
        all_pools, keys,
        budgets=budgets,
        budget_weights=budget_weights,
        lambda_high=lambda_high,
    )

    return {
        'feature_keys': keys,
        'J': overall['J'],
        'J_random_term': overall['J_random_term'],
        'J_high_term': overall['J_high_term'],
        'J_by_dataset': {ds: per_dataset[ds]['J'] for ds in dataset_order},
        'trc_high_conf': overall['trc_high_conf'],
        'trc_all': overall['trc_all'],
        'trc_random': overall['trc_random'],
        'n_pools': overall['n_pools'],
        'n_high_conf': overall['n_high_conf'],
        'n_random': overall['n_random'],
        'per_dataset': per_dataset,
        'datasets': list(dataset_order),
        'aggregation': 'pool_level_direct_mean',
        'per_pool': overall.get('per_pool', []),
    }


def _metrics_for_json(eval_result, budgets=SEARCH_BUDGET_RATIOS):
    out = {
        'J': eval_result['J'],
        'trc_high_conf': {_budget_key(b): eval_result['trc_high_conf'][float(b)] for b in budgets},
        'trc_all': {_budget_key(b): eval_result['trc_all'][float(b)] for b in budgets},
        'trc_random': {_budget_key(b): eval_result['trc_random'][float(b)] for b in budgets},
    }
    if 'J_by_dataset' in eval_result:
        out['J_by_dataset'] = {
            ds: float(v) for ds, v in eval_result['J_by_dataset'].items()
        }
    return out


def _trc_row_cells(trc_map, budgets=SEARCH_BUDGET_RATIOS):
    return [_fmt(trc_map[float(b)]) for b in budgets]


def soft_family_blocks_candidate(
    feature_key,
    current_keys,
    eval_current,
    eval_new,
    *,
    enabled,
    delta_family=SEARCH_DELTA_FAMILY,
    family_map=None,
):
    if not enabled:
        return False
    if family_map is None:
        family_map = FEATURE_FAMILY
    fam = family_map.get(feature_key)
    if fam is None:
        return False
    shares = any(family_map.get(k) == fam for k in current_keys)
    if not shares:
        return False
    gain = (
        eval_new['trc_high_conf'][0.05]
        - eval_current['trc_high_conf'][0.05]
    )
    if not np.isfinite(gain):
        return True
    return float(gain) < float(delta_family)


def random_floor_blocks_candidate(
    eval_new,
    deepgini_eval,
    *,
    enabled,
    budgets=SEARCH_BUDGET_RATIOS,
    eps=SEARCH_RANDOM_FLOOR_EPS,
):
    """Block if random-pool mean TRC drops > eps vs DeepGini at any budget.

    Supports global evals (`per_dataset` dict) and single-dataset flat evals.
    """
    if not enabled or deepgini_eval is None or eval_new is None:
        return False, None

    if 'per_dataset' in eval_new and 'per_dataset' in deepgini_eval:
        pairs = []
        for ds, ds_new in eval_new['per_dataset'].items():
            ds_base = deepgini_eval['per_dataset'].get(ds)
            if ds_base is not None:
                pairs.append((ds, ds_new, ds_base))
    else:
        # Flat per-dataset eval result
        if 'trc_random' not in eval_new or 'trc_random' not in deepgini_eval:
            return False, None
        pairs = [('local', eval_new, deepgini_eval)]

    for ds, ds_new, ds_base in pairs:
        for b in budgets:
            bf = float(b)
            new_v = ds_new['trc_random'][bf]
            base_v = ds_base['trc_random'][bf]
            if not (np.isfinite(new_v) and np.isfinite(base_v)):
                continue
            drop = float(base_v - new_v)
            if drop > float(eps):
                return True, {
                    'dataset': ds,
                    'budget': bf,
                    'trc_new': float(new_v),
                    'trc_deepgini': float(base_v),
                    'drop': drop,
                    'eps': float(eps),
                }
    return False, None


def forward_greedy_search(
    candidates,
    *,
    eval_fn,
    init_keys=SEARCH_INIT_KEYS,
    eps_stop=SEARCH_EPS_STOP,
    enable_family_soft_penalty=False,
    delta_family=SEARCH_DELTA_FAMILY,
    enable_random_floor=False,
    random_floor_eps=SEARCH_RANDOM_FLOOR_EPS,
    deepgini_eval=None,
    budgets=SEARCH_BUDGET_RATIOS,
    eval_cache=None,
):
    """Forward greedy maximizing eval_fn(S)['J']."""
    if eval_cache is None:
        eval_cache = {}

    def _cached(keys):
        key_list = list(keys)
        fk = frozenset(key_list)
        if fk not in eval_cache:
            eval_cache[fk] = eval_fn(key_list)
        eval_cache[fk]['feature_keys'] = key_list
        return eval_cache[fk]

    S = [k for k in init_keys if k in candidates]
    if not S:
        raise ValueError(f'init keys {init_keys} not found in candidates')

    path = []
    cur = _cached(S)
    path.append({
        'step': 0,
        'action': 'init',
        'added': None,
        'S': list(S),
        'J': cur['J'],
        'delta_J': None,
        'metrics': _metrics_for_json(cur, budgets=budgets),
        'blocked': [],
    })

    step = 0
    while True:
        step += 1
        best_f = None
        best_eval = cur
        blocked = []
        blocked_floor = []
        for f in candidates:
            if f in S:
                continue
            trial_keys = S + [f]
            trial = _cached(trial_keys)
            if soft_family_blocks_candidate(
                f, S, cur, trial,
                enabled=enable_family_soft_penalty,
                delta_family=delta_family,
            ):
                blocked.append(f)
                continue
            floor_hit, floor_info = random_floor_blocks_candidate(
                trial, deepgini_eval,
                enabled=enable_random_floor,
                budgets=budgets,
                eps=random_floor_eps,
            )
            if floor_hit:
                blocked_floor.append({'feature': f, **(floor_info or {})})
                continue
            if trial['J'] > best_eval['J']:
                best_f = f
                best_eval = trial

        if best_f is None:
            path.append({
                'step': step,
                'action': 'stop',
                'added': None,
                'S': list(S),
                'J': cur['J'],
                'delta_J': 0.0,
                'metrics': _metrics_for_json(cur, budgets=budgets),
                'blocked': blocked,
                'blocked_random_floor': blocked_floor,
                'reason': 'no candidate improves J(S)',
            })
            break

        delta = best_eval['J'] - cur['J']
        if delta < float(eps_stop):
            path.append({
                'step': step,
                'action': 'stop',
                'added': None,
                'S': list(S),
                'J': cur['J'],
                'delta_J': float(delta),
                'best_candidate': best_f,
                'best_candidate_J': best_eval['J'],
                'metrics': _metrics_for_json(cur, budgets=budgets),
                'blocked': blocked,
                'blocked_random_floor': blocked_floor,
                'reason': f'max_delta_J ({_fmt(delta)}) < eps_stop ({eps_stop})',
            })
            break

        S = S + [best_f]
        cur = best_eval
        path.append({
            'step': step,
            'action': 'add',
            'added': best_f,
            'S': list(S),
            'J': cur['J'],
            'delta_J': float(delta),
            'metrics': _metrics_for_json(cur, budgets=budgets),
            'blocked': blocked,
            'blocked_random_floor': blocked_floor,
        })

    return {'S': list(S), 'eval': cur, 'path': path, 'eval_cache': eval_cache}


def backward_prune_search(
    feature_keys,
    *,
    eval_fn,
    delta_prune=SEARCH_DELTA_PRUNE,
    keep_deepgini=False,
    budgets=SEARCH_BUDGET_RATIOS,
    eval_cache=None,
    enable_random_floor=False,
    random_floor_eps=SEARCH_RANDOM_FLOOR_EPS,
    deepgini_eval=None,
):
    if eval_cache is None:
        eval_cache = {}

    def _cached(keys):
        key_list = list(keys)
        fk = frozenset(key_list)
        if fk not in eval_cache:
            eval_cache[fk] = eval_fn(key_list)
        eval_cache[fk]['feature_keys'] = key_list
        return eval_cache[fk]

    S = list(feature_keys)
    log = []
    cur = _cached(S)
    log.append({
        'step': 0,
        'action': 'start',
        'removed': None,
        'S': list(S),
        'J': cur['J'],
        'drop': None,
        'metrics': _metrics_for_json(cur, budgets=budgets),
    })

    step = 0
    while len(S) > 1:
        step += 1
        best_f = None
        best_drop = None
        best_new = None
        blocked_floor = []
        for f in S:
            if keep_deepgini and f == 'deepgini':
                continue
            trial_keys = [k for k in S if k != f]
            trial = _cached(trial_keys)
            floor_hit, floor_info = random_floor_blocks_candidate(
                trial, deepgini_eval,
                enabled=enable_random_floor,
                budgets=budgets,
                eps=random_floor_eps,
            )
            if floor_hit:
                blocked_floor.append({'feature': f, **(floor_info or {})})
                continue
            drop = cur['J'] - trial['J']
            if best_drop is None or drop < best_drop:
                best_f = f
                best_drop = float(drop)
                best_new = trial

        if best_f is None:
            log.append({
                'step': step,
                'action': 'stop',
                'removed': None,
                'S': list(S),
                'J': cur['J'],
                'drop': None,
                'metrics': _metrics_for_json(cur, budgets=budgets),
                'blocked_random_floor': blocked_floor,
                'reason': 'no removable feature without violating random floor',
            })
            break
        if best_drop < float(delta_prune):
            S = [k for k in S if k != best_f]
            cur = best_new
            log.append({
                'step': step,
                'action': 'remove',
                'removed': best_f,
                'S': list(S),
                'J': cur['J'],
                'drop': best_drop,
                'metrics': _metrics_for_json(cur, budgets=budgets),
                'blocked_random_floor': blocked_floor,
            })
        else:
            log.append({
                'step': step,
                'action': 'stop',
                'removed': None,
                'S': list(S),
                'J': cur['J'],
                'drop': best_drop,
                'candidate_remove': best_f,
                'metrics': _metrics_for_json(cur, budgets=budgets),
                'blocked_random_floor': blocked_floor,
                'reason': f'min_drop ({_fmt(best_drop)}) >= delta_prune ({delta_prune})',
            })
            break

    return {'S': list(S), 'eval': cur, 'log': log, 'eval_cache': eval_cache}


def _load_search_pools(
    feature_cache_dir,
    sampled_root,
    datasets,
    pool_types,
    error_ratios,
    seeds,
    sampling_modes,
):
    pools = list(iter_cached_pools(
        feature_cache_dir, sampled_root, datasets, pool_types,
        error_ratios, seeds, sampling_modes,
    ))
    return pools


def _write_single_feature_trc_outputs(
    out_dir, single_rows, budgets=SEARCH_BUDGET_RATIOS, *, name_prefix='',
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{name_prefix}single_feature_trc'
    payload = {
        'budgets': [float(b) for b in budgets],
        'rows': single_rows,
    }
    (out_dir / f'{stem}.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8',
    )

    headers = (
        ['feature', 'J']
        + [f'TRC_rand@{_budget_key(b)}' for b in budgets]
        + [f'TRC_high@{_budget_key(b)}' for b in budgets]
        + ['n_pools']
    )
    md_rows = []
    for row in sorted(single_rows, key=lambda r: (-(r['J'] if np.isfinite(r['J']) else -1e9), r['feature'])):
        md_rows.append(
            [row['feature'], _fmt(row['J'])]
            + _trc_row_cells(row['trc_random'], budgets)
            + _trc_row_cells(row['trc_high_conf'], budgets)
            + [str(row.get('n_pools', ''))]
        )
    md = [
        '# Single-feature TRC (Dev, diagnostic)',
        '',
        'Each row is $S=\\{f\\}$ under pure risk-ranking TRC; **not** used as a hard filter.',
        '',
        _markdown_table(headers, md_rows),
        '',
    ]
    (out_dir / f'{stem}.md').write_text('\n'.join(md), encoding='utf-8')


def _write_greedy_path_outputs(out_dir, path, budgets=SEARCH_BUDGET_RATIOS, *, name_prefix=''):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{name_prefix}greedy_path'
    (out_dir / f'{stem}.json').write_text(
        json.dumps({'path': path}, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    headers = (
        ['step', 'action', 'added', 'delta_J', 'J']
        + [f'TRC_rand@{_budget_key(b)}' for b in budgets]
        + [f'TRC_high@{_budget_key(b)}' for b in budgets]
        + ['S']
    )
    md_rows = []
    for step in path:
        m = step.get('metrics') or {}
        rand_map = {
            float(b): m.get('trc_random', {}).get(_budget_key(b), np.nan) for b in budgets
        }
        high_map = {
            float(b): m.get('trc_high_conf', {}).get(_budget_key(b), np.nan) for b in budgets
        }
        md_rows.append(
            [
                step.get('step'),
                step.get('action'),
                step.get('added') or '-',
                _fmt(step.get('delta_J')) if step.get('delta_J') is not None else '-',
                _fmt(step.get('J')),
            ]
            + _trc_row_cells(rand_map, budgets)
            + _trc_row_cells(high_map, budgets)
            + [', '.join(step.get('S') or [])]
        )
    md = [
        '# Forward greedy path',
        '',
        'Each add step maximizes $J(S\\cup\\{f\\})$; stop when max $\\Delta J < \\varepsilon_{stop}$.',
        '',
        _markdown_table(headers, md_rows),
        '',
    ]
    (out_dir / f'{stem}.md').write_text('\n'.join(md), encoding='utf-8')


def _write_prune_log_outputs(out_dir, log, budgets=SEARCH_BUDGET_RATIOS, *, name_prefix=''):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{name_prefix}prune_log'
    (out_dir / f'{stem}.json').write_text(
        json.dumps({'log': log}, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    headers = (
        ['step', 'action', 'removed', 'drop', 'J']
        + [f'TRC_rand@{_budget_key(b)}' for b in budgets]
        + [f'TRC_high@{_budget_key(b)}' for b in budgets]
        + ['S']
    )
    md_rows = []
    for step in log:
        m = step.get('metrics') or {}
        rand_map = {float(b): m.get('trc_random', {}).get(_budget_key(b), np.nan) for b in budgets}
        high_map = {float(b): m.get('trc_high_conf', {}).get(_budget_key(b), np.nan) for b in budgets}
        md_rows.append(
            [
                step.get('step'),
                step.get('action'),
                step.get('removed') or '-',
                _fmt(step.get('drop')) if step.get('drop') is not None else '-',
                _fmt(step.get('J')),
            ]
            + _trc_row_cells(rand_map, budgets)
            + _trc_row_cells(high_map, budgets)
            + [', '.join(step.get('S') or [])]
        )
    md = [
        '# Backward prune log',
        '',
        'Remove features with drop $J(S)-J(S\\setminus\\{f\\}) < \\delta_{prune}$.',
        '',
        _markdown_table(headers, md_rows),
        '',
    ]
    (out_dir / f'{stem}.md').write_text('\n'.join(md), encoding='utf-8')


def _write_comparison_md(
    out_dir, star_eval, gini_eval, budgets=SEARCH_BUDGET_RATIOS,
    *, name_prefix='', title=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{name_prefix}comparison'
    headers = (
        ['method', 'aggregate', 'J']
        + [f'TRC@{_budget_key(b)}' for b in budgets]
        + ['n_feat']
    )
    rows = []
    for name, ev in (('deepgini', gini_eval), ('S*', star_eval)):
        for agg_name, trc_map in (
            ('random', ev['trc_random']),
            ('high_conf', ev['trc_high_conf']),
            ('all', ev['trc_all']),
        ):
            rows.append(
                [
                    name,
                    agg_name,
                    _fmt(ev['J']) if agg_name == 'random' else '-',
                ]
                + _trc_row_cells(trc_map, budgets)
                + [str(len(ev['feature_keys'])) if agg_name == 'random' else '-']
            )

    by_ds = {}
    for ev_name, ev in (('deepgini', gini_eval), ('S*', star_eval)):
        for row in ev.get('per_pool') or []:
            ds = row['dataset']
            by_ds.setdefault(ds, {'deepgini': [], 'S*': []})
            by_ds[ds][ev_name].append(row)

    extra = ['', '## Per-dataset means (Dev)', '']
    for ds in sorted(by_ds):
        extra.append(f'### {ds}')
        extra.append('')
        ds_headers = ['method', 'mode'] + [f'TRC@{_budget_key(b)}' for b in budgets]
        ds_rows = []
        for method in ('deepgini', 'S*'):
            for mode in ('high_conf', 'random', 'all'):
                bucket = [
                    r for r in by_ds[ds][method]
                    if mode == 'all' or r['sampling_mode'] == mode
                ]
                if not bucket:
                    continue
                means = {}
                for b in budgets:
                    means[float(b)] = _mean_finite([r['trc'][_budget_key(b)] for r in bucket])
                ds_rows.append([method, mode] + _trc_row_cells(means, budgets))
        extra.append(_markdown_table(ds_headers, ds_rows))
        extra.append('')

    if title is None:
        title = 'S* vs DeepGini (pure risk-ranking TRC)'
    md = [
        f'# {title}',
        '',
        'Primary selection uses $J(S)$; DeepGini is a soft reference only.',
        '',
        _markdown_table(headers, rows),
        *extra,
    ]
    (out_dir / f'{stem}.md').write_text('\n'.join(md), encoding='utf-8')
    (out_dir / f'{stem}.json').write_text(
        json.dumps({
            'star': _metrics_for_json(star_eval, budgets=budgets),
            'deepgini': _metrics_for_json(gini_eval, budgets=budgets),
            'star_keys': list(star_eval.get('feature_keys') or []),
        }, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def _resolve_search_seeds(seeds):
    available_seeds = sorted({int(s) for s in seeds})
    dev_seeds = [s for s in available_seeds if s in SEARCH_DEV_SEEDS]
    test_seeds = [s for s in available_seeds if s in SEARCH_TEST_SEEDS]
    if not dev_seeds:
        dev_seeds = list(available_seeds)
        seed_note = (
            f'No seeds in DEV={list(SEARCH_DEV_SEEDS)}; '
            f'using all available seeds as Dev: {dev_seeds}'
        )
    elif not test_seeds and set(dev_seeds) == set(available_seeds):
        seed_note = (
            f'Only Dev seeds available ({dev_seeds}); Test empty. '
            'Dev==available; re-validate when seeds 3/4 exist.'
        )
    else:
        seed_note = f'Dev={dev_seeds}, Test={test_seeds}'
    return available_seeds, dev_seeds, test_seeds, seed_note


def _run_feature_search_one_dataset(
    dataset_name,
    feature_cache_dir,
    sampled_root,
    pool_types,
    error_ratios,
    available_seeds,
    dev_seeds,
    test_seeds,
    seed_note,
    sampling_modes,
    out_dir,
    *,
    lambda_high,
    eps_stop,
    delta_prune,
    keep_deepgini,
    enable_family_soft_penalty,
    delta_family,
    budgets,
    budget_weights,
    enable_random_floor=True,
    random_floor_eps=SEARCH_RANDOM_FLOOR_EPS,
):
    """Run greedy feature search on a single dataset; write artifacts under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(CONTINUOUS_RISK_FEATURE_KEYS)

    all_pools = _load_search_pools(
        feature_cache_dir, sampled_root, [dataset_name], pool_types,
        error_ratios, available_seeds, sampling_modes,
    )
    if not all_pools:
        raise FileNotFoundError(
            f'No feature caches for dataset={dataset_name}. Run --phase compute first.',
        )

    dev_pools = [p for p in all_pools if p['seed'] in set(dev_seeds)]
    test_pools = [p for p in all_pools if p['seed'] in set(test_seeds)]
    if not dev_pools:
        raise RuntimeError(f'No Dev pools for dataset={dataset_name}')

    print(
        f'[search:{dataset_name}] Dev pools={len(dev_pools)}, '
        f'Test pools={len(test_pools)} ({seed_note}); '
        f'random_floor={"on" if enable_random_floor else "off"}, eps={random_floor_eps}',
    )

    config = {
        'dataset': dataset_name,
        'scope': 'per_dataset',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'budgets': [float(b) for b in budgets],
        'budget_weights': {_budget_key(b): float(budget_weights[float(b)]) for b in budgets},
        'lambda_high': float(lambda_high),
        'eps_stop': float(eps_stop),
        'delta_prune': float(delta_prune),
        'init': list(SEARCH_INIT_KEYS),
        'keep_deepgini': bool(keep_deepgini),
        'enable_family_soft_penalty': bool(enable_family_soft_penalty),
        'delta_family': float(delta_family),
        'enable_random_floor': bool(enable_random_floor),
        'random_floor_eps': float(random_floor_eps),
        'candidates': candidates,
        'pool_types': list(pool_types),
        'error_ratios': [float(x) for x in error_ratios],
        'sampling_modes': list(sampling_modes),
        'available_seeds': available_seeds,
        'dev_seeds': list(dev_seeds),
        'test_seeds': list(test_seeds),
        'seed_note': seed_note,
        'n_dev_pools': len(dev_pools),
        'n_test_pools': len(test_pools),
        'objective': (
            'J(S)=1*sum_b w_b*mean_TRC_random + lambda*sum_b w_b*mean_TRC_high; '
            'random_floor vs deepgini (optional); per dataset'
        ),
    }
    (out_dir / 'search_config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8',
    )

    print(f'[search:{dataset_name}] Step A: single-feature TRC')
    eval_cache = {}

    def _eval_local(keys):
        return evaluate_feature_subset(
            dev_pools, list(keys),
            budgets=budgets,
            budget_weights=budget_weights,
            lambda_high=lambda_high,
        )

    single_rows = []
    for f in tqdm(candidates, desc=f'{dataset_name} single-feature'):
        ev = _eval_local([f])
        eval_cache[frozenset([f])] = ev
        single_rows.append({
            'feature': f,
            'J': ev['J'],
            'trc_high_conf': ev['trc_high_conf'],
            'trc_all': ev['trc_all'],
            'trc_random': ev['trc_random'],
            'n_pools': ev['n_pools'],
        })
    _write_single_feature_trc_outputs(out_dir, single_rows, budgets=budgets)

    gini_eval = _eval_local(list(SEARCH_INIT_KEYS))
    eval_cache[frozenset(SEARCH_INIT_KEYS)] = gini_eval

    print(f'[search:{dataset_name}] Step B: forward greedy')
    fwd = forward_greedy_search(
        candidates,
        eval_fn=_eval_local,
        init_keys=SEARCH_INIT_KEYS,
        eps_stop=eps_stop,
        enable_family_soft_penalty=enable_family_soft_penalty,
        delta_family=delta_family,
        enable_random_floor=enable_random_floor,
        random_floor_eps=random_floor_eps,
        deepgini_eval=gini_eval,
        budgets=budgets,
        eval_cache=eval_cache,
    )
    _write_greedy_path_outputs(out_dir, fwd['path'], budgets=budgets)
    print(
        f"[search:{dataset_name}] after forward: |S|={len(fwd['S'])}, "
        f"J={_fmt(fwd['eval']['J'])}",
    )

    print(f'[search:{dataset_name}] Step C: backward prune')
    prn = backward_prune_search(
        fwd['S'],
        eval_fn=_eval_local,
        delta_prune=delta_prune,
        keep_deepgini=keep_deepgini,
        budgets=budgets,
        eval_cache=fwd['eval_cache'],
        enable_random_floor=enable_random_floor,
        random_floor_eps=random_floor_eps,
        deepgini_eval=gini_eval,
    )
    _write_prune_log_outputs(out_dir, prn['log'], budgets=budgets)

    star_keys = prn['S']
    star_eval = prn['eval']
    _write_comparison_md(out_dir, star_eval, gini_eval, budgets=budgets)

    test_metrics = None
    test_gini = None
    if test_pools:
        test_metrics = _metrics_for_json(
            evaluate_feature_subset(
                test_pools, star_keys,
                budgets=budgets,
                budget_weights=budget_weights,
                lambda_high=lambda_high,
            ),
            budgets=budgets,
        )
        test_gini = _metrics_for_json(
            evaluate_feature_subset(
                test_pools, list(SEARCH_INIT_KEYS),
                budgets=budgets,
                budget_weights=budget_weights,
                lambda_high=lambda_high,
            ),
            budgets=budgets,
        )

    best_payload = {
        'dataset': dataset_name,
        'scope': 'per_dataset',
        'feature_keys': star_keys,
        'selection_criterion': {
            'name': 'J_weighted_trc',
            'search': 'argmax J(S) via forward_greedy + backward_prune',
            'formula': (
                'sum_b w_b * mean_TRC_random(S,b) + lambda * sum_b w_b * mean_TRC_high(S,b)'
            ),
            'primary_term': 'random_trc',
            'secondary_term': 'high_conf_trc',
            'budgets': [float(b) for b in budgets],
            'budget_weights': {_budget_key(b): float(budget_weights[float(b)]) for b in budgets},
            'lambda_high': float(lambda_high),
            'eps_stop': float(eps_stop),
            'delta_prune': float(delta_prune),
            'enable_random_floor': bool(enable_random_floor),
            'random_floor_eps': float(random_floor_eps),
            'init': list(SEARCH_INIT_KEYS),
            'keep_deepgini': bool(keep_deepgini),
            'enable_family_soft_penalty': bool(enable_family_soft_penalty),
            'deepgini_role': 'search_anchor_and_reference_only_not_hard_constraint',
            'optimization_scope': 'per_dataset',
        },
        'dev_seeds': list(dev_seeds),
        'test_seeds': list(test_seeds),
        'seed_note': seed_note,
        'forward_S': fwd['S'],
        'metrics': {
            'dev': _metrics_for_json(star_eval, budgets=budgets),
            'test': test_metrics,
        },
        'reference_deepgini': {
            'trc_random': {
                _budget_key(b): gini_eval['trc_random'][float(b)] for b in budgets
            },
            'trc_high_conf': {
                _budget_key(b): gini_eval['trc_high_conf'][float(b)] for b in budgets
            },
            'trc_all': {
                _budget_key(b): gini_eval['trc_all'][float(b)] for b in budgets
            },
            'J': gini_eval['J'],
            'test': test_gini,
            'note': 'for comparison only',
        },
    }
    (out_dir / 'best_feature_keys.json').write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False), encoding='utf-8',
    )

    print(f'[search:{dataset_name}] S* ({len(star_keys)}): {star_keys}')
    print(
        f"[search:{dataset_name}] Dev J(S*)={_fmt(star_eval['J'])}  "
        f"J(deepgini)={_fmt(gini_eval['J'])}",
    )
    print(f'[search:{dataset_name}] outputs -> {out_dir}')
    return best_payload


def _write_cross_dataset_summary(
    summary_md_path,
    per_dataset_results,
    budgets=SEARCH_BUDGET_RATIOS,
    *,
    per_dataset_dirs=None,
):
    """Compare per-dataset S*; write cache_files/feature_exploration_summary.md (+ .json)."""
    summary_md_path = Path(summary_md_path)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path = summary_md_path.with_suffix('.json')

    ds_names = list(per_dataset_results.keys())
    key_sets = {
        ds: list(per_dataset_results[ds]['feature_keys']) for ds in ds_names
    }
    frozensets = {ds: frozenset(keys) for ds, keys in key_sets.items()}
    all_equal = len(set(frozensets.values())) == 1 if frozensets else True
    intersection = set.intersection(*(set(v) for v in frozensets.values())) if frozensets else set()
    union = set.union(*(set(v) for v in frozensets.values())) if frozensets else set()

    verdict = (
        'shared (identical S* across datasets)'
        if all_equal
        else 'adaptive (dataset-specific S*)'
    )

    summary = {
        'scope': 'per_dataset',
        'verdict': verdict,
        'identical_across_datasets': all_equal,
        'datasets': ds_names,
        'feature_keys_by_dataset': key_sets,
        'intersection': sorted(intersection),
        'union': sorted(union),
        'only_in': {
            ds: sorted(frozensets[ds] - intersection) for ds in ds_names
        },
        'metrics_by_dataset': {
            ds: {
                'J': per_dataset_results[ds]['metrics']['dev']['J'],
                'trc_random': per_dataset_results[ds]['metrics']['dev']['trc_random'],
                'trc_high_conf': per_dataset_results[ds]['metrics']['dev']['trc_high_conf'],
                'trc_all': per_dataset_results[ds]['metrics']['dev']['trc_all'],
                'J_deepgini': per_dataset_results[ds]['reference_deepgini']['J'],
            }
            for ds in ds_names
        },
        'artifact_dirs': {
            ds: str(Path(per_dataset_dirs[ds]))
            for ds in ds_names
        } if per_dataset_dirs else {},
        'note': (
            'J(S)=weighted TRC_random + lambda*weighted TRC_high; maximized per dataset. '
            'If S* differs, the risk feature set is adaptive. '
            'Artifacts: cache_files/{dataset}/feature_search/.'
        ),
    }
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8',
    )

    headers = (
        ['dataset', 'n_feat', 'J_star', 'J_deepgini']
        + [f'TRC_rand@{_budget_key(b)}' for b in budgets]
        + [f'TRC_high@{_budget_key(b)}' for b in budgets]
        + ['feature_keys']
    )
    rows = []
    for ds in ds_names:
        m = per_dataset_results[ds]['metrics']['dev']
        rows.append(
            [
                ds,
                str(len(key_sets[ds])),
                _fmt(m['J']),
                _fmt(per_dataset_results[ds]['reference_deepgini']['J']),
            ]
            + [_fmt(m['trc_random'][_budget_key(b)]) for b in budgets]
            + [_fmt(m['trc_high_conf'][_budget_key(b)]) for b in budgets]
            + [', '.join(key_sets[ds])]
        )

    artifact_lines = []
    if per_dataset_dirs:
        artifact_lines.extend(['', '## Artifact locations', ''])
        for ds in ds_names:
            artifact_lines.append(f'- `{ds}`: `{Path(per_dataset_dirs[ds]).as_posix()}`')

    md = [
        '# Feature exploration summary',
        '',
        f'**Verdict:** {verdict}',
        '',
        'Search maximizes $J(S)$ **separately per dataset** (not pooled).',
        '',
        'Primary = random TRC; secondary = $\\lambda$ · high_conf TRC.',
        '',
        _markdown_table(headers, rows),
        '',
        '## Set comparison',
        '',
        f'- Intersection ({len(intersection)}): {", ".join(sorted(intersection)) or "(empty)"}',
        f'- Union ({len(union)}): {", ".join(sorted(union)) or "(empty)"}',
        '',
    ]
    for ds in ds_names:
        only = summary['only_in'][ds]
        md.append(f'- Only in `{ds}`: {", ".join(only) if only else "(none)"}')
    md.extend(artifact_lines)
    md.extend([
        '',
        '## Interpretation',
        '',
        (
            'Identical $S^*$ → a shared default `feature_keys` is supported. '
            'Different $S^*$ → report an **adaptive** risk-feature set '
            '(dataset-conditioned), rather than forcing one global subset.'
        ),
        '',
    ])
    summary_md_path.write_text('\n'.join(md), encoding='utf-8')
    return summary


def _write_global_feature_exploration_summary(
    summary_md_path,
    global_payload,
    budgets=SEARCH_BUDGET_RATIOS,
    *,
    per_dataset_adaptive=None,
):
    """Write cache_files/global_feature_exploration_summary.md (+ .json)."""
    summary_md_path = Path(summary_md_path)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    star_keys = list(global_payload['feature_keys'])
    per_ds_metrics = global_payload['metrics']['dev_by_dataset']
    gini_by_ds = global_payload['reference_deepgini']['by_dataset']

    delta_rows = {}
    for ds, m in per_ds_metrics.items():
        g = gini_by_ds[ds]
        delta_rows[ds] = {
            _budget_key(b): float(m['trc_random'][_budget_key(b)] - g['trc_random'][_budget_key(b)])
            for b in budgets
        }

    summary = {
        'scope': 'global',
        'feature_keys': star_keys,
        'J_global': global_payload['metrics']['dev']['J'],
        'J_deepgini_global': global_payload['reference_deepgini']['J'],
        'metrics_by_dataset': per_ds_metrics,
        'delta_trc_random_vs_deepgini': delta_rows,
        'reference_deepgini_by_dataset': gini_by_ds,
        'artifacts': global_payload.get('artifacts', {}),
    }
    if per_dataset_adaptive:
        adaptive_keys = {
            ds: list(per_dataset_adaptive[ds]['feature_keys'])
            for ds in per_dataset_adaptive
        }
        summary['adaptive_feature_keys_by_dataset'] = adaptive_keys
        summary['adaptive_vs_global'] = {
            ds: {
                'identical_to_global': set(adaptive_keys[ds]) == set(star_keys),
                'only_adaptive': sorted(set(adaptive_keys[ds]) - set(star_keys)),
                'only_global': sorted(set(star_keys) - set(adaptive_keys[ds])),
            }
            for ds in adaptive_keys
        }

    summary_md_path.with_suffix('.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8',
    )

    headers = (
        ['dataset', 'n_feat', 'J_d', 'J_deepgini']
        + [f'TRC_rand@{_budget_key(b)}' for b in budgets]
        + [f'delta_rand@{_budget_key(b)}' for b in budgets]
        + [f'TRC_high@{_budget_key(b)}' for b in budgets]
    )
    rows = []
    for ds in sorted(per_ds_metrics.keys()):
        m = per_ds_metrics[ds]
        g = gini_by_ds[ds]
        rows.append(
            [
                ds,
                str(len(star_keys)),
                _fmt(m['J']),
                _fmt(g['J']),
            ]
            + [_fmt(m['trc_random'][_budget_key(b)]) for b in budgets]
            + [_fmt(delta_rows[ds][_budget_key(b)]) for b in budgets]
            + [_fmt(m['trc_high_conf'][_budget_key(b)]) for b in budgets]
        )

    md = [
        '# Global feature exploration summary',
        '',
        f'**S_global** ({len(star_keys)}): {", ".join(star_keys)}',
        '',
        f'$J_{{\\mathrm{{global}}}}(S^*)$ = {_fmt(global_payload["metrics"]["dev"]["J"])} '
        f'(DeepGini {_fmt(global_payload["reference_deepgini"]["J"])})',
        '',
        'Primary = random TRC; secondary = $\\lambda$ · high_conf. '
        '$J_{\\mathrm{global}}$ = direct mean over all Dev pools '
        '(no per-dataset re-weighting).',
        '',
        '## Per-dataset TRC under S_global',
        '',
        _markdown_table(headers, rows),
        '',
        '## Delta random TRC vs DeepGini (S_global - deepgini)',
        '',
    ]
    d_headers = ['dataset'] + [f'delta@{_budget_key(b)}' for b in budgets]
    d_rows = [
        [ds] + [_fmt(delta_rows[ds][_budget_key(b)]) for b in budgets]
        for ds in sorted(delta_rows)
    ]
    md.append(_markdown_table(d_headers, d_rows))
    md.append('')

    if per_dataset_adaptive:
        md.extend(['## Adaptive S* (per_dataset) vs S_global', ''])
        a_headers = ['dataset', 'n_feat', 'feature_keys', 'identical']
        a_rows = []
        for ds in sorted(per_dataset_adaptive):
            keys = list(per_dataset_adaptive[ds]['feature_keys'])
            a_rows.append([
                ds,
                str(len(keys)),
                ', '.join(keys),
                'yes' if set(keys) == set(star_keys) else 'no',
            ])
        md.append(_markdown_table(a_headers, a_rows))
        md.append('')

    md.extend([
        '## Artifacts',
        '',
        f'- Top-level: `{summary_md_path.parent.as_posix()}/global_*`',
        '- Per-dataset eval: `cache_files/{dataset}/feature_search/global_*`',
        '',
    ])
    summary_md_path.write_text('\n'.join(md), encoding='utf-8')
    return summary


def _run_feature_search_global(
    feature_cache_dir,
    sampled_root,
    datasets,
    pool_types,
    error_ratios,
    available_seeds,
    dev_seeds,
    test_seeds,
    seed_note,
    sampling_modes,
    *,
    lambda_high,
    eps_stop,
    delta_prune,
    keep_deepgini,
    enable_family_soft_penalty,
    delta_family,
    budgets,
    budget_weights,
    summary_path=None,
    per_dataset_adaptive=None,
    enable_random_floor=True,
    random_floor_eps=SEARCH_RANDOM_FLOOR_EPS,
):
    """Optimize S*_global via pool-level J + optional random floor; write global_* artifacts."""
    feature_cache_dir = Path(feature_cache_dir)
    dataset_list = list(datasets)
    candidates = list(CONTINUOUS_RISK_FEATURE_KEYS)

    pools_by_dataset = {}
    for ds in dataset_list:
        all_pools = _load_search_pools(
            feature_cache_dir, sampled_root, [ds], pool_types,
            error_ratios, available_seeds, sampling_modes,
        )
        dev_pools = [p for p in all_pools if p['seed'] in set(dev_seeds)]
        if not dev_pools:
            raise RuntimeError(f'No Dev pools for dataset={ds}')
        pools_by_dataset[ds] = dev_pools
        print(f'[search:global] {ds}: Dev pools={len(dev_pools)}')

    if summary_path is None:
        summary_path = feature_cache_dir / 'global_feature_exploration_summary.md'
    else:
        summary_path = Path(summary_path)

    config = {
        'scope': 'global',
        'aggregation': 'pool_level_direct_mean',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'datasets': dataset_list,
        'budgets': [float(b) for b in budgets],
        'budget_weights': {_budget_key(b): float(budget_weights[float(b)]) for b in budgets},
        'lambda_high': float(lambda_high),
        'eps_stop': float(eps_stop),
        'delta_prune': float(delta_prune),
        'init': list(SEARCH_INIT_KEYS),
        'keep_deepgini': bool(keep_deepgini),
        'enable_family_soft_penalty': bool(enable_family_soft_penalty),
        'delta_family': float(delta_family),
        'enable_random_floor': bool(enable_random_floor),
        'random_floor_eps': float(random_floor_eps),
        'candidates': candidates,
        'dev_seeds': list(dev_seeds),
        'test_seeds': list(test_seeds),
        'seed_note': seed_note,
        'objective': (
            'J_global=direct mean over all Dev pools of '
            '(sum_b w_b*TRC_rand + lambda*sum_b w_b*TRC_high); '
            'forward skips adds that drop any dataset random mean TRC '
            f'> {float(random_floor_eps):.2f} vs deepgini at any budget'
        ),
    }
    (feature_cache_dir / 'global_search_config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8',
    )

    def _eval_global(keys):
        return evaluate_feature_subset_global(
            pools_by_dataset, list(keys),
            budgets=budgets,
            budget_weights=budget_weights,
            lambda_high=lambda_high,
            dataset_order=dataset_list,
        )

    print('[search:global] Step A: single-feature TRC (J_global)')
    eval_cache = {}
    single_rows = []
    for f in tqdm(candidates, desc='global single-feature'):
        ev = _eval_global([f])
        eval_cache[frozenset([f])] = ev
        single_rows.append({
            'feature': f,
            'J': ev['J'],
            'trc_high_conf': ev['trc_high_conf'],
            'trc_all': ev['trc_all'],
            'trc_random': ev['trc_random'],
            'n_pools': ev['n_pools'],
            'J_by_dataset': ev.get('J_by_dataset'),
        })
    _write_single_feature_trc_outputs(
        feature_cache_dir, single_rows, budgets=budgets, name_prefix='global_',
    )

    gini_eval = _eval_global(list(SEARCH_INIT_KEYS))
    eval_cache[frozenset(SEARCH_INIT_KEYS)] = gini_eval

    print(
        f'[search:global] Step B: forward greedy '
        f'(random_floor={"on" if enable_random_floor else "off"}, '
        f'eps={random_floor_eps})',
    )
    fwd = forward_greedy_search(
        candidates,
        eval_fn=_eval_global,
        init_keys=SEARCH_INIT_KEYS,
        eps_stop=eps_stop,
        enable_family_soft_penalty=enable_family_soft_penalty,
        delta_family=delta_family,
        enable_random_floor=enable_random_floor,
        random_floor_eps=random_floor_eps,
        deepgini_eval=gini_eval,
        budgets=budgets,
        eval_cache=eval_cache,
    )
    _write_greedy_path_outputs(
        feature_cache_dir, fwd['path'], budgets=budgets, name_prefix='global_',
    )
    print(f"[search:global] after forward: |S|={len(fwd['S'])}, J={_fmt(fwd['eval']['J'])}")

    print('[search:global] Step C: backward prune')
    prn = backward_prune_search(
        fwd['S'],
        eval_fn=_eval_global,
        delta_prune=delta_prune,
        keep_deepgini=keep_deepgini,
        budgets=budgets,
        eval_cache=fwd['eval_cache'],
        enable_random_floor=enable_random_floor,
        random_floor_eps=random_floor_eps,
        deepgini_eval=gini_eval,
    )
    _write_prune_log_outputs(
        feature_cache_dir, prn['log'], budgets=budgets, name_prefix='global_',
    )

    star_keys = prn['S']
    star_eval = prn['eval']

    artifacts = {
        'top_level': str(feature_cache_dir),
        'per_dataset': {},
    }
    dev_by_dataset = {}
    gini_by_dataset = {}
    for ds in dataset_list:
        ds_dir = feature_cache_dir / ds / 'feature_search'
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_star = star_eval['per_dataset'][ds]
        ds_gini = gini_eval['per_dataset'][ds]
        # ensure feature_keys present for comparison writer
        ds_star = {**ds_star, 'feature_keys': list(star_keys)}
        ds_gini = {**ds_gini, 'feature_keys': list(SEARCH_INIT_KEYS)}
        _write_comparison_md(
            ds_dir, ds_star, ds_gini, budgets=budgets,
            name_prefix='global_',
            title=f'S_global vs DeepGini on {ds}',
        )
        eval_metrics = {
            'dataset': ds,
            'scope': 'global_eval',
            'feature_keys': star_keys,
            'metrics': _metrics_for_json(ds_star, budgets=budgets),
            'reference_deepgini': _metrics_for_json(ds_gini, budgets=budgets),
            'J_global': star_eval['J'],
        }
        (ds_dir / 'global_eval_metrics.json').write_text(
            json.dumps(eval_metrics, indent=2, ensure_ascii=False), encoding='utf-8',
        )
        keys_payload = {
            'dataset': ds,
            'scope': 'global',
            'feature_keys': star_keys,
            'metrics': _metrics_for_json(ds_star, budgets=budgets),
            'reference_deepgini': _metrics_for_json(ds_gini, budgets=budgets),
            'note': 'Same S*_global as cache_files/global_best_feature_keys.json',
        }
        (ds_dir / 'global_feature_keys.json').write_text(
            json.dumps(keys_payload, indent=2, ensure_ascii=False), encoding='utf-8',
        )
        artifacts['per_dataset'][ds] = str(ds_dir)
        dev_by_dataset[ds] = _metrics_for_json(ds_star, budgets=budgets)
        gini_by_dataset[ds] = _metrics_for_json(ds_gini, budgets=budgets)

    best_payload = {
        'scope': 'global',
        'feature_keys': star_keys,
        'selection_criterion': {
            'name': 'J_global_pool_direct_mean',
            'search': 'argmax J_global via forward_greedy + backward_prune',
            'formula': (
                'J_global=pool_level_mean(sum_b w_b*TRC_rand + lambda*sum_b w_b*TRC_high); '
                'random_floor: skip add if any dataset random mean TRC drops > eps vs deepgini'
            ),
            'primary_term': 'random_trc',
            'secondary_term': 'high_conf_trc',
            'aggregation': 'pool_level_direct_mean',
            'budgets': [float(b) for b in budgets],
            'budget_weights': {_budget_key(b): float(budget_weights[float(b)]) for b in budgets},
            'lambda_high': float(lambda_high),
            'eps_stop': float(eps_stop),
            'delta_prune': float(delta_prune),
            'enable_random_floor': bool(enable_random_floor),
            'random_floor_eps': float(random_floor_eps),
            'init': list(SEARCH_INIT_KEYS),
            'keep_deepgini': bool(keep_deepgini),
            'enable_family_soft_penalty': bool(enable_family_soft_penalty),
        },
        'datasets': dataset_list,
        'dev_seeds': list(dev_seeds),
        'test_seeds': list(test_seeds),
        'seed_note': seed_note,
        'forward_S': fwd['S'],
        'metrics': {
            'dev': _metrics_for_json(star_eval, budgets=budgets),
            'dev_by_dataset': dev_by_dataset,
        },
        'reference_deepgini': {
            'J': gini_eval['J'],
            'trc_random': {
                _budget_key(b): gini_eval['trc_random'][float(b)] for b in budgets
            },
            'trc_high_conf': {
                _budget_key(b): gini_eval['trc_high_conf'][float(b)] for b in budgets
            },
            'by_dataset': gini_by_dataset,
            'note': 'for comparison only',
        },
        'artifacts': artifacts,
    }
    (feature_cache_dir / 'global_best_feature_keys.json').write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False), encoding='utf-8',
    )

    summary = _write_global_feature_exploration_summary(
        summary_path,
        best_payload,
        budgets=budgets,
        per_dataset_adaptive=per_dataset_adaptive,
    )

    print(f'[search:global] S* ({len(star_keys)}): {star_keys}')
    print(
        f"[search:global] J_global(S*)={_fmt(star_eval['J'])}  "
        f"J_global(deepgini)={_fmt(gini_eval['J'])}",
    )
    for ds in dataset_list:
        m = dev_by_dataset[ds]
        g = gini_by_dataset[ds]
        deltas = [
            f"{_budget_key(b)}={_fmt(m['trc_random'][_budget_key(b)] - g['trc_random'][_budget_key(b)])}"
            for b in budgets
        ]
        print(f"[search:global]   {ds} delta_rand: {', '.join(deltas)}")
    print(f'[search:global] summary -> {summary_path}')
    return best_payload, summary


def run_feature_search(
    feature_cache_dir,
    sampled_root,
    datasets,
    pool_types,
    error_ratios,
    seeds,
    sampling_modes,
    summary_path=None,
    *,
    search_scope='global',
    lambda_high=SEARCH_LAMBDA_HIGH,
    eps_stop=SEARCH_EPS_STOP,
    delta_prune=SEARCH_DELTA_PRUNE,
    keep_deepgini=False,
    enable_family_soft_penalty=False,
    delta_family=SEARCH_DELTA_FAMILY,
    enable_random_floor=True,
    random_floor_eps=SEARCH_RANDOM_FLOOR_EPS,
    budgets=SEARCH_BUDGET_RATIOS,
    budget_weights=None,
):
    """
    Feature-subset search.

    search_scope:
      - global: pool-level J_global + random floor → cache_files/global_*
      - per_dataset: each dataset → cache_files/{ds}/feature_search/ + feature_exploration_summary
      - both: global then per_dataset (rewrite global summary with adaptive contrast)
    """
    if budget_weights is None:
        budget_weights = dict(SEARCH_BUDGET_WEIGHTS)
    if search_scope not in ('global', 'per_dataset', 'both'):
        raise ValueError(f'unknown search_scope={search_scope!r}')

    feature_cache_dir = Path(feature_cache_dir)
    available_seeds, dev_seeds, test_seeds, seed_note = _resolve_search_seeds(seeds)
    dataset_list = list(datasets)
    result = {'search_scope': search_scope}

    common_kw = dict(
        lambda_high=lambda_high,
        eps_stop=eps_stop,
        delta_prune=delta_prune,
        keep_deepgini=keep_deepgini,
        enable_family_soft_penalty=enable_family_soft_penalty,
        delta_family=delta_family,
        budgets=budgets,
        budget_weights=budget_weights,
        enable_random_floor=enable_random_floor,
        random_floor_eps=random_floor_eps,
    )

    global_payload = None
    global_summary_path = feature_cache_dir / 'global_feature_exploration_summary.md'
    if search_scope == 'global' and summary_path is not None:
        global_summary_path = Path(summary_path)

    if search_scope in ('global', 'both'):
        print('[search] ===== scope=global =====')
        global_payload, global_summary = _run_feature_search_global(
            feature_cache_dir=feature_cache_dir,
            sampled_root=sampled_root,
            datasets=dataset_list,
            pool_types=pool_types,
            error_ratios=error_ratios,
            available_seeds=available_seeds,
            dev_seeds=dev_seeds,
            test_seeds=test_seeds,
            seed_note=seed_note,
            sampling_modes=sampling_modes,
            summary_path=global_summary_path,
            per_dataset_adaptive=None,
            **common_kw,
        )
        result['global'] = global_payload
        result['global_summary'] = global_summary
        result['global_summary_path'] = str(global_summary_path)

    per_dataset = None
    if search_scope in ('per_dataset', 'both'):
        adaptive_summary_path = feature_cache_dir / 'feature_exploration_summary.md'
        if search_scope == 'per_dataset' and summary_path is not None:
            adaptive_summary_path = Path(summary_path)
        per_dataset = {}
        per_dataset_dirs = {}
        for dataset_name in dataset_list:
            ds_out = feature_cache_dir / dataset_name / 'feature_search'
            per_dataset_dirs[dataset_name] = ds_out
            print(f'[search] ===== per_dataset={dataset_name} =====')
            per_dataset[dataset_name] = _run_feature_search_one_dataset(
                dataset_name=dataset_name,
                feature_cache_dir=feature_cache_dir,
                sampled_root=sampled_root,
                pool_types=pool_types,
                error_ratios=error_ratios,
                available_seeds=available_seeds,
                dev_seeds=dev_seeds,
                test_seeds=test_seeds,
                seed_note=seed_note,
                sampling_modes=sampling_modes,
                out_dir=ds_out,
                **common_kw,
            )
        adaptive_summary = _write_cross_dataset_summary(
            adaptive_summary_path,
            per_dataset,
            budgets=budgets,
            per_dataset_dirs=per_dataset_dirs,
        )
        print(f"[search] per_dataset verdict: {adaptive_summary['verdict']}")
        print(f'[search] per_dataset summary -> {adaptive_summary_path}')
        result['per_dataset'] = per_dataset
        result['per_dataset_summary'] = adaptive_summary
        result['per_dataset_summary_path'] = str(adaptive_summary_path)

        if search_scope == 'both' and global_payload is not None:
            # rewrite global summary with adaptive contrast section
            result['global_summary'] = _write_global_feature_exploration_summary(
                global_summary_path,
                global_payload,
                budgets=budgets,
                per_dataset_adaptive=per_dataset,
            )

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Risk feature exploration: compute features, univariate analysis, '
            'correlation, and greedy feature-subset search.'
        ),
    )
    parser.add_argument(
        '--phase',
        choices=('compute', 'analyze', 'correlate', 'search', 'all'),
        default='all',
        help='compute | analyze | correlate | search | compute+analyze+correlate',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=list(EXPLORE_DATASETS),
        choices=sorted(DATASET_CONFIG),
    )
    parser.add_argument('--pool-types', nargs='+', default=list(EXPLORE_POOL_TYPES), choices=list(EXPLORE_POOL_TYPES))
    parser.add_argument('--error-ratios', nargs='+', type=float, default=list(EXPLORE_ERROR_RATIOS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(EXPLORE_SEEDS))
    parser.add_argument(
        '--sampling-modes',
        nargs='+',
        default=list(EXPLORE_SAMPLING_MODES),
        choices=list(EXPLORE_SAMPLING_MODES),
    )
    parser.add_argument('--sampled-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--cache-root', default=str(CACHE_ROOT))
    parser.add_argument(
        '--output-dir',
        default=str(CACHE_ROOT),
        help='directory for univariate analysis outputs (default: experiments/cache_files)',
    )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument(
        '--corr-min-abs',
        type=float,
        default=0.85,
        help='hierarchical cluster cut: group features with |Spearman rho| >= this value',
    )
    parser.add_argument(
        '--corr-linkage',
        default='average',
        choices=('average', 'complete', 'single', 'weighted'),
        help='linkage method for feature clustering',
    )
    parser.add_argument(
        '--corr-min-errors',
        type=int,
        default=3,
        help='minimum error samples in a pool to compute correlation',
    )
    parser.add_argument(
        '--search-scope',
        choices=('global', 'per_dataset', 'both'),
        default='global',
        help='feature search scope (default: global)',
    )
    parser.add_argument(
        '--summary-path',
        default=None,
        help=(
            'optional summary markdown path; default depends on --search-scope '
            '(global_feature_exploration_summary.md or feature_exploration_summary.md)'
        ),
    )
    parser.add_argument(
        '--search-lambda-high',
        type=float,
        default=SEARCH_LAMBDA_HIGH,
        help='weight on high_conf TRC secondary term in J',
    )
    parser.add_argument('--eps-stop', type=float, default=SEARCH_EPS_STOP)
    parser.add_argument('--delta-prune', type=float, default=SEARCH_DELTA_PRUNE)
    parser.add_argument(
        '--keep-deepgini',
        action='store_true',
        help='never remove deepgini during backward prune',
    )
    parser.add_argument(
        '--enable-family-soft-penalty',
        action='store_true',
        help='optional same-family soft filter during forward greedy',
    )
    parser.add_argument('--delta-family', type=float, default=SEARCH_DELTA_FAMILY)
    parser.add_argument(
        '--disable-random-floor',
        action='store_true',
        help='disable global random TRC floor vs DeepGini during forward greedy',
    )
    parser.add_argument(
        '--random-floor-eps',
        type=float,
        default=SEARCH_RANDOM_FLOOR_EPS,
        help='max allowed drop of any dataset random mean TRC vs DeepGini (default: 0.03)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.seeds = (0, 1, 2, 3, 4)
    args.phase = 'compute'
    args.search_scope = 'both'
    if args.phase in ('compute', 'all'):
        run_feature_compute(
            sampled_root=args.sampled_root,
            datasets=args.datasets,
            pool_types=args.pool_types,
            error_ratios=args.error_ratios,
            seeds=args.seeds,
            sampling_modes=args.sampling_modes,
            cache_root=args.cache_root,
            batch_size=args.batch_size,
            force_recompute=args.force_recompute,
        )
    if args.phase in ('analyze', 'all'):
        run_univariate_analysis(
            feature_cache_dir=args.cache_root,
            sampled_root=args.sampled_root,
            datasets=args.datasets,
            pool_types=args.pool_types,
            error_ratios=args.error_ratios,
            seeds=args.seeds,
            sampling_modes=args.sampling_modes,
            output_dir=args.output_dir,
        )
    if args.phase in ('correlate', 'all'):
        run_correlation_analysis(
            feature_cache_dir=args.cache_root,
            sampled_root=args.sampled_root,
            datasets=args.datasets,
            pool_types=args.pool_types,
            error_ratios=args.error_ratios,
            seeds=args.seeds,
            sampling_modes=args.sampling_modes,
            cache_root=args.cache_root,
            min_abs_corr=args.corr_min_abs,
            linkage_method=args.corr_linkage,
            min_errors=args.corr_min_errors,
        )
    if args.phase in ('search', 'all'):
        run_feature_search(
            feature_cache_dir=args.cache_root,
            sampled_root=args.sampled_root,
            datasets=args.datasets,
            pool_types=args.pool_types,
            error_ratios=args.error_ratios,
            seeds=args.seeds,
            sampling_modes=args.sampling_modes,
            summary_path=args.summary_path,
            search_scope=args.search_scope,
            lambda_high=args.search_lambda_high,
            eps_stop=args.eps_stop,
            delta_prune=args.delta_prune,
            keep_deepgini=args.keep_deepgini,
            enable_family_soft_penalty=args.enable_family_soft_penalty,
            delta_family=args.delta_family,
            enable_random_floor=not args.disable_random_floor,
            random_floor_eps=args.random_floor_eps,
        )


if __name__ == '__main__':
    main()