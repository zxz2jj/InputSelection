import importlib
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from training_models.load_data import load_fmnist, load_cifar10


def _keras_preprocessing_layer(class_name, *args, **kwargs):
    if hasattr(tf.keras.layers, class_name):
        cls = getattr(tf.keras.layers, class_name)
        return cls(*args, **kwargs)
    for mod_name in (
        'tensorflow.keras.layers.experimental.preprocessing',
        'keras.layers.experimental.preprocessing',
    ):
        try:
            pre = importlib.import_module(mod_name)
            cls = getattr(pre, class_name)
            return cls(*args, **kwargs)
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        f'ImportError: {class_name} not found',
    )

_rotation_augment = _keras_preprocessing_layer(
    'RandomRotation', 0.03, fill_mode='nearest', interpolation='bilinear')
_translation_augment = _keras_preprocessing_layer(
    'RandomTranslation', 0.05, 0.05, fill_mode='nearest', interpolation='bilinear')
_contrast_augment = _keras_preprocessing_layer('RandomContrast', 0.20)


def keras_logits_fn(model, logits_layer_index=-2):
    logits_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[logits_layer_index].output,
        name=model.name + '_logits' if model.name else 'logits_submodel',
    )

    def _logit_fn(batch_tensor):
        return logits_model(batch_tensor, training=False)

    return _logit_fn


def _probs_from_model_output_tensor(raw_out):
    raw_out = tf.cast(raw_out, tf.float32)
    row_sums = tf.reduce_sum(raw_out, axis=-1)
    looks_like_probs = tf.reduce_all((raw_out >= 0.0) & (raw_out <= 1.0)) & tf.reduce_all(
        tf.abs(row_sums - 1.0) < 1e-2
    )
    return tf.cond(looks_like_probs, lambda: raw_out, lambda: tf.nn.softmax(raw_out, axis=-1))


def _hidden_to_flat_batch(hid, hidden_layer):

    hid = tf.cast(hid, tf.float32)
    b = tf.shape(hid)[0]

    if isinstance(hidden_layer, tf.keras.layers.Dense):
        return tf.reshape(hid, [b, -1])

    _conv_types = (
        tf.keras.layers.Conv1D,
        tf.keras.layers.Conv2D,
        tf.keras.layers.Conv3D,
        tf.keras.layers.Conv2DTranspose,
        tf.keras.layers.Conv3DTranspose,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv1D,
        tf.keras.layers.SeparableConv2D,
    )
    if isinstance(hidden_layer, _conv_types):
        rank = hid.shape.rank
        if rank == 4:
            return tf.reduce_mean(hid, axis=[1, 2])
        if rank == 5:
            return tf.reduce_mean(hid, axis=[1, 2, 3])
        if rank == 3:
            return tf.reduce_mean(hid, axis=1)
        return tf.reshape(hid, [b, -1])

    if isinstance(
        hidden_layer,
        (
            tf.keras.layers.Flatten,
            tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.GlobalAveragePooling3D,
        ),
    ):
        return tf.reshape(hid, [b, -1])

    rank = hid.shape.rank
    if rank is None:
        return tf.reshape(hid, [b, -1])
    if rank == 2:
        return tf.reshape(hid, [b, -1])
    if rank == 3:
        return tf.reduce_mean(hid, axis=1)
    if rank == 4:
        return tf.reduce_mean(hid, axis=[1, 2])
    if rank == 5:
        return tf.reduce_mean(hid, axis=[1, 2, 3])
    return tf.reshape(hid, [b, -1])


def _batch_distance_risk_features(hid_flat, pred, prototypes):

    hid_flat = np.asarray(hid_flat, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    B = hid_flat.shape[0]
    C = prototypes.shape[0]
    valid_proto = ~np.isnan(prototypes).any(axis=1)
    eps = 1e-8

    dist_pred_proto = np.full(B, np.nan, dtype=np.float64)
    dist_ratio = np.full(B, np.nan, dtype=np.float64)

    # 可以向量化优化时间复杂度
    for i in range(B):
        h = hid_flat[i]
        p = int(pred[i])
        if p < 0 or p >= C or not valid_proto[p]:
            continue
        d_all = np.linalg.norm(prototypes - h, axis=1)
        d_p = float(d_all[p])
        dist_pred_proto[i] = d_p
        mask = valid_proto & (np.arange(C, dtype=np.int64) != p)
        if not np.any(mask):
            continue
        d_min_other = float(np.min(d_all[mask]))
        dist_ratio[i] = d_p / (d_min_other + eps)

    return dist_pred_proto.astype(np.float32), dist_ratio.astype(np.float32)


# 可以与上一个函数合并，优化时间复杂度
def _batch_pred_class_is_min_distance(hid_flat, pred, prototypes):
    """
    对每个样本判断：在当前层上，预测类别原型距离是否为最小（有效类别内）。
    返回 0/1 浮点数组，1 表示一致，0 表示不一致。
    """
    hid_flat = np.asarray(hid_flat, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    B = hid_flat.shape[0]
    C = prototypes.shape[0]
    valid_proto = ~np.isnan(prototypes).any(axis=1)
    tol = 1e-8

    is_min = np.zeros(B, dtype=np.float32)
    for i in range(B):
        p = int(pred[i])
        if p < 0 or p >= C or not valid_proto[p]:
            continue
        d_all = np.linalg.norm(prototypes - hid_flat[i], axis=1)
        if not np.any(valid_proto):
            continue
        d_pred = float(d_all[p])
        d_min = float(np.min(d_all[valid_proto]))
        if d_pred <= d_min + tol:
            is_min[i] = 1.0
    return is_min


def build_or_load_class_prototypes(
    model,
    train_data,
    train_labels,
    layer_index,
    batch_size=64,
):

    if layer_index is None:
        raise TypeError('layer_index is None')

    y_arr = np.asarray(train_labels)
    if y_arr.ndim > 1:
        y_flat = np.argmax(y_arr, axis=-1).reshape(-1)
    else:
        y_flat = y_arr.reshape(-1)
    y_flat = y_flat.astype(np.int64, copy=False)
    num_classes = int(y_flat.max()) + 1

    try:
        hidden_layer = model.layers[layer_index]
    except IndexError as e:
        raise ValueError(
            f'layer_index={layer_index!r} out of range (model has {len(model.layers)} layers)',
        ) from e

    # 双输出一次前向：若拆成「仅隐藏层」子模型再单独 model(x)，每个 batch 会跑两遍网络。
    forward = tf.keras.Model(
        inputs=model.input,
        outputs=[hidden_layer.output, model.output],
        name='class_prototype_forward',
    )

    sums = None
    counts = np.zeros(num_classes, dtype=np.int64)

    for start in tqdm(range(0, len(train_data), batch_size), desc=f'class prototypes:{layer_index}'):
        end = min(start + batch_size, len(train_data))
        bx = train_data[start:end]
        by = y_flat[start:end]
        bt = tf.convert_to_tensor(bx)
        hid, raw_out = forward(bt, training=False)
        probs = _probs_from_model_output_tensor(raw_out)
        pred = tf.argmax(probs, axis=-1).numpy()
        hid_flat = _hidden_to_flat_batch(hid, hidden_layer).numpy().astype(np.float64, copy=False)

        if sums is None:
            sums = np.zeros((num_classes, hid_flat.shape[1]), dtype=np.float64)

        correct = pred == by
        for c in range(num_classes):
            mask = correct & (by == c)
            if np.any(mask):
                sums[c] += hid_flat[mask].sum(axis=0)
                counts[c] += int(np.sum(mask))

    prototypes = np.full_like(sums, np.nan, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] = sums[c] / counts[c]

    prototypes_f32 = prototypes.astype(np.float32)
    return prototypes_f32


def build_or_load_class_prototypes_dict(
    model,
    train_data,
    train_labels,
    layer_indices,
    dataset_name,
    batch_size=64,
    force_recompute=False,
):

    if layer_indices is None:
        raise TypeError('layer_indices is None')
    unique_layer_indices = list(dict.fromkeys([int(i) for i in layer_indices]))
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / 'data' / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / 'class_prototypes_by_layer.npz'

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        cached_indices = [int(i) for i in np.asarray(z['layer_indices']).tolist()]
        if cached_indices == unique_layer_indices:
            out = {}
            for layer_index in unique_layer_indices:
                key = f'layer_{layer_index}'
                if key not in z.files:
                    raise ValueError(f'cache file missing key {key}: {cache_path}')
                out[layer_index] = np.asarray(z[key], dtype=np.float32)
            print(f'Loaded class prototypes dict from {cache_path}')
            return out
        print(
            f'Cached layer_indices {cached_indices} mismatch requested {unique_layer_indices}, recomputing...',
        )

    # 这里可以优化，一次推理拿到多个层的信息而不用循环多次推理
    out = {}
    for layer_index in unique_layer_indices:
        out[layer_index] = build_or_load_class_prototypes(
            model=model,
            train_data=train_data,
            train_labels=train_labels,
            layer_index=layer_index,
            batch_size=batch_size,
        )

    save_dict = {'layer_indices': np.asarray(unique_layer_indices, dtype=np.int32)}
    for layer_index, proto in out.items():
        save_dict[f'layer_{layer_index}'] = np.asarray(proto, dtype=np.float32)
    np.savez_compressed(cache_path, **save_dict)
    print(f'Saved class prototypes dict to {cache_path}')
    return out


def _gaussian_kernel2d(size, sigma):
    coords = tf.range(size, dtype=tf.float32) - tf.cast(size - 1, tf.float32) / 2.0
    g = tf.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = tf.tensordot(g, g, axes=0)
    return g / tf.reduce_sum(g)


def _blur_augment(x):
    x = tf.cast(x, tf.float32)
    sigma = tf.random.uniform((), 0.35, 0.9)
    k = 5
    ker2 = _gaussian_kernel2d(k, sigma)
    in_ch = tf.shape(x)[-1]
    ker4 = tf.reshape(ker2, [k, k, 1, 1])
    ker4 = tf.tile(ker4, [1, 1, in_ch, 1])
    pad = k // 2
    x_pad = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
    out = tf.nn.depthwise_conv2d(
        x_pad, ker4, strides=[1, 1, 1, 1], padding='VALID')
    return tf.clip_by_value(out, 0.0, 1.0)


def image_data_augmentation(x, transform_id=None):

    x = tf.cast(x, tf.float32)
    if transform_id is None:
        transform_id = random.randint(1, 5)

    if transform_id == 1:
        delta = tf.random.uniform((), -0.12, 0.12)
        return tf.clip_by_value(tf.image.adjust_brightness(x, delta), 0.0, 1.0)
    if transform_id == 2:
        return tf.clip_by_value(_rotation_augment(x, training=True), 0.0, 1.0)
    if transform_id == 3:
        return tf.clip_by_value(_translation_augment(x, training=True), 0.0, 1.0)
    if transform_id == 4:
        return tf.clip_by_value(_contrast_augment(x, training=True), 0.0, 1.0)
    if transform_id == 5:
        return _blur_augment(x)

    raise ValueError(f'unknown transform_id={transform_id!r} (expected 1–5 or None)')


def get_risk_features(data_without_labelling,
                      model,
                      prototypes_by_layer_map,
                      distance_feature_layer_index,
                      consistency_feature_layer_indices,
                      logits_layer_index=-2,
                      batch_size=16,
                      num_augmentations=5,
                      augment_repeats_per_transform=3,
                      ):

    if data_without_labelling is None:
        raise TypeError('data_without_labelling is None')
    if model is None:
        raise TypeError('model is None')
    if prototypes_by_layer_map is None:
        raise TypeError('prototypes_by_layer_map is None')
    if consistency_feature_layer_indices is None:
        raise TypeError('consistency_feature_layer_indices is None')
    if distance_feature_layer_index is None:
        raise TypeError('distance_feature_layer_index is None')
    if augment_repeats_per_transform is None or int(augment_repeats_per_transform) < 1:
        raise ValueError('augment_repeats_per_transform must be >= 1')
    augment_repeats_per_transform = int(augment_repeats_per_transform)

    logit_fn = keras_logits_fn(model, logits_layer_index=logits_layer_index)

    entropy_list = []
    energy_list = []
    top12_margin_risk_list = []
    pred_classes_list = []
    stability_flip_list = []
    stability_maxvar_list = []
    stability_mkl_list = []
    dist_pred_proto_list = []
    dist_ratio_list = []
    dist_layer_inconsistency_list = []

    consistency_feature_layer_indices = [int(i) for i in consistency_feature_layer_indices]
    hidden_layer_indices = list(dict.fromkeys([int(distance_feature_layer_index)] + consistency_feature_layer_indices))

    hidden_layers = {}
    hidden_prototypes = {}
    for idx in hidden_layer_indices:
        if idx not in prototypes_by_layer_map:
            raise ValueError(f'prototypes_by_layer_map missing layer_index={idx}')
        try:
            hidden_layers[idx] = model.layers[idx]
        except IndexError as e:
            raise ValueError(
                f'layer_index={idx!r} out of range (model has {len(model.layers)} layers)',
            ) from e
        hidden_prototypes[idx] = np.asarray(prototypes_by_layer_map[idx], dtype=np.float32)

    logits_layer = None
    try:
        logits_layer = model.layers[logits_layer_index]
    except IndexError as e:
        raise ValueError(
            f'logits_layer_index={logits_layer_index!r} out of range '
            f'(model has {len(model.layers)} layers)',
        ) from e

    # 原始样本一次前向同时拿到多层隐藏表示与 logits，避免重复推理。
    base_forward = tf.keras.Model(
        inputs=model.input,
        outputs=[hidden_layers[idx].output for idx in hidden_layer_indices] + [logits_layer.output],
        name='risk_base_forward',
    )

    eps = tf.constant(1e-8, tf.float32)
    _proto_dim_checked = set()

    for i in tqdm(range(0, len(data_without_labelling), batch_size), desc='computing risk features'):
        batch = data_without_labelling[i:i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch)

        forward_outs = base_forward(batch_tensor, training=False)
        hid_outs = forward_outs[:-1]
        logits = forward_outs[-1]
        logits = tf.cast(tf.convert_to_tensor(logits), tf.float32)
        probs = tf.nn.softmax(logits, axis=-1)
        prediction_entropy = -tf.reduce_sum(tf.math.xlogy(probs, probs), axis=-1)
        energy_score = -tf.math.reduce_logsumexp(logits, axis=-1)

        top2_vals, _ = tf.nn.top_k(probs, k=2)
        top12_margin_risk = 1.0 - (top2_vals[:, 0] - top2_vals[:, 1])

        pred_classes = tf.argmax(probs, axis=-1)
        pred_np = pred_classes.numpy()

        hid_flat_by_layer = {}
        for idx, hid in zip(hidden_layer_indices, hid_outs):
            hid_flat = _hidden_to_flat_batch(hid, hidden_layers[idx]).numpy()
            hid_flat_by_layer[idx] = hid_flat
            if idx not in _proto_dim_checked and hid_flat.shape[1] != hidden_prototypes[idx].shape[1]:
                raise ValueError(
                    f'layer {idx}: hidden dim {hid_flat.shape[1]} != prototypes dim {hidden_prototypes[idx].shape[1]}',
                )
            _proto_dim_checked.add(idx)

        dp, dr = _batch_distance_risk_features(
            hid_flat_by_layer[int(distance_feature_layer_index)],
            pred_np,
            hidden_prototypes[int(distance_feature_layer_index)],
        )

        layer_consistency_flags = []
        for idx in consistency_feature_layer_indices:
            layer_consistency_flags.append(
                _batch_pred_class_is_min_distance(
                    hid_flat_by_layer[idx],
                    pred_np,
                    hidden_prototypes[idx],
                ),
            )
        layer_consistency_flags = np.stack(layer_consistency_flags, axis=1)
        inter_layer_inconsistency = 1.0 - np.mean(layer_consistency_flags, axis=1)

        aug_prob_list = []
        chosen_ids = np.arange(1, num_augmentations+1)
        for tid in chosen_ids:
            for _ in range(augment_repeats_per_transform):
                x_aug = image_data_augmentation(batch_tensor, transform_id=tid)
                lg = tf.cast(tf.convert_to_tensor(logit_fn(x_aug)), tf.float32)
                aug_prob_list.append(tf.nn.softmax(lg, axis=-1))
        aug_probs = tf.stack(aug_prob_list, axis=1)

        aug_preds = tf.argmax(aug_probs, axis=-1)
        pred_orig = tf.expand_dims(pred_classes, 1)
        stability_class_change_rate = tf.reduce_mean(
            tf.cast(tf.not_equal(pred_orig, aug_preds), tf.float32),
            axis=1,
        )

        max_per_aug = tf.reduce_max(aug_probs, axis=-1)
        stability_max_prob_variance = tf.math.reduce_variance(max_per_aug, axis=1)

        p_orig = probs[:, tf.newaxis, :]
        log_p = tf.math.log(aug_probs + eps)
        log_q = tf.math.log(p_orig + eps)
        kl_per_aug = tf.reduce_sum(aug_probs * (log_p - log_q), axis=-1)
        stability_mean_kl = tf.reduce_mean(kl_per_aug, axis=1)

        entropy_list.append(np.asarray(prediction_entropy.numpy(), dtype=np.float32).reshape(-1))
        energy_list.append(np.asarray(energy_score.numpy(), dtype=np.float32).reshape(-1))
        top12_margin_risk_list.append(np.asarray(top12_margin_risk.numpy(), dtype=np.float32).reshape(-1))
        pred_classes_list.append(np.asarray(pred_classes.numpy(), dtype=np.int64).reshape(-1))
        stability_flip_list.append(np.asarray(stability_class_change_rate.numpy(), dtype=np.float32).reshape(-1))
        stability_maxvar_list.append(np.asarray(stability_max_prob_variance.numpy(), dtype=np.float32).reshape(-1))
        stability_mkl_list.append(np.asarray(stability_mean_kl.numpy(), dtype=np.float32).reshape(-1))
        dist_pred_proto_list.append(dp.reshape(-1))
        dist_ratio_list.append(dr.reshape(-1))
        dist_layer_inconsistency_list.append(inter_layer_inconsistency.astype(np.float32).reshape(-1))

    out = {
        'prediction_entropy': np.concatenate(entropy_list),
        'energy_score': np.concatenate(energy_list),
        'top12_margin': np.concatenate(top12_margin_risk_list),
        'pred_classes': np.concatenate(pred_classes_list),
        'stability_class_change_rate': np.concatenate(stability_flip_list),
        'stability_max_prob_variance': np.concatenate(stability_maxvar_list),
        'stability_mean_kl': np.concatenate(stability_mkl_list),
        'dist_pred_class_prototype': np.concatenate(dist_pred_proto_list),
        'dist_ratio_pred_to_nearest_other_prototype': np.concatenate(dist_ratio_list),
        'dist_layer_inconsistency': np.concatenate(dist_layer_inconsistency_list),
    }
    return out


def build_or_load_risk_features(
    cache_dir,
    cache_name,
    data,
    model,
    prototypes_by_layer_map,
    distance_feature_layer_index,
    consistency_feature_layer_indices,
    logits_layer_index=-2,
    batch_size=16,
    num_augmentations=5,
    augment_repeats_per_transform=3,
    force_recompute=False,
):

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        out = {k: np.asarray(z[k]) for k in z.files}
        print(f'Loaded risk features from {cache_path}')
        return out

    out = get_risk_features(
        data_without_labelling=data,
        model=model,
        prototypes_by_layer_map=prototypes_by_layer_map,
        distance_feature_layer_index=distance_feature_layer_index,
        consistency_feature_layer_indices=consistency_feature_layer_indices,
        logits_layer_index=logits_layer_index,
        batch_size=batch_size,
        num_augmentations=num_augmentations,
        augment_repeats_per_transform=augment_repeats_per_transform,
    )
    np.savez_compressed(cache_path, **out)
    print(f'Saved risk features to {cache_path}')
    return out


BINS_BY_FEATURE = {
    'prediction_entropy': 30,
    'energy_score': 30,
    'top12_margin': 30,
    'dist_pred_class_prototype': 30,
    'dist_ratio_pred_to_nearest_other_prototype': 30,
    'dist_layer_inconsistency': 5,
    'stability_class_change_rate': 15,
    'stability_max_prob_variance': 30,
    'stability_mean_kl': 30,
}


def plot_risk_feature_distributions(
    named_feature_dicts,
    feature_key,
    save_dir,
    bins_by_feature=None,
    ncols=5,
):
    """
    在一个画布上绘制同一风险特征在不同数据子集上的分布柱状图（直方图），每行 ncols 个子图。
    named_feature_dicts: [(name, feature_dict_or_None), ...]
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    bins_by_feature = bins_by_feature or {}
    if feature_key not in bins_by_feature:
        raise KeyError(f'Missing bins config for feature: {feature_key}')
    bins = int(bins_by_feature[feature_key])

    n = len(named_feature_dicts)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    pooled = []
    for _, feature_dict in named_feature_dicts:
        if feature_dict is None or feature_key not in feature_dict:
            continue
        vals = np.asarray(feature_dict[feature_key], dtype=np.float32).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            pooled.append(vals)
    if pooled:
        all_vals = np.concatenate(pooled)
        vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
        bin_edges = np.linspace(vmin, vmax, bins + 1)
    else:
        bin_edges = np.linspace(0.0, 1.0, bins + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for i, (name, feature_dict) in enumerate(named_feature_dicts):
        ax = axes[i]
        if feature_dict is None or feature_key not in feature_dict:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=9)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        vals = np.asarray(feature_dict[feature_key], dtype=np.float32).reshape(-1)
        vals = vals[np.isfinite(vals)]
        counts, edges = np.histogram(vals, bins=bin_edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = (edges[1:] - edges[:-1]) * 0.9
        ax.bar(centers, counts, width=widths, alpha=0.85)
        ax.set_title(name)
        ax.tick_params(labelsize=8)
    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{feature_key} distribution', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = save_dir / f'distribution_{feature_key}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved plot: {out_path}')


if __name__ == "__main__":
    data_name = 'fmnist'
    # dataset_name = 'cifar10'

    if data_name == 'fmnist':
        model_path = '../models/lenet_fmnist/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        x_train, y_train, x_test, y_test = load_fmnist()
        adv_dir = Path('../data/fmnist/adversarial')
        distance_layer_index = -4
        consistency_layer_indices = [-10, -8, -6, -4]
    elif data_name == 'cifar10':
        model_path = '../models/vgg19_cifar10/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        x_train, y_train, x_test, y_test = load_cifar10()
        adv_dir = Path('../data/cifar10/adversarial')
        distance_layer_index = None
        consistency_layer_indices = None
    else:
        exit()
    if np.asarray(y_train).ndim > 1:
        y_train = np.argmax(y_train, axis=-1)
    if np.asarray(y_test).ndim > 1:
        y_test = np.argmax(y_test, axis=-1)
    
    risk_features_cache_dir = (Path(__file__).resolve().parent.parent / 'data' / data_name)

    # 主距离层 + 层间一致性层（可按模型结构调整）
    all_prototype_layers = list(dict.fromkeys([distance_layer_index] + consistency_layer_indices))
    prototypes_by_layer = build_or_load_class_prototypes_dict(
        cnn_model,
        train_data=x_train,
        train_labels=y_train,
        layer_indices=all_prototype_layers,
        dataset_name=data_name,
        batch_size=64,
    )
    print('prototypes_by_layer', {k: v.shape for k, v in prototypes_by_layer.items()})

    y_test_pred = np.argmax(cnn_model.predict(x_test, batch_size=64, verbose=0), axis=-1)
    correct_mask = (y_test_pred == y_test)
    correct_x_test = x_test[correct_mask]
    wrong_x_test = x_test[~correct_mask]
    print(f'{data_name} -> correct: {len(correct_x_test)}, wrong: {len(wrong_x_test)}')

    if len(correct_x_test) > 0:
        print('computing correct_x_test risk features')
        correct_risk_features = build_or_load_risk_features(
            cache_dir=risk_features_cache_dir,
            cache_name='risk_features_correct_x_test.npz',
            data=correct_x_test,
            model=cnn_model,
            prototypes_by_layer_map=prototypes_by_layer,
            distance_feature_layer_index=distance_layer_index,
            consistency_feature_layer_indices=consistency_layer_indices,
            batch_size=16,
        )
        print('correct_x_test risk features')
    else:
        print('correct_x_test is empty, skip risk feature computation')
        correct_risk_features = None

    if len(wrong_x_test) > 0:
        print('computing wrong_x_test risk features')
        wrong_risk_features = build_or_load_risk_features(
            cache_dir=risk_features_cache_dir,
            cache_name='risk_features_wrong_x_test.npz',
            data=wrong_x_test,
            model=cnn_model,
            prototypes_by_layer_map=prototypes_by_layer,
            distance_feature_layer_index=distance_layer_index,
            consistency_feature_layer_indices=consistency_layer_indices,
            batch_size=16,
        )
        print('wrong_x_test risk features')
    else:
        print('wrong_x_test is empty, skip risk feature computation')
        wrong_risk_features = None

    plot_groups = [
        ('correct_x_test', correct_risk_features),
        ('wrong_x_test', wrong_risk_features),
    ]

    adv_files = sorted(adv_dir.glob('*_adv_data.npy'))
    if not adv_files:
        print(f'No files matching *_adv_data.npy under {adv_dir.resolve()}')
    for adv_path in adv_files:
        adv_data = np.load(adv_path)
        adv_prefix = adv_path.name.removesuffix('_adv_data.npy')
        print(f'{adv_path.name}: shape={getattr(adv_data, "shape", None)}')
        print('computing adv_data risk features')
        risk_features = build_or_load_risk_features(
            cache_dir=risk_features_cache_dir,
            cache_name=f'risk_features_{adv_prefix}.npz',
            data=adv_data,
            model=cnn_model,
            prototypes_by_layer_map=prototypes_by_layer,
            distance_feature_layer_index=distance_layer_index,
            consistency_feature_layer_indices=consistency_layer_indices,
            batch_size=16,
        )
        plot_groups.append((adv_prefix, risk_features))

    first_available = next((d for _, d in plot_groups if d is not None), None)
    if first_available is not None:
        plot_dir = risk_features_cache_dir / 'plots'
        for feature_name in BINS_BY_FEATURE.keys():
            if feature_name not in first_available:
                continue
            plot_risk_feature_distributions(
                named_feature_dicts=plot_groups,
                feature_key=feature_name,
                save_dir=plot_dir,
                bins_by_feature=BINS_BY_FEATURE,
                ncols=5,
            )

    