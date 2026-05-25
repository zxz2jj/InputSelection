import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from snorkel.labeling.model import LabelModel

from training_models.load_data import load_fmnist, load_cifar10

ABSTAIN = -1


def _as_1d_int(arr, name):
    out = np.asarray(arr, dtype=np.int64).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f'{name} must be 1D')
    return out


def _as_1d_float(arr, name):
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f'{name} must be 1D')
    return out


def lf_non_pred_softmax_gap(
    non_pred_top1_class,
    non_pred_top1_prob,
    non_pred_top2_prob,
    gap_threshold=0.05,
):
    """
    LF1: 当 (top1_non_pred_prob - top2_non_pred_prob) >= gap_threshold 时发言，
    输出 top1 非预测类，否则 ABSTAIN(-1)。
    """
    c1 = _as_1d_int(non_pred_top1_class, 'non_pred_top1_class')
    p1 = _as_1d_float(non_pred_top1_prob, 'non_pred_top1_prob')
    p2 = _as_1d_float(non_pred_top2_prob, 'non_pred_top2_prob')
    if not (len(c1) == len(p1) == len(p2)):
        raise ValueError('LF1 input lengths must match')

    labels = np.full(len(c1), ABSTAIN, dtype=np.int64)
    valid = (c1 >= 0) & np.isfinite(p1) & np.isfinite(p2)
    strong = (p1 - p2) >= float(gap_threshold)
    labels[valid & strong] = c1[valid & strong]
    return labels


def lf_multi_layer_nearest_non_pred_consensus(
    feature_map,
    layer_indices,
):
    """
    LF2: 多层最近非预测类一致性。
    1) 统计多个层上的最近非预测类投票，输出票数最高类别；
    2) 若参与投票层数为偶数且第一、第二票数平票 -> ABSTAIN。
    """
    layer_arrays = []
    for idx in layer_indices:
        cls_key = f'nearest_non_pred_class_layer_{int(idx)}'
        if cls_key not in feature_map:
            raise KeyError(f'Missing key in feature_map: {cls_key}')
        layer_arrays.append(_as_1d_int(feature_map[cls_key], cls_key))

    if len(layer_arrays) == 0:
        raise ValueError('No layers provided for LF2')

    n = len(layer_arrays[0])
    for idx, x in enumerate(layer_arrays[1:], start=1):
        if len(x) != n:
            raise ValueError(f'Layer length mismatch at layer {idx}')

    labels = np.full(n, ABSTAIN, dtype=np.int64)
    for i in range(n):
        votes = []
        for layer_vote in layer_arrays:
            cls = int(layer_vote[i])
            if cls < 0:
                continue
            votes.append(cls)

        if len(votes) == 0:
            continue

        vote_vals, vote_cnts = np.unique(np.asarray(votes, dtype=np.int64), return_counts=True)
        order = np.argsort(vote_cnts)[::-1]
        top1_cnt = int(vote_cnts[order[0]])
        top2_cnt = int(vote_cnts[order[1]]) if len(order) > 1 else -1
        tied_top2 = top1_cnt == top2_cnt

        if (len(votes) % 2 == 0) and tied_top2:
            labels[i] = ABSTAIN
        else:
            labels[i] = int(vote_vals[order[0]])
    return labels


def lf_external_model_non_pred_max(
    external_model,
    data,
    pred_classes,
):
    """
    LF3: 外部异构模型非预测类最大 softmax 概率类别（无阈值版）。
    """
    external_outputs = external_model.predict(data, batch_size=64, verbose=0)
    external_outputs = np.asarray(external_outputs, dtype=np.float32)
    probs = tf.nn.softmax(external_outputs, axis=1).numpy()
    pred = _as_1d_int(pred_classes, 'pred_classes')
    if probs.shape[0] != len(pred):
        raise ValueError('LF3: batch size mismatch between model outputs and pred_classes')

    n, c = probs.shape
    labels = np.full(n, ABSTAIN, dtype=np.int64)
    for i in range(n):
        p = int(pred[i])
        if p < 0 or p >= c:
            continue
        row = probs[i].copy()
        row[p] = -np.inf
        labels[i] = int(np.argmax(row))
    return labels

    
def lf_augmented_non_pred_mode(
    aug_non_pred_mode_class,
):

    mode_cls = _as_1d_int(aug_non_pred_mode_class, 'aug_non_pred_mode_class')
    labels = np.full(len(mode_cls), ABSTAIN, dtype=np.int64)
    valid = mode_cls >= 0
    labels[valid] = mode_cls[valid]
    return labels


def _lf_metrics(pred_labels, true_labels):
    pred = _as_1d_int(pred_labels, 'pred_labels')
    truth = _as_1d_int(true_labels, 'true_labels')
    if len(pred) != len(truth):
        raise ValueError(f'Length mismatch: pred={len(pred)} vs truth={len(truth)}')

    abstain_mask = pred == ABSTAIN
    abstain_rate = float(np.mean(abstain_mask))
    non_abstain = ~abstain_mask
    if np.any(non_abstain):
        acc = float(np.mean(pred[non_abstain] == truth[non_abstain]))
    else:
        acc = float('nan')
    return abstain_rate, acc


def plot_lf_metrics(adv_metrics, data_name):
    n = len(adv_metrics)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0), squeeze=False)
    lf_names = ['LF1', 'LF2', 'LF3', 'LF4']
    x = np.arange(len(lf_names))
    width = 0.38

    for i, (adv_name, stats) in enumerate(adv_metrics):
        ax = axes[i // cols][i % cols]
        abstain_vals = [stats[name][0] for name in lf_names]
        acc_vals = [stats[name][1] for name in lf_names]
        ax.bar(x - width / 2, abstain_vals, width=width, label='Abstain rate')
        ax.bar(x + width / 2, acc_vals, width=width, label='Accuracy')
        ax.set_title(adv_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(lf_names)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        for j, val in enumerate(acc_vals):
            if np.isnan(val):
                ax.text(x[j] + width / 2, 0.02, 'N/A', rotation=90, ha='center', va='bottom', fontsize=8)

    total_axes = rows * cols
    for j in range(n, total_axes):
        axes[j // cols][j % cols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.suptitle(f'{data_name} adversarial sets: LF abstain rate and accuracy', y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.45)
    save_dir = Path(f'../data/{data_name}/plots')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'lf_metrics_subplots.png'
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved figure to: {save_path.resolve()}')


def snorkel_analysis(labels_list, cardinality, topk=2):
    L = np.column_stack(labels_list).astype(np.int64)
    label_model = LabelModel(cardinality=cardinality, verbose=True)

    label_model.fit(
        L_train=L,
        n_epochs=500,
        lr=0.01,
        log_freq=50,
        seed=123
    )
    probs = label_model.predict_proba(L)
    topk_idx = np.argsort(probs, axis=1)[:, -topk:][:, ::-1]
    topk_prob = np.take_along_axis(probs, topk_idx, axis=1)

    return {'probs': probs, 'topk_idx': topk_idx, 'topk_prob': topk_prob}


def compute_lf_labels_from_risk_features(
    data,
    risk_features,
    assist_model,
    consistency_layer_indices,
    *,
    gap_threshold=0.05,
):
    """
    由 risk_features、assist_model 与 data 计算 LF1–LF4，与 __main__ 中对抗集流程一致。
    返回四个长度均为 len(data) 的整型标签向量。
    """
    rf = risk_features
    lf1_labels = lf_non_pred_softmax_gap(
        non_pred_top1_class=rf['non_pred_top1_class'],
        non_pred_top1_prob=rf['non_pred_top1_prob'],
        non_pred_top2_prob=rf['non_pred_top2_prob'],
        gap_threshold=gap_threshold,
    )
    lf2_labels = lf_multi_layer_nearest_non_pred_consensus(
        feature_map=rf,
        layer_indices=consistency_layer_indices,
    )
    lf3_labels = lf_external_model_non_pred_max(
        external_model=assist_model,
        data=data,
        pred_classes=rf['pred_classes'],
    )
    lf4_labels = lf_augmented_non_pred_mode(
        aug_non_pred_mode_class=rf['aug_non_pred_mode_class'],
    )
    n = len(data)
    for name, arr in (
        ('LF1', lf1_labels),
        ('LF2', lf2_labels),
        ('LF3', lf3_labels),
        ('LF4', lf4_labels),
    ):
        if len(arr) != n:
            raise ValueError(f'{name} length {len(arr)} != len(data)={n}')
    return lf1_labels, lf2_labels, lf3_labels, lf4_labels


def build_snorkel_result_from_risk_features(
    data,
    risk_features,
    assist_model,
    consistency_layer_indices,
    cardinality,
    *,
    gap_threshold=0.05,
    topk=2,
):

    lf1_labels, lf2_labels, lf3_labels, lf4_labels = compute_lf_labels_from_risk_features(
        data,
        risk_features,
        assist_model,
        consistency_layer_indices,
        gap_threshold=gap_threshold,
    )
    snorkel_out = snorkel_analysis(
        [lf1_labels, lf2_labels, lf3_labels, lf4_labels],
        cardinality,
        topk=topk,
    )
    return {
        **snorkel_out,
        'lf1_labels': lf1_labels,
        'lf2_labels': lf2_labels,
        'lf3_labels': lf3_labels,
        'lf4_labels': lf4_labels,
    }


_SNORKEL_RESULT_KEYS = ('probs', 'topk_idx', 'topk_prob')
_LF_LABEL_KEYS = ('lf1_labels', 'lf2_labels', 'lf3_labels', 'lf4_labels')


def _assert_snorkel_cache_matches(z, n_pool, cardinality, gap_threshold, topk, consistency_layer_indices):
    if int(z['n_pool']) != int(n_pool):
        raise ValueError(
            f'snorkel cache n_pool={int(z["n_pool"])} != current pool size {n_pool}',
        )
    if int(z['cardinality']) != int(cardinality):
        raise ValueError('snorkel cache cardinality mismatch')
    if not np.isclose(float(z['gap_threshold']), float(gap_threshold), rtol=0.0, atol=1e-6):
        raise ValueError('snorkel cache gap_threshold mismatch')
    if int(z['topk']) != int(topk):
        raise ValueError('snorkel cache topk mismatch')
    cached_layers = np.asarray(z['consistency_layer_indices'], dtype=np.int64).reshape(-1)
    expected_layers = np.asarray(consistency_layer_indices, dtype=np.int64).reshape(-1)
    if not np.array_equal(cached_layers, expected_layers):
        raise ValueError('snorkel cache consistency_layer_indices mismatch')


def build_or_load_snorkel_result_from_risk_features(
    cache_dir,
    cache_name,
    data,
    risk_feature_map,
    lf_assist_model,
    lf_layer_indices,
    cardinality,
    *,
    gap_threshold=0.05,
    topk=2,
    force_recompute=False,
):
    n_pool = len(data)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path)
        meta_keys = {'n_pool', 'cardinality', 'gap_threshold', 'topk', 'consistency_layer_indices'}
        if meta_keys.issubset(z.files):
            try:
                _assert_snorkel_cache_matches(
                    z, n_pool, cardinality, gap_threshold, topk, lf_layer_indices,
                )
                out = {k: np.asarray(z[k]) for k in _SNORKEL_RESULT_KEYS}
                if all(k in z.files for k in _LF_LABEL_KEYS):
                    for k in _LF_LABEL_KEYS:
                        out[k] = np.asarray(z[k])
                print(f'Loaded snorkel result from {cache_path}')
                return out
            except ValueError as exc:
                print(f'Snorkel cache stale ({exc}), recomputing...')
        else:
            print(f'Snorkel cache missing metadata in {cache_path}, recomputing...')

    out = build_snorkel_result_from_risk_features(
        data,
        risk_feature_map,
        lf_assist_model,
        lf_layer_indices,
        cardinality,
        gap_threshold=gap_threshold,
        topk=topk,
    )
    np.savez_compressed(
        cache_path,
        probs=out['probs'],
        topk_idx=out['topk_idx'],
        topk_prob=out['topk_prob'],
        lf1_labels=out['lf1_labels'],
        lf2_labels=out['lf2_labels'],
        lf3_labels=out['lf3_labels'],
        lf4_labels=out['lf4_labels'],
        n_pool=np.int32(n_pool),
        cardinality=np.int32(cardinality),
        gap_threshold=np.float32(gap_threshold),
        topk=np.int32(topk),
        consistency_layer_indices=np.asarray(lf_layer_indices, dtype=np.int64),
    )
    print(f'Saved snorkel result to {cache_path}')
    return out


def snorkel_distribution_metrics(snorkel_result, true_labels, topk):
    """
    topk 准确率: 真实标签落在 Snorkel 预测分布的 top-k 类中。
    mean_true_class_prob: 真实标签在完整预测分布上的平均概率。
    """
    probs = np.asarray(snorkel_result['probs'], dtype=np.float64)
    topk_idx = np.asarray(snorkel_result['topk_idx'], dtype=np.int64)
    y = _as_1d_int(true_labels, 'true_labels')
    n = len(y)
    if probs.shape[0] != n or topk_idx.shape[0] != n:
        raise ValueError('Snorkel outputs length must match true_labels')
    if topk_idx.shape[1] < topk:
        raise ValueError('topk_idx has fewer columns than requested topk')

    in_topk = np.any(topk_idx[:, :topk] == y[:, np.newaxis], axis=1)
    topk_acc = float(np.mean(in_topk))
    row = np.arange(n, dtype=np.int64)
    true_probs = probs[row, y].astype(np.float64)
    mean_true_class_prob = float(np.mean(true_probs))
    return topk_acc, mean_true_class_prob


def plot_snorkel_distribution_metrics(adv_snorkel_stats, data_name, topk):
    """
    每个对抗集一个子图: Top-k 准确率 与 真实类在 Snorkel 分布上的平均概率。
    adv_snorkel_stats: list of (adv_name, (topk_acc, mean_true_class_prob))
    """
    n = len(adv_snorkel_stats)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0), squeeze=False)
    metric_names = [f'Top-{topk} acc', 'Mean P(true)']
    x = np.arange(len(metric_names))
    width = 0.55

    for i, (adv_name, (tka, mtp)) in enumerate(adv_snorkel_stats):
        ax = axes[i // cols, i % cols]
        ax.bar(x, [tka, mtp], width=width, color=['#4c72b0', '#dd8452'])
        ax.set_title(adv_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    total_axes = rows * cols
    for j in range(n, total_axes):
        axes[j // cols][j % cols].axis('off')

    fig.suptitle(
        f'{data_name} adversarial sets: Snorkel top-{topk} hit rate and mean true-class probability',
        y=0.99,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.45)
    save_dir = Path(f'../data/{data_name}/plots')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'snorkel_top{topk}_metrics_subplots.png'
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved figure to: {save_path.resolve()}')


if __name__ == "__main__":

    # data_name = 'fmnist'
    data_name = 'cifar10'

    if data_name == 'fmnist':
        model_path = '../models/lenet_fmnist/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        assist_model_path = '../models/resnet18_fmnist/tf_model.h5'
        assist_model = tf.keras.models.load_model(assist_model_path)
        x_train, y_train, x_test, y_test = load_fmnist()
        adv_dir = Path('../data/fmnist/adversarial')
        distance_layer_index = -4
        consistency_layer_indices = [-10, -8, -6, -4]
        risk_feature_dir = Path('../data/fmnist/')
    elif data_name == 'cifar10':
        model_path = '../models/vgg19_cifar10/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        assist_model_path = '../models/resnet18_cifar10/tf_model.h5'
        assist_model = tf.keras.models.load_model(assist_model_path)
        x_train, y_train, x_test, y_test = load_cifar10()
        adv_dir = Path('../data/cifar10/adversarial')
        distance_layer_index = -5
        consistency_layer_indices = [-19, -15, -11, -5]
        risk_feature_dir = Path('../data/cifar10/')
    else:
        exit()

    n_labels = int(cnn_model.output_shape[-1])
    if np.asarray(y_train).ndim > 1:
        y_train = np.argmax(y_train, axis=-1)
    if np.asarray(y_test).ndim > 1:
        y_test = np.argmax(y_test, axis=-1)
    y_test_pred = np.argmax(cnn_model.predict(x_test, batch_size=64, verbose=0), axis=-1)
    correct_mask = (y_test_pred == y_test)
    correct_x_test = x_test[correct_mask]
    wrong_x_test = x_test[~correct_mask]

    adv_files = sorted(adv_dir.glob('*_adv_data.npy'))
    if not adv_files:
        print(f'No files matching *_adv_data.npy under {adv_dir.resolve()}')

    snorkel_topk = 2
    adv_metrics = []
    adv_snorkel_results = []
    for adv_path in adv_files:
        adv_prefix = adv_path.name.removesuffix('_adv_data.npy')
        adv_data = np.load(adv_path)
        rf_path = risk_feature_dir / f'risk_features_{adv_prefix}.npz'
        clean_label_path = adv_dir / f'{adv_prefix}_clean_labels.npy'
        if not rf_path.is_file():
            print(f'Skip {adv_prefix}: missing {rf_path}')
            continue
        if not clean_label_path.is_file():
            print(f'Skip {adv_prefix}: missing {clean_label_path}')
            continue

        z = np.load(rf_path)
        risk_features = {k: np.asarray(z[k]) for k in z.files}
        clean_labels = _as_1d_int(np.load(clean_label_path), 'clean_labels')
        if len(clean_labels) != len(adv_data):
            print(f'Skip {adv_prefix}: clean_labels length mismatch with adv_data')
            continue

        try:
            snorkel_result = build_or_load_snorkel_result_from_risk_features(
                cache_dir=risk_feature_dir,
                cache_name=f'snorkel_result_{adv_prefix}.npz',
                data=adv_data,
                risk_feature_map=risk_features,
                lf_assist_model=assist_model,
                lf_layer_indices=consistency_layer_indices,
                cardinality=n_labels,
                gap_threshold=0.05,
                topk=snorkel_topk,
            )
        except ValueError as e:
            print(f'Skip {adv_prefix}: {e}')
            continue

        if all(k in snorkel_result for k in _LF_LABEL_KEYS):
            lf_stats = {
                'LF1': _lf_metrics(snorkel_result['lf1_labels'], clean_labels),
                'LF2': _lf_metrics(snorkel_result['lf2_labels'], clean_labels),
                'LF3': _lf_metrics(snorkel_result['lf3_labels'], clean_labels),
                'LF4': _lf_metrics(snorkel_result['lf4_labels'], clean_labels),
            }
            adv_metrics.append((adv_prefix, lf_stats))
            print(
                f'{adv_prefix}: '
                f'LF1(abstain={lf_stats["LF1"][0]:.3f}, acc={lf_stats["LF1"][1]:.3f}), '
                f'LF2(abstain={lf_stats["LF2"][0]:.3f}, acc={lf_stats["LF2"][1]:.3f}), '
                f'LF3(abstain={lf_stats["LF3"][0]:.3f}, acc={lf_stats["LF3"][1]:.3f}), '
                f'LF4(abstain={lf_stats["LF4"][0]:.3f}, acc={lf_stats["LF4"][1]:.3f})',
            )

        t_topk_acc, mean_p_true = snorkel_distribution_metrics(
            snorkel_result,
            clean_labels,
            topk=snorkel_topk,
        )
        adv_snorkel_results.append((adv_prefix, (t_topk_acc, mean_p_true)))
        print(
            f'Snorkel: top-{snorkel_topk} acc={t_topk_acc:.4f}, '
            f'mean P(true class)={mean_p_true:.4f}',
        )

    if adv_metrics:
        plot_lf_metrics(adv_metrics, data_name)

    if adv_snorkel_results:
        plot_snorkel_distribution_metrics(adv_snorkel_results, data_name, snorkel_topk)

