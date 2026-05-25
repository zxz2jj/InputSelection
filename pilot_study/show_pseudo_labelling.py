"""
对 pseudo_labelling（Snorkel）结果做可视化：按原始类别分组，选取满足条件的对抗样本，
用神经网络指定隐藏层的激活曲线展示。

数据路径（相对仓库根目录 `data/{dataset_name}/`）：
  - 对抗样本：`adversarial/{adv_name}_adv_data.npy`
  - 对抗目标类：`adversarial/{adv_name}_adv_targets.npy`
  - 原始类别：`adversarial/{adv_name}_clean_labels.npy`
  - Snorkel：`snorkel_result_{adv_name}.npz` 中的 `topk_idx`、`topk_prob`

选取规则（每个对抗目标类尽量选 n 个）：
  1. 对抗目标类相同（都被错分到同一类），原始类别两两不同。
  2. 指定隐藏层上的表示两两欧氏距离不超过 `max_pairwise_l2`。
  3. 伪标注 top-1 等于各自原始类别（`topk_idx[:, 0] == clean`），便于展示在相同 target、
     相近隐藏状态下，pseudo labelling 仍为不同真实类给出不同伪标签。
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training_models.global_config import num_of_labels


def _flatten_hidden_batch(h):
    h = np.asarray(h, dtype=np.float32)
    if h.ndim == 1:
        return h.reshape(1, -1)
    return h.reshape(h.shape[0], -1)


def get_hidden_outputs(model, layer_index, images, batch_size=64):
    """
    用子模型 forward 取中间层输出。`keras.backend.function` 在部分环境下会把 batch
    维丢掉，只得到 (D,) 而不是 (B, D)；`Model(...).predict` 一般能保持 (B, …)。
    """
    layer = model.layers[layer_index]
    sub = tf.keras.Model(
        inputs=model.layers[0].input,
        outputs=layer.output,
    )
    outs = []
    for start in range(0, len(images), batch_size):
        batch = images[start: start + batch_size]
        raw = sub.predict(batch, verbose=0)
        outs.append(_flatten_hidden_batch(raw))
    return np.concatenate(outs, axis=0)


def load_merged_adv_snorkel(dataset_name, adv_names, data_root=None):

    root = data_root or (_REPO_ROOT / 'data' / dataset_name)
    adv_dir = root / 'adversarial'

    chunks_x, chunks_t, chunks_c, chunks_ti, chunks_tp = [], [], [], [], []
    for name in adv_names:
        p_x = adv_dir / f'{name}_adv_data.npy'
        p_t = adv_dir / f'{name}_adv_targets.npy'
        p_c = adv_dir / f'{name}_clean_labels.npy'
        p_z = root / f'snorkel_result_{name}.npz'
        for p in (p_x, p_t, p_c, p_z):
            if not p.is_file():
                raise FileNotFoundError(p)

        x = np.load(p_x)
        t = np.asarray(np.load(p_t), dtype=np.int64).reshape(-1)
        c = np.asarray(np.load(p_c), dtype=np.int64).reshape(-1)
        z = np.load(p_z)
        ti = np.asarray(z['topk_idx'], dtype=np.int64)
        tp = np.asarray(z['topk_prob'], dtype=np.float32)
        n = x.shape[0]
        if not (len(t) == len(c) == n and ti.shape[0] == n and tp.shape[0] == n):
            raise ValueError(f'Length mismatch for adversarial set {name}: x={n}, t,c,topk={len(t)},{ti.shape}')

        chunks_x.append(x)
        chunks_t.append(t)
        chunks_c.append(c)
        chunks_ti.append(ti)
        chunks_tp.append(tp)

    return {
        'adv_data': np.concatenate(chunks_x, axis=0),
        'adv_targets': np.concatenate(chunks_t),
        'clean_labels': np.concatenate(chunks_c),
        'topk_idx': np.concatenate(chunks_ti, axis=0),
        'topk_prob': np.concatenate(chunks_tp, axis=0),
    }


def _max_pairwise_l2(rows):
    if len(rows) <= 1:
        return 0.0
    d = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=-1)
    iu = np.triu_indices(len(rows), k=1)
    return float(np.max(d[iu]))


def select_indices_per_adv_target(
    clean_labels,
    adv_targets,
    topk_idx,
    hidden,
    target_class,
    n,
    max_pairwise_l2,
    rng,
    max_trials=400,
):

    pseudo_top1 = topk_idx[:, 0]
    pool = np.where(
        (adv_targets == target_class)
        & (clean_labels != target_class)
        & (pseudo_top1 == clean_labels)
    )[0]
    if len(pool) < n:
        return None

    for _ in range(max_trials):
        i0 = int(rng.choice(pool))
        selected = [i0]
        used_clean = {int(clean_labels[i0])}

        while len(selected) < n:
            best_j = None
            best_score = np.inf
            for j in pool:
                if j in selected:
                    continue
                cj = int(clean_labels[j])
                if cj in used_clean:
                    continue
                h_j = hidden[j]
                dists = [float(np.linalg.norm(hidden[k] - h_j)) for k in selected]
                worst = max(dists)
                if worst < best_score:
                    best_score = worst
                    best_j = j

            if best_j is None or best_score > max_pairwise_l2:
                break

            selected.append(best_j)
            used_clean.add(int(clean_labels[best_j]))

        if len(selected) == n and _max_pairwise_l2(hidden[np.array(selected)]) <= max_pairwise_l2:
            return selected

    return None


def plot_selected_hidden_curves(
    hidden_rows,
    clean_labels,
    adv_targets,
    topk_idx,
    topk_prob,
    indices,
):
    x = np.arange(1, hidden_rows.shape[1] + 1)
    plt.figure(figsize=(9, 4.5))
    for rank, idx in enumerate(indices):
        row = hidden_rows[idx]
        c = int(clean_labels[idx])
        t = int(adv_targets[idx])
        kcols = min(topk_idx.shape[1], topk_prob.shape[1])
        pl_str = ', '.join(
            f'{int(topk_idx[idx, k])}: {float(topk_prob[idx, k]):.3f}'
            for k in range(kcols)
        )
        plt.plot(x, row, alpha=0.85, label=f'#{rank + 1} ({c}→{t})  PL=[{pl_str}]')

    plt.xlabel('Index')
    plt.ylabel('Activation Values')
    plt.legend(loc='upper right', fontsize=8)
    # plt.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.show()


def run_showcase(
    dataset_name,
    model_path,
    adv_names,
    hidden_layer_index=-4,
    samples_per_target=5,
    max_pairwise_l2=None,
    data_root=None,
    random_seed=0,
    batch_size=64,
):
    n_classes = num_of_labels[dataset_name]
    pack = load_merged_adv_snorkel(dataset_name, adv_names, data_root=data_root)
    x = pack['adv_data']
    adv_targets = pack['adv_targets']
    clean_labels = pack['clean_labels']
    topk_idx = pack['topk_idx']
    topk_prob = pack['topk_prob']

    model = tf.keras.models.load_model(model_path)
    hidden = get_hidden_outputs(model, hidden_layer_index, x, batch_size=batch_size)

    if max_pairwise_l2 is None:
        scale = float(np.median(np.linalg.norm(hidden, axis=1)) + 1e-8)
        max_pairwise_l2 = 0.25 * scale

    rng = np.random.default_rng(random_seed)
    for t in range(n_classes):
        idxs = select_indices_per_adv_target(
            clean_labels,
            adv_targets,
            topk_idx,
            hidden,
            target_class=t,
            n=samples_per_target,
            max_pairwise_l2=max_pairwise_l2,
            rng=rng,
        )
        if idxs is None:
            print(
                f'Adv target {t}: could not find {samples_per_target} samples '
                f'(distinct clean labels, adv_target=={t}, pseudo top1 = clean, '
                f'pairwise L2 ≤ {max_pairwise_l2:.4f}). '
                'Try raising max_pairwise_l2 or lowering samples_per_target.',
            )
            continue

        mp = _max_pairwise_l2(hidden[np.array(idxs)])
        cleans = [int(clean_labels[i]) for i in idxs]
        print(
            f'Adv target {t}: selected indices {idxs}, clean labels {cleans}, '
            f'max pairwise L2 = {mp:.4f}',
        )
        plot_selected_hidden_curves(
            hidden,
            clean_labels,
            adv_targets,
            topk_idx,
            topk_prob,
            idxs,
        )


if __name__ == '__main__':
    run_showcase(
        dataset_name='fmnist',
        model_path=_REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
        adv_names=['pixelattack', 'jsma', 'ead', 'cw_l2', 'hopskipjumpattack_l2', 'wassersteinattack'],
        hidden_layer_index=-4,
        samples_per_target=5,
        max_pairwise_l2=None,
        random_seed=1,
    )
