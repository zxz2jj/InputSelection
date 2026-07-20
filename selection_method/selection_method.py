import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

_sel_dir = Path(__file__).resolve().parent
_REPO_ROOT = _sel_dir.parent
for p in (_REPO_ROOT, _sel_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from training_models.load_data import load_cifar10, load_fmnist
from pseudo_labelling import build_or_load_snorkel_result_from_risk_features
from risk_scoring import (
    _hidden_to_flat_batch,
    build_or_load_class_prototypes_dict,
    build_or_load_mahalanobis_stats,
    build_or_load_risk_features,
    risk_scoring_function,
)


def compute_flat_hidden_vectors_at_layer(
    input_arr,
    keras_model,
    layer_index,
    forward_bs=64,
):
    if input_arr is None or len(input_arr) == 0:
        raise ValueError('input_arr is None or Empty!')

    try:
        hidden_layer = keras_model.layers[int(layer_index)]
    except IndexError as e:
        raise ValueError(
            f'distance_layer_index={layer_index!r} out of range '
            f'(model has {len(keras_model.layers)} layers)',
        ) from e

    sub = tf.keras.Model(
        inputs=keras_model.input,
        outputs=hidden_layer.output,
        name='selection_distance_hidden_forward',
    )
    chunks = []
    bs = int(forward_bs)
    for start in range(0, len(input_arr), bs):
        end = min(start + bs, len(input_arr))
        slice_batch = input_arr[start:end]
        hid = sub(slice_batch, training=False)
        flat = _hidden_to_flat_batch(hid, hidden_layer).numpy().astype(np.float32, copy=False)
        chunks.append(flat)
    return np.concatenate(chunks, axis=0)


def build_selection_storage_and_attributes(
    pool,
    *,
    num_correct,
    correct_labels,
    group_labels,
    predictions,
    risk_out,
    topk_idx,
    topk_prob,
    hidden_vectors,
):
    """
    构造传给选择算法的两份列表，相同样本使用相同 idx。

    sample_storage:
      idx, data, clean_label,
      is_wrongly_predicted: 所用预测类别 pred 是否与 clean_label 不一致，
      prediction: 该样本的预测类别（与 is_wrongly_predicted 所用 pred 一致），
        全部取自 pred_classes（与 risk_features 一致）。
    sample_attributes: idx, risk_score, prediction, top2_soft_pseudo_labelling, hidden_vector
        prediction 与 sample_storage 一致，取自 pred_classes。
        top2_soft_pseudo_labelling 为 Snorkel 的 ((类, 概率), (类, 概率))，与 pseudo_labelling.snorkel_analysis 一致。
    """
    n = int(len(pool))
    n_pool_tail = n - int(num_correct)
    if int(num_correct) < 0 or n_pool_tail < 0:
        raise ValueError('invalid num_correct for pool length')
    if len(correct_labels) != int(num_correct):
        raise ValueError('correct_labels length must equal num_correct')
    if len(group_labels) != n_pool_tail:
        raise ValueError('group_labels length must equal len(pool) - num_correct')
    pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if pred.shape[0] != n:
        raise ValueError('predictions length must match pool')

    rs = np.asarray(risk_out['risk_score'], dtype=np.float64).reshape(-1)
    si = np.asarray(risk_out['sample_index'], dtype=np.int64).reshape(-1)
    if rs.shape[0] != n or si.shape[0] != n:
        raise ValueError('risk_out length mismatch with pool')
    if not np.array_equal(si, np.arange(n, dtype=np.int64)):
        raise ValueError('risk_out sample_index must be 0..n-1 in order')

    tk = np.asarray(topk_idx, dtype=np.int64).reshape(n, -1)
    tp = np.asarray(topk_prob, dtype=np.float32).reshape(n, -1)
    if tk.shape[1] < 2 or tp.shape[1] < 2:
        raise ValueError('topk_idx/prob must have at least 2 columns for top-2 pseudo labels')
    hv = np.asarray(hidden_vectors, dtype=np.float32)
    if hv.shape[0] != n:
        raise ValueError('hidden_vectors row count mismatch')

    storage_out = []
    attributes_out = []
    n_correct_bound = int(num_correct)

    for i in range(n):
        if i < n_correct_bound:
            clean_l = int(correct_labels[i])
            p = int(pred[i])
        else:
            j_group = i - n_correct_bound
            clean_l = int(group_labels[j_group])
            p = int(pred[i])
        is_wrong = p != clean_l

        pseudo = (
            (int(tk[i, 0]), float(tp[i, 0])),
            (int(tk[i, 1]), float(tp[i, 1])),
        )

        storage_out.append({
            'idx': i,
            'data': pool[i],
            'clean_label': clean_l,
            'is_wrongly_predicted': bool(is_wrong),
            'prediction': p,
        })
        attributes_out.append({
            'idx': i,
            'risk_score': float(rs[i]),
            'prediction': p,
            'top2_soft_pseudo_labelling': pseudo,
            'hidden_vector': hv[i],
        })

    return storage_out, attributes_out


def _default_greedy_config():
    return {
        'alpha': 0.5,
        'phi_mode': 'sqrt',
        'phi_tau': None,
        'distance_metric': 'cosine',
        'eps': 1e-12,
        'filter_self_type': True,
        'risk_gate_power': 2.0,
    }


def build_fault_type_contributions(attribute_records, filter_self_type=True):
    """
    g_e(x) = p(x)：伪标注错误类型 e 的概率（Snorkel top-k 概率），不再乘以 risk_score。
    """
    sample_to_fault_types = {}
    fault_type_to_samples = defaultdict(list)

    for rec in attribute_records:
        sid = int(rec['idx'])
        pred = int(rec['prediction'])
        top2 = rec['top2_soft_pseudo_labelling']
        candidates = [
            ((int(top2[0][0]), pred), float(top2[0][1])),
            ((int(top2[1][0]), pred), float(top2[1][1])),
        ]
        pairs = []
        for fault_type, g in candidates:
            if filter_self_type and fault_type[0] == fault_type[1]:
                continue
            pairs.append((fault_type, float(g)))
        sample_to_fault_types[sid] = pairs
        for fault_type, _ in pairs:
            fault_type_to_samples[fault_type].append(sid)

    return sample_to_fault_types, dict(fault_type_to_samples)


def normalize_hidden_vectors(attribute_records, hidden_key='hidden_vector'):
    hidden_by_sample = {}
    for rec in attribute_records:
        sid = int(rec['idx'])
        vec = np.asarray(rec[hidden_key], dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec = np.zeros_like(vec)
        hidden_by_sample[sid] = vec.astype(np.float32, copy=False)
    return hidden_by_sample


def phi_fn(z, mode='sqrt', tau=None):
    scalar = np.isscalar(z)
    arr = np.asarray(z, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    if mode == 'sqrt':
        out = np.sqrt(arr)
    elif mode == 'log':
        out = np.log1p(arr)
    elif mode == 'cap':
        if tau is None:
            raise ValueError('phi cap mode requires tau')
        out = np.minimum(arr, float(tau))
    else:
        raise ValueError(f'unknown phi mode: {mode!r}')
    if scalar:
        return float(out)
    return out


def _hidden_distance(u, v, metric):
    if metric == 'cosine':
        return float(1.0 - np.dot(u, v))
    if metric == 'euclidean':
        return float(np.linalg.norm(u - v))
    raise ValueError(f'unknown distance_metric: {metric!r}')


def compute_coverage_gain(sample_id, greedy_state, greedy_cfg):
    total = 0.0
    for fault_type, g in greedy_state['sample_to_fault_types'].get(sample_id, []):
        if g <= 0.0:
            continue
        c_e = greedy_state['coverage_by_fault_type'].get(fault_type, 0.0)
        total += phi_fn(c_e + g, greedy_cfg['phi_mode'], greedy_cfg['phi_tau']) - phi_fn(
            c_e, greedy_cfg['phi_mode'], greedy_cfg['phi_tau'],
        )
    return total


def compute_type_conditional_novelty(sample_id, greedy_state, greedy_cfg):
    min_by_ft = greedy_state['min_dist_by_fault_type']
    total = 0.0
    for fault_type, g in greedy_state['sample_to_fault_types'].get(sample_id, []):
        if g <= 0.0:
            continue
        d_map = min_by_ft.get(fault_type)
        if not d_map:
            d_min = 1.0
        else:
            d_min = d_map.get(sample_id, 1.0)
        total += g * d_min
    return total


def _batch_dist_to_pick(hidden_matrix, rem_rows, pick_row, distance_metric):
    h_pick = hidden_matrix[pick_row]
    h_rem = hidden_matrix[rem_rows]
    if distance_metric == 'cosine':
        return (1.0 - h_rem @ h_pick).astype(np.float64, copy=False)
    if distance_metric == 'euclidean':
        return np.linalg.norm(h_rem - h_pick, axis=1).astype(np.float64, copy=False)
    raise ValueError(f'unknown distance_metric: {distance_metric!r}')


def _update_min_dist_after_pick(greedy_state, pick_id, greedy_cfg):
    remaining = greedy_state['remaining_ids']
    if not remaining:
        return

    id_to_row = greedy_state['sample_id_to_row']
    rem_rows = np.asarray([id_to_row[sid] for sid in remaining], dtype=np.int64)
    dist_vec = _batch_dist_to_pick(
        greedy_state['hidden_matrix'],
        rem_rows,
        id_to_row[pick_id],
        greedy_cfg['distance_metric'],
    )
    rem_dist = {remaining[i]: float(dist_vec[i]) for i in range(len(remaining))}
    remaining_set = set(remaining)
    min_by_ft = greedy_state['min_dist_by_fault_type']

    for fault_type, g in greedy_state['sample_to_fault_types'].get(pick_id, []):
        if g <= 0.0:
            continue
        bucket = min_by_ft[fault_type]
        for sid in greedy_state['fault_type_to_samples'][fault_type]:
            if sid not in remaining_set:
                continue
            d_new = rem_dist[sid]
            prev = bucket.get(sid)
            bucket[sid] = d_new if prev is None else min(prev, d_new)


def _replay_min_dist_updates(greedy_state, greedy_cfg):
    greedy_state['min_dist_by_fault_type'] = defaultdict(dict)
    remaining_set = set(greedy_state['sample_id_to_row'].keys())
    for pick_id in greedy_state['chosen_ids']:
        remaining_set.discard(pick_id)
        greedy_state['remaining_ids'] = list(remaining_set)
        _update_min_dist_after_pick(greedy_state, pick_id, greedy_cfg)


def _risk_gate_factor(risk_score, power):
    r = float(risk_score)
    if not np.isfinite(r):
        return 0.0
    return float(np.power(max(0.0, r), float(power)))


def normalize_candidate_scores(raw_scores, eps=1e-12):
    arr = np.asarray(raw_scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo + eps)


def _init_greedy_state(attribute_records, greedy_cfg):
    sample_to_fault_types, fault_type_to_samples = build_fault_type_contributions(
        attribute_records,
        filter_self_type=greedy_cfg['filter_self_type'],
    )
    hidden_by_sample = normalize_hidden_vectors(attribute_records)
    sample_ids = [int(rec['idx']) for rec in attribute_records]
    risk_by_sample = {int(rec['idx']): float(rec['risk_score']) for rec in attribute_records}
    hidden_matrix = np.stack(
        [hidden_by_sample[sid] for sid in sample_ids],
        axis=0,
    ).astype(np.float32, copy=False)
    return {
        'chosen_ids': [],
        'remaining_ids': list(sample_ids),
        'coverage_by_fault_type': {},
        'selected_by_fault_type': defaultdict(list),
        'sample_to_fault_types': sample_to_fault_types,
        'fault_type_to_samples': fault_type_to_samples,
        'hidden_by_sample': hidden_by_sample,
        'hidden_matrix': hidden_matrix,
        'sample_id_to_row': {sid: i for i, sid in enumerate(sample_ids)},
        'min_dist_by_fault_type': defaultdict(dict),
        'risk_by_sample': risk_by_sample,
        'score_log': [],
    }


def _update_greedy_state_after_selection(greedy_state, sample_id, greedy_cfg):
    greedy_state['chosen_ids'].append(sample_id)
    greedy_state['remaining_ids'].remove(sample_id)
    for fault_type, g in greedy_state['sample_to_fault_types'].get(sample_id, []):
        if g <= 0.0:
            continue
        greedy_state['coverage_by_fault_type'][fault_type] = (
            greedy_state['coverage_by_fault_type'].get(fault_type, 0.0) + g
        )
        bucket = greedy_state['selected_by_fault_type'][fault_type]
        if sample_id not in bucket:
            bucket.append(sample_id)
    _update_min_dist_after_pick(greedy_state, sample_id, greedy_cfg)


def _greedy_tie_break_key(sample_id, greedy_state, delta):
    risk = greedy_state['risk_by_sample'].get(sample_id, 0.0)
    return (delta, risk, -sample_id)


def _resolve_greedy_cfg(greedy_cfg=None):
    if greedy_cfg is None:
        return _default_greedy_config()
    merged = _default_greedy_config()
    merged.update(greedy_cfg)
    return merged


def _assert_greedy_run_cache_matches(z, greedy_cfg, selection_ratio, n_pool):
    merged = _resolve_greedy_cfg(greedy_cfg)
    if int(z['n_pool']) != int(n_pool):
        raise ValueError(
            f'greedy run cache n_pool={int(z["n_pool"])} != current pool size {n_pool}',
        )
    if not np.isclose(float(z['selection_ratio']), float(selection_ratio), rtol=0.0, atol=1e-6):
        raise ValueError('greedy run cache selection_ratio mismatch')
    if not np.isclose(float(z['alpha']), float(merged['alpha']), rtol=0.0, atol=1e-6):
        raise ValueError('greedy run cache alpha mismatch')
    if str(z['phi_mode'].item()) != str(merged['phi_mode']):
        raise ValueError('greedy run cache phi_mode mismatch')
    if bool(z['filter_self_type'].item()) != bool(merged['filter_self_type']):
        raise ValueError('greedy run cache filter_self_type mismatch')
    if str(z['distance_metric'].item()) != str(merged['distance_metric']):
        raise ValueError('greedy run cache distance_metric mismatch')
    if not np.isclose(
        float(z['risk_gate_power']),
        float(merged['risk_gate_power']),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError('greedy run cache risk_gate_power mismatch')


def _greedy_state_from_cached_run(greedy_order, attribute_records, greedy_cfg, cache_arrays):
    merged = _resolve_greedy_cfg(greedy_cfg)
    greedy_state = _init_greedy_state(attribute_records, merged)
    chosen = [int(i) for i in greedy_order]
    chosen_set = set(chosen)
    greedy_state['chosen_ids'] = chosen
    greedy_state['remaining_ids'] = [
        sid for sid in greedy_state['remaining_ids'] if sid not in chosen_set
    ]
    greedy_state['selection_ratio'] = float(cache_arrays['selection_ratio'])
    greedy_state['pick_count'] = int(cache_arrays['pick_count'])
    if 'score_log' in cache_arrays.files:
        greedy_state['score_log'] = list(cache_arrays['score_log'])
    _replay_min_dist_updates(greedy_state, merged)
    return greedy_state


def build_or_load_greedy_run(
    cache_dir,
    cache_name,
    attribute_records,
    selection_ratio=1.0,
    greedy_cfg=None,
    force_recompute=False,
):
    """
    缓存贪心完整运行结果；命中缓存时重建 greedy_state（含 sample_to_fault_types、score_log）。
    """
    merged_cfg = _resolve_greedy_cfg(greedy_cfg)
    if not 0.0 <= merged_cfg['alpha'] <= 1.0:
        raise ValueError('alpha must be in [0, 1]')

    n_pool = len(attribute_records)
    if n_pool == 0:
        raise ValueError('Candidate pool is empty.')
    sel_ratio = float(selection_ratio)
    if sel_ratio <= 0 or sel_ratio > 1:
        raise ValueError(f'selection_ratio must be in (0, 1], got {sel_ratio}')

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name

    if cache_path.is_file() and not force_recompute:
        z = np.load(cache_path, allow_pickle=True)
        _assert_greedy_run_cache_matches(z, merged_cfg, sel_ratio, n_pool)
        greedy_order = [int(i) for i in np.asarray(z['chosen_ids'], dtype=np.int64).reshape(-1)]
        greedy_state = _greedy_state_from_cached_run(
            greedy_order, attribute_records, merged_cfg, z,
        )
        print(f'Loaded greedy run from {cache_path} ({len(greedy_order)} selected)')
        return greedy_order, greedy_state

    greedy_order, greedy_state = greedy_select(
        attribute_records, selection_ratio=sel_ratio, greedy_cfg=merged_cfg,
    )
    score_log_arr = np.empty(len(greedy_state['score_log']), dtype=object)
    score_log_arr[:] = greedy_state['score_log']
    np.savez_compressed(
        cache_path,
        chosen_ids=np.asarray(greedy_order, dtype=np.int64),
        selection_ratio=np.float32(sel_ratio),
        pick_count=np.int32(greedy_state['pick_count']),
        n_pool=np.int32(n_pool),
        alpha=np.float32(merged_cfg['alpha']),
        phi_mode=np.array(merged_cfg['phi_mode']),
        filter_self_type=np.array(merged_cfg['filter_self_type']),
        distance_metric=np.array(merged_cfg['distance_metric']),
        risk_gate_power=np.float32(merged_cfg['risk_gate_power']),
        score_log=score_log_arr,
    )
    print(f'Saved greedy run to {cache_path}')
    return greedy_order, greedy_state


def greedy_select(attribute_records, selection_ratio, greedy_cfg=None):
    """
    全局贪心选择：覆盖/多样性基于伪标注类型概率 g_e(x)=p_e(x)；
    CovNorm、NovNorm 仅对候选池 min-max 归一化后加权求和，再经风险门控：
    Delta(x|S) = r(x)^risk_gate_power * (alpha*CovNorm + (1-alpha)*NovNorm)。
    selection_ratio: (0, 1] 的选择比例，实际个数为 ceil(ratio * len(attribute_records))，
        并限制在 [1, n]。
    返回 (chosen_ids, greedy_state)。
    """
    greedy_cfg = _resolve_greedy_cfg(greedy_cfg)
    if not 0.0 <= greedy_cfg['alpha'] <= 1.0:
        raise ValueError('alpha must be in [0, 1]')
    if float(greedy_cfg['risk_gate_power']) <= 0.0:
        raise ValueError('risk_gate_power must be > 0')

    n_pool = len(attribute_records)
    if n_pool == 0:
        raise ValueError('Candidate pool is empty.')
    sel_ratio = float(selection_ratio)
    if sel_ratio <= 0 or sel_ratio > 1:
        raise ValueError(f'Selection_ratio must be in (0, 1], got {sel_ratio}')
    n_pick = int(np.ceil(sel_ratio * n_pool))
    n_pick = max(1, min(n_pick, n_pool))

    greedy_state = _init_greedy_state(attribute_records, greedy_cfg)
    greedy_state['selection_ratio'] = sel_ratio
    greedy_state['pick_count'] = n_pick

    for round_idx in tqdm(range(n_pick), desc='greedy select', unit='sample'):
        if not greedy_state['remaining_ids']:
            break
        pool_candidates = list(greedy_state['remaining_ids'])
        cov_raw = np.array(
            [compute_coverage_gain(cid, greedy_state, greedy_cfg) for cid in pool_candidates],
            dtype=np.float64,
        )
        nov_raw = np.array(
            [
                compute_type_conditional_novelty(cid, greedy_state, greedy_cfg)
                for cid in pool_candidates
            ],
            dtype=np.float64,
        )
        cov_norm = normalize_candidate_scores(cov_raw, eps=greedy_cfg['eps'])
        nov_norm = normalize_candidate_scores(nov_raw, eps=greedy_cfg['eps'])
        combined_norm = (
            greedy_cfg['alpha'] * cov_norm + (1.0 - greedy_cfg['alpha']) * nov_norm
        )
        gate_power = float(greedy_cfg['risk_gate_power'])
        risk_gate = np.array(
            [
                _risk_gate_factor(greedy_state['risk_by_sample'].get(cid, 0.0), gate_power)
                for cid in pool_candidates
            ],
            dtype=np.float64,
        )
        score_delta = risk_gate * combined_norm

        best_idx = max(
            range(len(pool_candidates)),
            key=lambda i: _greedy_tie_break_key(
                pool_candidates[i], greedy_state, float(score_delta[i]),
            ),
        )
        pick_id = pool_candidates[best_idx]
        fault_info = greedy_state['sample_to_fault_types'].get(pick_id, [])
        greedy_state['score_log'].append({
            'round': round_idx,
            'sample_id': pick_id,
            'cov_gain_raw': float(cov_raw[best_idx]),
            'nov_raw': float(nov_raw[best_idx]),
            'cov_norm': float(cov_norm[best_idx]),
            'nov_norm': float(nov_norm[best_idx]),
            'combined_norm': float(combined_norm[best_idx]),
            'risk_gate': float(risk_gate[best_idx]),
            'risk_score': float(greedy_state['risk_by_sample'].get(pick_id, 0.0)),
            'delta': float(score_delta[best_idx]),
            'fault_types': [ft for ft, _ in fault_info],
            'g_values': [g for _, g in fault_info],
        })
        _update_greedy_state_after_selection(greedy_state, pick_id, greedy_cfg)

    return list(greedy_state['chosen_ids']), greedy_state


def fetch_selected_from_storage(storage_rows, chosen_ids):
    by_idx = {int(row['idx']): row for row in storage_rows}
    missing = [i for i in chosen_ids if int(i) not in by_idx]
    if missing:
        raise KeyError(f'storage_rows missing idx: {missing[:5]}')
    return [by_idx[int(i)] for i in chosen_ids]


def default_greedy_budget_ratios():
    return np.round(np.arange(0.05, 1.001, 0.05), 2).tolist()


def error_mask_from_storage(storage_rows):
    n = len(storage_rows)
    err = np.zeros(n, dtype=bool)
    for row in storage_rows:
        err[int(row['idx'])] = bool(row['is_wrongly_predicted'])
    return err


def real_error_type_universe_from_storage(storage_rows):
    """全池中真实误分类类型 (clean_label, prediction)。"""
    universe = set()
    for row in storage_rows:
        if row['is_wrongly_predicted']:
            universe.add((int(row['clean_label']), int(row['prediction'])))
    return universe


def real_error_types_covered_in_prefix(chosen_prefix, storage_rows, real_universe):
    """
    Count real fault types covered by a selected prefix. A type is counted only
    when the selected sample itself is truly mispredicted.
    """
    rows_by_id = {int(row['idx']): row for row in storage_rows}
    discovered = set()
    for sid in chosen_prefix:
        row = rows_by_id.get(int(sid))
        if row is None or not row['is_wrongly_predicted']:
            continue
        ft = (int(row['clean_label']), int(row['prediction']))
        if ft in real_universe:
            discovered.add(ft)
    return discovered


def compute_greedy_selection_curves(
    storage_rows,
    greedy_order,
    sample_to_fault_types,
    budget_ratios=None,
):
    """
    Evaluate prefixes of the greedy order. Error discovery uses the true error
    mask, and fault-type coverage counts only selected samples that are truly
    mispredicted, using their real (clean_label, prediction) type.
    """
    if budget_ratios is None:
        budget_ratios = default_greedy_budget_ratios()

    err = error_mask_from_storage(storage_rows)
    n_pool = len(storage_rows)
    total_errors = int(np.sum(err))
    real_universe = real_error_type_universe_from_storage(storage_rows)
    total_fault_types = len(real_universe)

    chosen = [int(i) for i in greedy_order]
    out_ratios = []
    out_budget_counts = []
    out_discovered_errors = []
    out_trc = []
    out_error_recall = []
    out_fault_type_count = []
    out_fault_type_ratio = []

    for r in budget_ratios:
        rr = float(r)
        if rr <= 0 or rr > 1:
            raise ValueError(f'budget ratio must be in (0, 1], got {rr}')
        k = int(np.ceil(rr * n_pool))
        k = max(1, min(k, n_pool))
        prefix = chosen[:k]
        discovered = int(np.sum(err[prefix]))
        denom = min(k, total_errors)
        trc = (discovered / denom) if denom > 0 else np.nan
        error_recall = (discovered / total_errors) if total_errors > 0 else np.nan
        discovered_types = real_error_types_covered_in_prefix(prefix, storage_rows, real_universe)
        ft_count = len(discovered_types)
        ft_ratio = (ft_count / total_fault_types) if total_fault_types > 0 else np.nan

        out_ratios.append(rr)
        out_budget_counts.append(k)
        out_discovered_errors.append(discovered)
        out_trc.append(trc)
        out_error_recall.append(error_recall)
        out_fault_type_count.append(ft_count)
        out_fault_type_ratio.append(ft_ratio)

    return {
        'budget_ratio': np.asarray(out_ratios, dtype=np.float32),
        'budget_count': np.asarray(out_budget_counts, dtype=np.int32),
        'discovered_errors': np.asarray(out_discovered_errors, dtype=np.int32),
        'total_errors': np.int32(total_errors),
        'trc': np.asarray(out_trc, dtype=np.float32),
        'error_recall': np.asarray(out_error_recall, dtype=np.float32),
        'fault_type_count': np.asarray(out_fault_type_count, dtype=np.int32),
        'fault_type_ratio': np.asarray(out_fault_type_ratio, dtype=np.float32),
        'total_fault_types': np.int32(total_fault_types),
        'chosen_ids': np.asarray(chosen, dtype=np.int64),
    }


def _greedy_curve_subplot_axes(named_results, ncols):
    n = len(named_results)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.9 * nrows))
    axes = np.asarray(axes).reshape(-1)
    x_min = 0.0
    x_max = 1.0 + 1e-3
    if n > 0:
        br = np.asarray(named_results[0][1]['budget_ratio'], dtype=np.float32).reshape(-1)
        br = br[np.isfinite(br)]
        if br.size > 0:
            x_min = min(0.0, float(np.min(br)))
            x_max = max(1.0, float(np.max(br))) + 1e-3
    return fig, axes, x_min, x_max


def plot_greedy_trc_subplots(named_results, save_dir, ncols=4):
    if len(named_results) == 0:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes, x_min, x_max = _greedy_curve_subplot_axes(named_results, ncols)

    for i, (name, curve) in enumerate(named_results):
        ax = axes[i]
        x = np.asarray(curve['budget_ratio'], dtype=np.float32)
        y_trc = np.asarray(curve['trc'], dtype=np.float32)
        y_error_recall = np.asarray(curve['error_recall'], dtype=np.float32)
        ax.plot(x, y_trc, marker='o', linewidth=1.8, label='TRC')
        ax.plot(x, y_error_recall, marker='s', linewidth=1.6, label='Error Recall')
        ax.set_title(name)
        ax.set_xlabel('Budget Ratio')
        ax.set_xticks(np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 2))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
        ax.legend(fontsize=8, loc='lower right')
        for idx, (xx, yy) in enumerate(zip(x, y_trc)):
            if idx % 2 == 0 and np.isfinite(yy):
                ax.annotate(
                    f'{yy:.2f}', (xx, yy), textcoords='offset points', xytext=(0, -10),
                    ha='center', fontsize=8,
                )

    for j in range(len(named_results), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Greedy selection: TRC and error recall', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = save_dir / 'greedy_trc_subplots.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved plot: {out_path}')


def plot_greedy_fault_type_subplots(named_results, save_dir, ncols=4):
    if len(named_results) == 0:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes, x_min, x_max = _greedy_curve_subplot_axes(named_results, ncols)

    for i, (name, curve) in enumerate(named_results):
        ax = axes[i]
        x = np.asarray(curve['budget_ratio'], dtype=np.float32)
        y_count = np.asarray(curve['fault_type_count'], dtype=np.int32)
        y_ratio = np.asarray(curve['fault_type_ratio'], dtype=np.float32)
        total_ft = int(curve['total_fault_types'])
        line_count, = ax.plot(
            x, y_count, marker='o', linewidth=1.8, color='C0', label='Fault type count',
        )
        ax.set_title(name)
        ax.set_xlabel('Budget Ratio')
        ax.set_ylabel('Fault type count', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_xticks(np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 2))
        ax.set_xlim(x_min, x_max)
        count_max = int(np.max(y_count)) if y_count.size > 0 else 0
        ax.set_ylim(0, max(count_max, total_ft, 1) + 0.5)
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)

        ax2 = ax.twinx()
        line_ratio, = ax2.plot(
            x, y_ratio, marker='s', linewidth=1.6, color='C1', label='Fault type ratio',
        )
        ax2.set_ylabel('Ratio to pool fault types', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        ax2.set_ylim(0.0, 1.05)
        ax.legend(
            handles=[line_count, line_ratio],
            labels=['Fault type count', 'Fault type ratio'],
            fontsize=8,
            loc='lower right',
        )

    for j in range(len(named_results), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Greedy selection: real fault type coverage', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = save_dir / 'greedy_fault_type_subplots.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    data_name = 'fmnist'
    # data_name = 'cifar10'

    if data_name == 'fmnist':
        model_path = '../models/lenet_fmnist/tf_model.h5'
        assist_model_path = '../models/resnet18_fmnist/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        assist_model = tf.keras.models.load_model(assist_model_path)
        x_train, y_train, x_test, y_test = load_fmnist()
        adv_dir = Path('../data/fmnist/adversarial')
        distance_layer_index = -4
        consistency_layer_indices = [-10, -8, -6, -4]
    elif data_name == 'cifar10':
        model_path = '../models/vgg19_cifar10/tf_model.h5'
        assist_model_path = '../models/resnet18_cifar10/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        assist_model = tf.keras.models.load_model(assist_model_path)
        x_train, y_train, x_test, y_test = load_cifar10()
        adv_dir = Path('../data/cifar10/adversarial')
        distance_layer_index = -5
        consistency_layer_indices = [-19, -15, -11, -5]
    else:
        raise SystemExit(f'unknown data_name={data_name!r}')
    if np.asarray(y_train).ndim > 1:
        y_train = np.argmax(y_train, axis=-1)
    if np.asarray(y_test).ndim > 1:
        y_test = np.argmax(y_test, axis=-1)

    n_labels = int(cnn_model.output_shape[-1])
    risk_features_cache_dir = Path(__file__).resolve().parent.parent / 'data' / data_name

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

    class_means, class_inv_covs = build_or_load_mahalanobis_stats(
        cnn_model,
        train_data=x_train,
        train_labels=y_train,
        layer_index=distance_layer_index,
        dataset_name=data_name,
        batch_size=64,
    )

    y_test_pred = np.argmax(cnn_model.predict(x_test, batch_size=64, verbose=0), axis=-1)
    correct_mask = (y_test_pred == y_test)
    correct_x_test = x_test[correct_mask]
    wrong_x_test = x_test[~correct_mask]
    print(f'{data_name} -> correct: {len(correct_x_test)}, wrong: {len(wrong_x_test)}')

    adv_files = sorted(adv_dir.glob('*_adv_data.npy'))
    # quick_eval_adv_prefixes = ('ba',  'cw_l2', 'brightness', 'rotation')
    # adv_files = []
    # for prefix in quick_eval_adv_prefixes:
    #     adv_path = adv_dir / f'{prefix}_adv_data.npy'
    #     if adv_path.is_file():
    #         adv_files.append(adv_path)
    #     else:
    #         print(f'Warning: missing adversarial data {adv_path.name}, skip')

    if len(correct_x_test) > 0:
        combined_eval_groups = []
        if len(wrong_x_test) > 0:
            combined_eval_groups.append(('wrong_x_test', wrong_x_test))
        for adv_path in adv_files:
            adv_data = np.load(adv_path)
            adv_prefix = adv_path.name.removesuffix('_adv_data.npy')
            combined_eval_groups.append((adv_prefix, adv_data))

        correct_clean_labels = y_test[correct_mask].astype(np.int64, copy=False)
        wrong_clean_labels = y_test[~correct_mask].astype(np.int64, copy=False)

        greedy_config = {'alpha': 0.5, 'phi_mode': 'sqrt', 'risk_gate_power': 3.0}
        greedy_curve_groups = []
        curve_save_dir = risk_features_cache_dir / 'plots'

        for group_name, group_data in combined_eval_groups:
            combined_data = np.concatenate([correct_x_test, group_data], axis=0)
            n_correct = len(correct_x_test)
            n_group = len(group_data)

            if group_name == 'wrong_x_test':
                group_clean_labels = wrong_clean_labels
            else:
                p_clean = adv_dir / f'{group_name}_clean_labels.npy'
                if not p_clean.is_file():
                    raise FileNotFoundError(f'Not found {p_clean.name}')
                group_clean_labels = np.asarray(np.load(p_clean), dtype=np.int64).reshape(-1)
                if group_clean_labels.shape[0] != n_group:
                    raise ValueError(
                        f'label file length mismatch for {group_name}: '
                        f'group_data={n_group}, clean labels={group_clean_labels.shape[0]}',
                    )

            combined_risk_features = build_or_load_risk_features(
                cache_dir=risk_features_cache_dir,
                cache_name=f'risk_features_combined_correct_{group_name}.npz',
                data=combined_data,
                model=cnn_model,
                prototypes_by_layer_map=prototypes_by_layer,
                distance_feature_layer_index=distance_layer_index,
                consistency_feature_layer_indices=consistency_layer_indices,
                batch_size=16,
                class_means=class_means,
                class_inv_covs=class_inv_covs,
            )
            risk_scores = risk_scoring_function(
                combined_risk_features,
                sample_indices=np.arange(len(combined_data), dtype=np.int64),
            )

            snorkel_result = build_or_load_snorkel_result_from_risk_features(
                cache_dir=risk_features_cache_dir,
                cache_name=f'snorkel_result_combined_correct_{group_name}.npz',
                data=combined_data,
                risk_feature_map=combined_risk_features,
                lf_assist_model=assist_model,
                lf_layer_indices=consistency_layer_indices,
                cardinality=n_labels,
                gap_threshold=0.05,
                topk=2,
            )
            snorkel_topk_idx = np.asarray(snorkel_result['topk_idx'], dtype=np.int64)
            snorkel_topk_prob = np.asarray(snorkel_result['topk_prob'], dtype=np.float32)
            n_combined = len(combined_data)
            if snorkel_topk_idx.shape[0] != n_combined or snorkel_topk_prob.shape[0] != n_combined:
                raise ValueError('Snorkel topk outputs length mismatch with combined_data')
            if snorkel_topk_idx.shape[1] < 2:
                raise ValueError('Snorkel topk_idx must have at least 2 columns for top-2 pseudo labels')

            hidden_flat = compute_flat_hidden_vectors_at_layer(
                combined_data,
                cnn_model,
                distance_layer_index,
            )
            sample_storage, sample_attributes = build_selection_storage_and_attributes(
                combined_data,
                num_correct=n_correct,
                correct_labels=correct_clean_labels,
                group_labels=group_clean_labels,
                predictions=combined_risk_features['pred_classes'],
                risk_out=risk_scores,
                topk_idx=snorkel_topk_idx,
                topk_prob=snorkel_topk_prob,
                hidden_vectors=hidden_flat,
            )
            assert len(sample_storage) == len(sample_attributes) == len(combined_data)
            assert sample_storage[0]['idx'] == sample_attributes[0]['idx'] == 0
            print(
                f'{group_name}: built {len(sample_storage)} storage rows and '
                f'{len(sample_attributes)} attribute rows (hidden_dim={hidden_flat.shape[1]})',
            )
            chosen_ids, greedy_run = build_or_load_greedy_run(
                cache_dir=risk_features_cache_dir,
                cache_name=f'greedy_run_combined_correct_{group_name}.npz',
                attribute_records=sample_attributes,
                selection_ratio=1.0,
                greedy_cfg=greedy_config,
            )
            curve_result = compute_greedy_selection_curves(
                sample_storage,
                chosen_ids,
                greedy_run['sample_to_fault_types'],
            )
            greedy_curve_groups.append((group_name, curve_result))
            print(
                f'{group_name}: greedy full run {len(chosen_ids)} / {len(sample_attributes)}; '
                f'total_errors={curve_result["total_errors"]}, '
                f'total_fault_types={curve_result["total_fault_types"]}',
            )
            print(
                f'  @100% budget: trc={curve_result["trc"][-1]:.3f}, '
                f'error_recall={curve_result["error_recall"][-1]:.3f}, '
                f'fault_types={curve_result["fault_type_count"][-1]}/'
                f'{curve_result["total_fault_types"]}',
            )

        if greedy_curve_groups:
            plot_greedy_trc_subplots(greedy_curve_groups, curve_save_dir)
            plot_greedy_fault_type_subplots(greedy_curve_groups, curve_save_dir)
