import json
import logging
import os
import sys
from datetime import datetime, timezone
from itertools import product
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

tf.get_logger().setLevel('ERROR')
logging.getLogger().setLevel(logging.WARNING)

from RQ1_fault_revealing import (
    DATASET_CONFIG,
    RQ1_BUDGET_RATIOS,
    _as_label_vector,
    _lookup_trc,
    build_flat_pool_storage_and_attributes,
    explore_feature_cache_name,
    load_sampled_pool,
    pool_cache_tag,
)
from pseudo_labelling import build_or_load_snorkel_result_from_risk_features
from risk_scoring import (
    DEFAULT_RISK_FEATURE_KEYS,
    build_or_load_class_prototypes_dict,
    build_or_load_mahalanobis_stats,
    build_or_load_risk_features,
    required_risk_feature_keys,
    risk_scoring_function,
)
from selection_method import (
    compute_flat_hidden_vectors_at_layer,
    compute_greedy_selection_curves,
    greedy_select,
)

# --- Fixed pool ---
DATASET_NAME = 'fmnist'
POOL_TYPE = 'adversarial'
SAMPLING_MODE = 'random'
ERROR_RATIO = 0.10
SEED = 0

# --- Search grid ---
RISK_GATE_POWERS = (3.0, 4.0, 5.0, 6.0)
ALPHAS = (0.5, 0.7, 0.9)
PHI_SETTINGS = (
    ('sqrt', None),
    ('log', None),
    ('cap', 1.0),
    ('cap', 2.0),
)

BUDGET_RATIOS = list(RQ1_BUDGET_RATIOS)
SAMPLED_ROOT = _EXP_DIR / 'sampled_data'
CACHE_DIR = _EXP_DIR / 'cache_files' / DATASET_NAME
OUT_JSON = CACHE_DIR / 'rq1_greedy_param_search.json'
OUT_MD = CACHE_DIR / 'rq1_greedy_param_search.md'


def _budget_key(b):
    return f'{float(b):.2f}'


def _trc_dict(curve, budgets=BUDGET_RATIOS):
    trc = np.asarray(curve['trc'], dtype=np.float64).reshape(-1)
    if len(trc) != len(budgets):
        raise ValueError('curve TRC length mismatch with budgets')
    return {_budget_key(b): float(trc[i]) for i, b in enumerate(budgets)}


def _fmt_trc(trc_by_budget, budgets=BUDGET_RATIOS):
    return ', '.join(
        f'{b:.0%}={trc_by_budget[_budget_key(b)]:.4f}' for b in budgets
    )


def build_pool_attributes(pool, cnn_model, assist_model, dataset_cfg, cache_dir):
    """Load risk features + Snorkel once; build storage/attributes for greedy."""
    pool_data = pool['data']
    n_pool = len(pool_data)
    tag = pool_cache_tag(DATASET_NAME, POOL_TYPE, ERROR_RATIO, SEED, SAMPLING_MODE)
    feature_cache_name = explore_feature_cache_name(
        DATASET_NAME, POOL_TYPE, ERROR_RATIO, SEED, SAMPLING_MODE,
    )
    required_keys = required_risk_feature_keys(dataset_cfg['consistency_layer_indices'])

    x_train, y_train, _, _ = dataset_cfg['loader']()
    y_train = _as_label_vector(y_train)
    all_prototype_layers = list(dict.fromkeys(
        [dataset_cfg['distance_layer_index']] + dataset_cfg['consistency_layer_indices'],
    ))
    prototypes_by_layer = build_or_load_class_prototypes_dict(
        cnn_model,
        train_data=x_train,
        train_labels=y_train,
        layer_indices=all_prototype_layers,
        dataset_name=DATASET_NAME,
        batch_size=64,
    )
    class_means, class_inv_covs = build_or_load_mahalanobis_stats(
        cnn_model,
        train_data=x_train,
        train_labels=y_train,
        layer_index=dataset_cfg['distance_layer_index'],
        dataset_name=DATASET_NAME,
        batch_size=64,
    )

    risk_features = build_or_load_risk_features(
        cache_dir=cache_dir,
        cache_name=feature_cache_name,
        data=pool_data,
        model=cnn_model,
        prototypes_by_layer_map=prototypes_by_layer,
        distance_feature_layer_index=dataset_cfg['distance_layer_index'],
        consistency_feature_layer_indices=dataset_cfg['consistency_layer_indices'],
        batch_size=16,
        force_recompute=False,
        class_means=class_means,
        class_inv_covs=class_inv_covs,
        required_feature_keys=required_keys,
        write_cache=False,
    )
    risk_scores = risk_scoring_function(
        risk_features,
        feature_keys=list(DEFAULT_RISK_FEATURE_KEYS),
        sample_indices=np.arange(n_pool, dtype=np.int64),
    )
    snorkel_result = build_or_load_snorkel_result_from_risk_features(
        cache_dir=cache_dir,
        cache_name=f'snorkel_result_{tag}.npz',
        data=pool_data,
        risk_feature_map=risk_features,
        lf_assist_model=assist_model,
        lf_layer_indices=dataset_cfg['consistency_layer_indices'],
        cardinality=int(cnn_model.output_shape[-1]),
        gap_threshold=0.05,
        topk=2,
        force_recompute=False,
    )
    hidden_flat = compute_flat_hidden_vectors_at_layer(
        pool_data,
        cnn_model,
        dataset_cfg['distance_layer_index'],
    )
    sample_storage, sample_attributes = build_flat_pool_storage_and_attributes(
        pool_data,
        pool['clean_labels'],
        pool['is_error'],
        risk_features['pred_classes'],
        risk_scores,
        snorkel_result['topk_idx'],
        snorkel_result['topk_prob'],
        hidden_flat,
    )
    return sample_storage, sample_attributes


def run_one_config(sample_storage, sample_attributes, greedy_cfg, max_budget_ratio):
    """Run greedy in-memory (no greedy_run_*.npz write) and return TRC."""
    greedy_order, _state = greedy_select(
        sample_attributes,
        selection_ratio=float(max_budget_ratio),
        greedy_cfg=greedy_cfg,
    )
    curve = compute_greedy_selection_curves(
        sample_storage,
        greedy_order,
        None,
        budget_ratios=BUDGET_RATIOS,
    )
    return _trc_dict(curve)


def format_summary_md(rows, budgets=BUDGET_RATIOS):
    headers = [
        'risk_gate_power', 'alpha', 'phi_mode', 'phi_tau',
        *[f'TRC@{b:.0%}' for b in budgets],
    ]
    lines = [
        '# RQ1 greedy hyperparameter search (FMNIST)',
        '',
        f'- Pool: `{DATASET_NAME}/{POOL_TYPE}/{SAMPLING_MODE}/'
        f'ratio={ERROR_RATIO:.2f}/seed={SEED}`',
        f'- Grid: p={list(RISK_GATE_POWERS)}, alpha={list(ALPHAS)}, '
        f'phi={list(PHI_SETTINGS)}',
        f'- Budgets: {budgets}',
        f'- Note: greedy_run caches are **not** written; snorkel/risk caches reused.',
        '',
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        tau = row['phi_tau']
        tau_s = '-' if tau is None else f'{float(tau):.1f}'
        cells = [
            f"{row['risk_gate_power']:.1f}",
            f"{row['alpha']:.1f}",
            str(row['phi_mode']),
            tau_s,
        ]
        for b in budgets:
            cells.append(f"{row['trc_by_budget'][_budget_key(b)]:.4f}")
        lines.append('| ' + ' | '.join(cells) + ' |')

    # Best by each budget + mean of four budgets
    lines.extend(['', '## Best by budget'])
    for b in budgets:
        key = _budget_key(b)
        best = max(rows, key=lambda r: r['trc_by_budget'][key])
        lines.append(
            f"- @{b:.0%}: p={best['risk_gate_power']:.1f}, "
            f"alpha={best['alpha']:.1f}, phi={best['phi_mode']}"
            + (f"(tau={best['phi_tau']})" if best['phi_tau'] is not None else '')
            + f" -> TRC={best['trc_by_budget'][key]:.4f}",
        )
    for row in rows:
        vals = [row['trc_by_budget'][_budget_key(b)] for b in budgets]
        row['_mean_trc'] = float(np.mean(vals))
    best_mean = max(rows, key=lambda r: r['_mean_trc'])
    lines.append(
        f"- mean(1/3/5/10%): p={best_mean['risk_gate_power']:.1f}, "
        f"alpha={best_mean['alpha']:.1f}, phi={best_mean['phi_mode']}"
        + (
            f"(tau={best_mean['phi_tau']})"
            if best_mean['phi_tau'] is not None else ''
        )
        + f" -> mean_TRC={best_mean['_mean_trc']:.4f}",
    )
    return '\n'.join(lines) + '\n'


def main():
    cfg = DATASET_CONFIG[DATASET_NAME]
    for path_key in ('model_path', 'assist_model_path'):
        if not Path(cfg[path_key]).is_file():
            raise FileNotFoundError(f'missing {path_key}: {cfg[path_key]}')

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pool = load_sampled_pool(
        SAMPLED_ROOT, DATASET_NAME, POOL_TYPE, ERROR_RATIO, SEED, SAMPLING_MODE,
    )
    print(
        f'Pool {DATASET_NAME}/{POOL_TYPE}/{SAMPLING_MODE}/'
        f'ratio={ERROR_RATIO:.2f}/seed={SEED} '
        f'(n={len(pool["data"])}, errors={int(np.sum(pool["is_error"]))})',
    )

    print('Loading models + risk/snorkel features (reuse caches when present)...')
    cnn_model = tf.keras.models.load_model(cfg['model_path'], compile=False)
    assist_model = tf.keras.models.load_model(cfg['assist_model_path'], compile=False)
    sample_storage, sample_attributes = build_pool_attributes(
        pool, cnn_model, assist_model, cfg, CACHE_DIR,
    )
    # Free GPU memory; greedy only needs CPU arrays
    del cnn_model, assist_model
    tf.keras.backend.clear_session()

    max_budget = max(BUDGET_RATIOS)
    configs = list(product(RISK_GATE_POWERS, ALPHAS, PHI_SETTINGS))
    print(f'Running {len(configs)} greedy configs (no greedy cache writes)...')

    rows = []
    for i, (power, alpha, (phi_mode, phi_tau)) in enumerate(configs, start=1):
        greedy_cfg = {
            'alpha': float(alpha),
            'phi_mode': str(phi_mode),
            'phi_tau': None if phi_tau is None else float(phi_tau),
            'risk_gate_power': float(power),
        }
        print(
            f'[{i}/{len(configs)}] p={power}, alpha={alpha}, '
            f'phi={phi_mode}, tau={phi_tau}',
        )
        trc_by_budget = run_one_config(
            sample_storage, sample_attributes, greedy_cfg, max_budget,
        )
        print(f'  TRC: {_fmt_trc(trc_by_budget)}')
        rows.append({
            'dataset': DATASET_NAME,
            'pool_type': POOL_TYPE,
            'sampling_mode': SAMPLING_MODE,
            'error_ratio': float(ERROR_RATIO),
            'seed': int(SEED),
            'risk_gate_power': float(power),
            'alpha': float(alpha),
            'phi_mode': str(phi_mode),
            'phi_tau': None if phi_tau is None else float(phi_tau),
            'trc_by_budget': trc_by_budget,
        })

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'pool': {
            'dataset': DATASET_NAME,
            'pool_type': POOL_TYPE,
            'sampling_mode': SAMPLING_MODE,
            'error_ratio': float(ERROR_RATIO),
            'seed': int(SEED),
            'n_pool': int(len(pool['data'])),
            'n_errors': int(np.sum(pool['is_error'])),
        },
        'grid': {
            'risk_gate_powers': list(RISK_GATE_POWERS),
            'alphas': list(ALPHAS),
            'phi_settings': [
                {'phi_mode': m, 'phi_tau': t} for m, t in PHI_SETTINGS
            ],
            'budgets': [float(b) for b in BUDGET_RATIOS],
        },
        'note': (
            'greedy_select in-memory only (no greedy_run_*.npz); '
            'snorkel_result / all_risk_features_explore_* loaded when present'
        ),
        'results': rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    OUT_MD.write_text(format_summary_md(rows), encoding='utf-8')
    print(f'\nSaved JSON: {OUT_JSON}')
    print(f'Saved markdown: {OUT_MD}')


if __name__ == '__main__':
    main()
