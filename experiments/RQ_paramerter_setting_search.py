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

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
logging.getLogger().setLevel(logging.WARNING)

from RQ1_fault_revealing import (
    DATASET_CONFIG,
    RQ1_BUDGET_RATIOS,
    _as_label_vector,
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

# --- Search grid (narrowed) ---
ALPHAS = (0.5, 0.7, 0.9)
FIXED_RISK_GATE_POWER = 1.0  # multiply by r, no tunable exponent
PHI_MODES = ('sqrt', 'log')

BUDGET_RATIOS = list(RQ1_BUDGET_RATIOS)
SAMPLED_ROOT = _EXP_DIR / 'sampled_data'
CACHE_DIR = _EXP_DIR / 'cache_files' / DATASET_NAME
OUT_JSON = CACHE_DIR / 'rq1_greedy_shortlist_param_search.json'
OUT_MD = CACHE_DIR / 'rq1_greedy_shortlist_param_search.md'


def _budget_key(b):
    return f'{float(b):.2f}'


def _fmt_trc(trc_by_budget, budgets=BUDGET_RATIOS):
    return ', '.join(
        f'{b:.0%}={trc_by_budget[_budget_key(b)]:.4f}' for b in budgets
    )


def risk_top_k_shortlist(attribute_records, k):
    """Keep the top-k attribute records by risk_score (desc)."""
    k = int(k)
    if k <= 0:
        raise ValueError(f'k must be positive, got {k}')
    ranked = sorted(
        attribute_records,
        key=lambda rec: (float(rec['risk_score']), -int(rec['idx'])),
        reverse=True,
    )
    return ranked[: min(k, len(ranked))]


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


def run_one_budget(
    sample_storage,
    sample_attributes,
    *,
    budget_ratio,
    shortlist_multiplier,
    greedy_cfg,
):
    """Per-budget risk shortlist + greedy; TRC vs global n."""
    n_pool = len(sample_storage)
    if n_pool != len(sample_attributes):
        raise ValueError('storage / attributes length mismatch')

    b = float(budget_ratio)
    s = int(np.ceil(b * n_pool))
    s = max(1, min(s, n_pool))
    k = int(np.ceil(b * n_pool * float(shortlist_multiplier)))
    k = max(s, min(k, n_pool))  # need at least s candidates

    shortlist = risk_top_k_shortlist(sample_attributes, k)
    # selection_ratio so that ceil(ratio * k) == s
    selection_ratio = float(s) / float(len(shortlist))
    greedy_order, _state = greedy_select(
        shortlist,
        selection_ratio=selection_ratio,
        greedy_cfg=greedy_cfg,
    )
    if len(greedy_order) < s:
        raise RuntimeError(
            f'greedy returned {len(greedy_order)} < required s={s} '
            f'(b={b}, k={k}, m={shortlist_multiplier})',
        )
    # TRC on full pool with global n (budget_ratios=[b] => denom uses ceil(b*n))
    curve = compute_greedy_selection_curves(
        sample_storage,
        greedy_order,
        None,
        budget_ratios=[b],
    )
    trc = float(np.asarray(curve['trc'], dtype=np.float64).reshape(-1)[0])
    return {
        'budget': b,
        'n_pool': int(n_pool),
        'n_select': int(s),
        'n_shortlist': int(len(shortlist)),
        'selection_ratio_on_shortlist': float(selection_ratio),
        'trc': trc,
    }


def run_one_config(
    sample_storage,
    sample_attributes,
    *,
    shortlist_multiplier,
    alpha,
    phi_mode,
    budgets=BUDGET_RATIOS,
):
    greedy_cfg = {
        'alpha': float(alpha),
        'phi_mode': str(phi_mode),
        'phi_tau': None,
        'risk_gate_power': float(FIXED_RISK_GATE_POWER),
    }
    trc_by_budget = {}
    per_budget_meta = {}
    for b in budgets:
        info = run_one_budget(
            sample_storage,
            sample_attributes,
            budget_ratio=b,
            shortlist_multiplier=shortlist_multiplier,
            greedy_cfg=greedy_cfg,
        )
        trc_by_budget[_budget_key(b)] = float(info['trc'])
        per_budget_meta[_budget_key(b)] = {
            'n_select': info['n_select'],
            'n_shortlist': info['n_shortlist'],
            'selection_ratio_on_shortlist': info['selection_ratio_on_shortlist'],
        }
    return trc_by_budget, per_budget_meta


def format_summary_md(rows, *, multipliers, budgets=BUDGET_RATIOS):
    headers = [
        'm', 'alpha', 'phi_mode', 'risk_gate_power',
        *[f'TRC@{b:.0%}' for b in budgets],
    ]
    lines = [
        '# RQ1 greedy shortlist hyperparameter search (FMNIST)',
        '',
        f'- Pool: `{DATASET_NAME}/{POOL_TYPE}/{SAMPLING_MODE}/'
        f'ratio={ERROR_RATIO:.2f}/seed={SEED}`',
        f'- Shortlist: per-budget top-k, k=ceil(b*n*m), m={list(multipliers)}',
        f'- Gate: risk_gate_power={FIXED_RISK_GATE_POWER} (multiply by r, no exponent sweep)',
        f'- Grid: alpha={list(ALPHAS)}, phi_mode={list(PHI_MODES)}',
        f'- Budgets: {budgets}',
        '- Note: explore-only in this script; greedy_run caches not written; '
        'snorkel/risk caches reused. TRC always uses global n.',
        '',
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        cells = [
            f"{row['risk_shortlist_multiplier']:.1f}",
            f"{row['alpha']:.1f}",
            str(row['phi_mode']),
            f"{row['risk_gate_power']:.1f}",
        ]
        for b in budgets:
            cells.append(f"{row['trc_by_budget'][_budget_key(b)]:.4f}")
        lines.append('| ' + ' | '.join(cells) + ' |')

    lines.extend(['', '## Best by budget'])
    for b in budgets:
        key = _budget_key(b)
        best = max(rows, key=lambda r: r['trc_by_budget'][key])
        lines.append(
            f"- @{b:.0%}: m={best['risk_shortlist_multiplier']:.1f}, "
            f"alpha={best['alpha']:.1f}, phi={best['phi_mode']} "
            f"-> TRC={best['trc_by_budget'][key]:.4f}",
        )
    for row in rows:
        vals = [row['trc_by_budget'][_budget_key(b)] for b in budgets]
        row['_mean_trc'] = float(np.mean(vals))
    best_mean = max(rows, key=lambda r: r['_mean_trc'])
    lines.append(
        f"- mean(1/3/5/10%): m={best_mean['risk_shortlist_multiplier']:.1f}, "
        f"alpha={best_mean['alpha']:.1f}, phi={best_mean['phi_mode']} "
        f"-> mean_TRC={best_mean['_mean_trc']:.4f}",
    )
    return '\n'.join(lines) + '\n'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'RQ1 greedy param search with per-budget risk shortlist '
            '(fmnist adversarial random r10 seed0 only).'
        ),
    )
    parser.add_argument(
        '--risk-shortlist-multiplier',
        type=float,
        nargs='+',
        default=[3.0],
        help='m in k=ceil(b*n*m); default: 3 only',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    multipliers = [float(x) for x in args.risk_shortlist_multiplier]
    if any(m <= 0 for m in multipliers):
        raise ValueError('--risk-shortlist-multiplier values must be > 0')

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
    del cnn_model, assist_model
    tf.keras.backend.clear_session()

    configs = [
        (m, alpha, phi)
        for m in multipliers
        for alpha in ALPHAS
        for phi in PHI_MODES
    ]
    print(
        f'Running {len(configs)} configs '
        f'(m={multipliers}, alpha={list(ALPHAS)}, phi={list(PHI_MODES)}, '
        f'p={FIXED_RISK_GATE_POWER}; per-budget shortlist; no greedy cache)...',
    )

    rows = []
    for i, (m, alpha, phi_mode) in enumerate(configs, start=1):
        print(f'[{i}/{len(configs)}] m={m}, alpha={alpha}, phi={phi_mode}')
        trc_by_budget, per_budget_meta = run_one_config(
            sample_storage,
            sample_attributes,
            shortlist_multiplier=m,
            alpha=alpha,
            phi_mode=phi_mode,
        )
        print(f'  TRC: {_fmt_trc(trc_by_budget)}')
        for b in BUDGET_RATIOS:
            meta = per_budget_meta[_budget_key(b)]
            print(
                f'    @{b:.0%}: shortlist={meta["n_shortlist"]}, '
                f'select={meta["n_select"]}',
            )
        rows.append({
            'dataset': DATASET_NAME,
            'pool_type': POOL_TYPE,
            'sampling_mode': SAMPLING_MODE,
            'error_ratio': float(ERROR_RATIO),
            'seed': int(SEED),
            'risk_shortlist_multiplier': float(m),
            'alpha': float(alpha),
            'phi_mode': str(phi_mode),
            'phi_tau': None,
            'risk_gate_power': float(FIXED_RISK_GATE_POWER),
            'trc_by_budget': trc_by_budget,
            'per_budget': per_budget_meta,
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
        'design': {
            'per_budget_shortlist': True,
            'shortlist_formula': 'k=min(n, ceil(b*n*m)), select s=ceil(b*n)',
            'trc_uses_global_n': True,
            'risk_gate': 'r^1 (power fixed at 1)',
        },
        'grid': {
            'risk_shortlist_multipliers': multipliers,
            'alphas': list(ALPHAS),
            'phi_modes': list(PHI_MODES),
            'risk_gate_power': float(FIXED_RISK_GATE_POWER),
            'budgets': [float(b) for b in BUDGET_RATIOS],
        },
        'note': (
            'explore-only in RQ1_fault_revealing_paramerter_setting.py; '
            'no greedy_run_*.npz; snorkel/risk caches loaded when present'
        ),
        'results': rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    # markdown uses the first/default m label in header context; include all rows
    OUT_MD.write_text(
        format_summary_md(rows, multipliers=multipliers),
        encoding='utf-8',
    )
    print(f'\nSaved JSON: {OUT_JSON}')
    print(f'Saved markdown: {OUT_MD}')


if __name__ == '__main__':
    main()
