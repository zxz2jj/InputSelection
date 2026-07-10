import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent
for _p in (_REPO_ROOT, _EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training_models.load_data import load_cifar10, load_fmnist


ADVERSARIAL_PREFIXES = [
    'ba',
    'cw_l2',
    'ead',
    'hopskipjumpattack_l2',
    'jsma',
    'newtonfool',
    'pixelattack',
    'wassersteinattack',
]

TRANSFORMATION_PREFIXES = [
    'blur',
    'brightness',
    'contrast',
    'rotation',
    'scale',
    'shear',
    'shift',
]

DATASET_CONFIG = {
    'fmnist': {
        'loader': load_fmnist,
        'model_path': _REPO_ROOT / 'models' / 'lenet_fmnist' / 'tf_model.h5',
    },
    'cifar10': {
        'loader': load_cifar10,
        'model_path': _REPO_ROOT / 'models' / 'vgg19_cifar10' / 'tf_model.h5',
    },
}

POOL_PREFIXES = {
    'adversarial': ADVERSARIAL_PREFIXES,
    'transformation': TRANSFORMATION_PREFIXES,
}


@dataclass
class CorrectPool:
    data: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    confidence: np.ndarray
    source_index: np.ndarray


@dataclass
class ErrorSource:
    prefix: str
    data: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    confidence: np.ndarray
    confidence_bin: np.ndarray
    source_index: np.ndarray


def _as_label_vector(arr):
    out = np.asarray(arr)
    if out.ndim > 1:
        out = np.argmax(out, axis=-1)
    return out.reshape(-1).astype(np.int64, copy=False)


def _probs_from_model_outputs(raw):
    raw = np.asarray(raw, dtype=np.float64)
    row_sums = np.sum(raw, axis=-1)
    looks_like_probs = (
        np.all(raw >= 0.0)
        and np.all(raw <= 1.0)
        and np.all(np.abs(row_sums - 1.0) < 1e-2)
    )
    if looks_like_probs:
        return raw.astype(np.float32, copy=False)
    shifted = raw - np.max(raw, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    return probs.astype(np.float32, copy=False)


def predict_labels_and_confidence(model, data, batch_size):
    raw = model.predict(data, batch_size=batch_size, verbose=0)
    probs = _probs_from_model_outputs(raw)
    labels = np.argmax(probs, axis=-1).astype(np.int64, copy=False)
    confidence = np.max(probs, axis=-1).astype(np.float32, copy=False)
    return labels, confidence


def assign_confidence_bins(confidence, n_bins):
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    bins = np.digitize(conf, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1], right=False)
    return np.clip(bins, 0, int(n_bins) - 1).astype(np.int16, copy=False)


def load_correct_pool(dataset_name, model, batch_size):
    loader = DATASET_CONFIG[dataset_name]['loader']
    _, _, x_test, y_test = loader()
    y_test = _as_label_vector(y_test)
    predictions, confidence = predict_labels_and_confidence(model, x_test, batch_size)
    correct_mask = predictions == y_test
    correct_indices = np.where(correct_mask)[0].astype(np.int64, copy=False)
    if correct_indices.size == 0:
        raise ValueError(f'{dataset_name}: no correctly predicted clean test samples found')
    return CorrectPool(
        data=x_test[correct_mask],
        labels=y_test[correct_mask],
        predictions=predictions[correct_mask],
        confidence=confidence[correct_mask],
        source_index=correct_indices,
    )


def _check_required_source_files(source_dir, prefix):
    required = [
        source_dir / f'{prefix}_adv_data.npy',
        source_dir / f'{prefix}_clean_labels.npy',
    ]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'{prefix}: missing files under {source_dir}: {missing}')


def load_error_source(
    dataset_name,
    prefix,
    model,
    batch_size,
    confidence_bins,
    filter_successful=True,
):
    source_dir = _REPO_ROOT / 'data' / dataset_name / 'adversarial'
    _check_required_source_files(source_dir, prefix)

    adv_data = np.load(source_dir / f'{prefix}_adv_data.npy')
    clean_labels = _as_label_vector(np.load(source_dir / f'{prefix}_clean_labels.npy'))

    n = len(adv_data)
    if len(clean_labels) != n:
        raise ValueError(
            f'{dataset_name}/{prefix}: length mismatch between adv_data and clean_labels',
        )

    predictions, confidence = predict_labels_and_confidence(model, adv_data, batch_size)

    keep = np.ones(n, dtype=bool)
    if filter_successful:
        keep &= predictions != clean_labels
    if not np.any(keep):
        raise ValueError(f'{dataset_name}/{prefix}: no erroneous generated samples after filtering')

    kept_conf = confidence[keep]
    return ErrorSource(
        prefix=prefix,
        data=adv_data[keep],
        labels=clean_labels[keep],
        predictions=predictions[keep],
        confidence=kept_conf,
        confidence_bin=assign_confidence_bins(kept_conf, confidence_bins),
        source_index=np.where(keep)[0].astype(np.int64, copy=False),
    )


def confidence_stratified_sample_indices(confidence_bin, target_count, n_bins, rng):
    bins = np.asarray(confidence_bin, dtype=np.int64).reshape(-1)
    target_count = int(target_count)
    if target_count < 0:
        raise ValueError('target_count must be non-negative')
    if target_count > len(bins):
        raise ValueError(f'target_count={target_count} exceeds population={len(bins)}')
    if target_count == 0:
        return np.empty(0, dtype=np.int64)

    per_bin_indices = []
    for b in range(int(n_bins)):
        idx = np.where(bins == b)[0].astype(np.int64, copy=False)
        rng.shuffle(idx)
        per_bin_indices.append(idx)

    bin_order = np.arange(int(n_bins), dtype=np.int64)
    rng.shuffle(bin_order)

    base = target_count // int(n_bins)
    remainder = target_count % int(n_bins)
    desired = np.full(int(n_bins), base, dtype=np.int64)
    desired[bin_order[:remainder]] += 1

    take_counts = np.array(
        [min(int(desired[b]), len(per_bin_indices[b])) for b in range(int(n_bins))],
        dtype=np.int64,
    )
    remaining = target_count - int(np.sum(take_counts))

    while remaining > 0:
        progressed = False
        rng.shuffle(bin_order)
        for b in bin_order:
            capacity = len(per_bin_indices[int(b)]) - int(take_counts[int(b)])
            if capacity <= 0:
                continue
            take_counts[int(b)] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    chosen = [
        per_bin_indices[b][: int(take_counts[b])]
        for b in range(int(n_bins))
        if take_counts[b] > 0
    ]
    out = np.concatenate(chosen).astype(np.int64, copy=False)
    rng.shuffle(out)
    if len(out) != target_count:
        raise RuntimeError(f'failed to sample requested count: got {len(out)}, expected {target_count}')
    return out


def compute_per_source_error_count(correct_count_available, error_ratio, source_counts, max_errors_per_source):
    rho = float(error_ratio)
    if rho <= 0.0 or rho >= 1.0:
        raise ValueError(f'error_ratio must be in (0, 1), got {rho}')
    n_sources = len(source_counts)
    if n_sources == 0:
        raise ValueError('source_counts is empty')

    max_total_errors_by_correct = int(np.floor(correct_count_available * rho / (1.0 - rho)))
    per_source = min(int(np.min(source_counts)), max_total_errors_by_correct // n_sources)
    if max_errors_per_source is not None:
        per_source = min(per_source, int(max_errors_per_source))
    if per_source <= 0:
        raise ValueError(
            'not enough correct/error samples to build this pool. '
            f'correct_available={correct_count_available}, error_ratio={rho}, '
            f'source_counts={source_counts}',
        )
    return int(per_source)


def build_sampled_pool(
    correct_pool,
    error_sources,
    *,
    dataset_name,
    pool_type,
    error_ratio,
    confidence_bins,
    per_source_error_count,
    seed,
):
    rng = np.random.default_rng(int(seed))
    source_error_chunks = []
    source_label_chunks = []
    source_pred_chunks = []
    source_conf_chunks = []
    source_bin_chunks = []
    source_name_chunks = []
    source_kind_chunks = []
    source_index_chunks = []
    is_error_chunks = []

    for src in error_sources:
        idx = confidence_stratified_sample_indices(
            src.confidence_bin,
            per_source_error_count,
            confidence_bins,
            rng,
        )
        n = len(idx)
        source_error_chunks.append(src.data[idx])
        source_label_chunks.append(src.labels[idx])
        source_pred_chunks.append(src.predictions[idx])
        source_conf_chunks.append(src.confidence[idx])
        source_bin_chunks.append(src.confidence_bin[idx])
        source_name_chunks.append(np.full(n, src.prefix, dtype='<U64'))
        source_kind_chunks.append(np.full(n, pool_type, dtype='<U32'))
        source_index_chunks.append(src.source_index[idx])
        is_error_chunks.append(np.ones(n, dtype=bool))

    error_data = np.concatenate(source_error_chunks, axis=0)
    error_labels = np.concatenate(source_label_chunks, axis=0)
    error_predictions = np.concatenate(source_pred_chunks, axis=0)
    error_confidence = np.concatenate(source_conf_chunks, axis=0)
    error_bins = np.concatenate(source_bin_chunks, axis=0)
    error_sources_arr = np.concatenate(source_name_chunks, axis=0)
    error_kind_arr = np.concatenate(source_kind_chunks, axis=0)
    error_source_index = np.concatenate(source_index_chunks, axis=0)
    error_mask = np.concatenate(is_error_chunks, axis=0)

    n_errors = len(error_data)
    n_correct = int(round(n_errors * (1.0 - float(error_ratio)) / float(error_ratio)))
    if n_correct > len(correct_pool.data):
        raise ValueError(
            f'need {n_correct} correct samples, but only {len(correct_pool.data)} are available',
        )

    correct_idx = rng.choice(len(correct_pool.data), size=n_correct, replace=False)
    correct_bins = assign_confidence_bins(correct_pool.confidence[correct_idx], confidence_bins)

    data = np.concatenate([correct_pool.data[correct_idx], error_data], axis=0)
    clean_labels = np.concatenate([correct_pool.labels[correct_idx], error_labels], axis=0)
    predictions = np.concatenate([correct_pool.predictions[correct_idx], error_predictions], axis=0)
    confidence = np.concatenate([correct_pool.confidence[correct_idx], error_confidence], axis=0)
    confidence_bin = np.concatenate([correct_bins, error_bins], axis=0)
    is_error = np.concatenate([np.zeros(n_correct, dtype=bool), error_mask], axis=0)
    source_prefix = np.concatenate([
        np.full(n_correct, 'clean_test', dtype='<U64'),
        error_sources_arr,
    ])
    source_kind = np.concatenate([
        np.full(n_correct, 'clean', dtype='<U32'),
        error_kind_arr,
    ])
    source_local_index = np.concatenate([
        correct_pool.source_index[correct_idx],
        error_source_index,
    ]).astype(np.int64, copy=False)

    order = np.arange(len(data), dtype=np.int64)
    rng.shuffle(order)

    metadata = {
        'dataset': dataset_name,
        'pool_type': pool_type,
        'error_ratio': float(error_ratio),
        'seed': int(seed),
        'confidence_bins': int(confidence_bins),
        'per_source_error_count': int(per_source_error_count),
        'num_sources': len(error_sources),
        'num_errors': int(n_errors),
        'num_correct': int(n_correct),
        'num_total': int(len(data)),
        'source_prefixes': [src.prefix for src in error_sources],
        'sampling': 'source-balanced and confidence-stratified generated errors',
    }

    return {
        'data': data[order],
        'clean_labels': clean_labels[order].astype(np.int64, copy=False),
        'predictions': predictions[order].astype(np.int64, copy=False),
        'is_error': is_error[order],
        'confidence': confidence[order].astype(np.float32, copy=False),
        'confidence_bin': confidence_bin[order].astype(np.int16, copy=False),
        'source_prefix': source_prefix[order],
        'source_kind': source_kind[order],
        'source_local_index': source_local_index[order],
        'metadata_json': np.array(json.dumps(metadata, sort_keys=True)),
    }, metadata


def save_pool(out_root, dataset_name, pool_type, error_ratio, seed, arrays, metadata, overwrite):
    ratio_name = f'error_ratio_{int(round(float(error_ratio) * 100)):02d}'
    out_dir = Path(out_root) / dataset_name / pool_type / ratio_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f'seed_{int(seed)}.npz'
    out_json = out_dir / f'seed_{int(seed)}.json'
    if out_npz.exists() and not overwrite:
        print(f'Skip existing {out_npz}')
        return
    np.savez_compressed(out_npz, **arrays)
    out_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    print(
        f'Saved {out_npz} '
        f'(total={metadata["num_total"]}, errors={metadata["num_errors"]}, '
        f'correct={metadata["num_correct"]})',
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build source- and confidence-balanced mixed pools for input selection experiments.',
    )
    parser.add_argument('--datasets', nargs='+', default=['fmnist', 'cifar10'], choices=sorted(DATASET_CONFIG))
    parser.add_argument(
        '--pool-types',
        nargs='+',
        default=['adversarial', 'transformation'],
        choices=sorted(POOL_PREFIXES),
    )
    parser.add_argument('--error-ratios', nargs='+', type=float, default=[0.10, 0.20])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--confidence-bins', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-errors-per-source', type=int, default=None)
    parser.add_argument('--output-root', default=str(_EXP_DIR / 'sampled_data'))
    parser.add_argument('--include-unsuccessful', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if int(args.confidence_bins) <= 0:
        raise ValueError('--confidence-bins must be positive')

    all_manifest = []
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIG[dataset_name]
        model_path = Path(cfg['model_path'])
        if not model_path.is_file():
            raise FileNotFoundError(f'model not found for {dataset_name}: {model_path}')

        print(f'Loading model for {dataset_name}: {model_path}')
        model = tf.keras.models.load_model(model_path, compile=False)
        correct_pool = load_correct_pool(dataset_name, model, args.batch_size)
        print(f'{dataset_name}: correct clean test samples = {len(correct_pool.data)}')

        for pool_type in args.pool_types:
            prefixes = POOL_PREFIXES[pool_type]
            print(f'{dataset_name}/{pool_type}: loading {len(prefixes)} sources')
            error_sources = [
                load_error_source(
                    dataset_name,
                    prefix,
                    model,
                    args.batch_size,
                    args.confidence_bins,
                    filter_successful=not args.include_unsuccessful,
                )
                for prefix in prefixes
            ]
            source_counts = [len(src.data) for src in error_sources]
            print(f'{dataset_name}/{pool_type}: available erroneous samples by source = {dict(zip(prefixes, source_counts))}')

            for error_ratio in args.error_ratios:
                per_source_count = compute_per_source_error_count(
                    len(correct_pool.data),
                    error_ratio,
                    source_counts,
                    args.max_errors_per_source,
                )
                total_errors = per_source_count * len(error_sources)
                total_correct = int(round(total_errors * (1.0 - float(error_ratio)) / float(error_ratio)))
                print(
                    f'{dataset_name}/{pool_type}/ratio={error_ratio:.2f}: '
                    f'per_source_errors={per_source_count}, total_errors={total_errors}, '
                    f'total_correct={total_correct}',
                )
                if args.dry_run:
                    continue

                for seed in args.seeds:
                    arrays, metadata = build_sampled_pool(
                        correct_pool,
                        error_sources,
                        dataset_name=dataset_name,
                        pool_type=pool_type,
                        error_ratio=error_ratio,
                        confidence_bins=args.confidence_bins,
                        per_source_error_count=per_source_count,
                        seed=seed,
                    )
                    save_pool(
                        args.output_root,
                        dataset_name,
                        pool_type,
                        error_ratio,
                        seed,
                        arrays,
                        metadata,
                        args.overwrite,
                    )
                    all_manifest.append(metadata)

    if not args.dry_run:
        manifest_path = Path(args.output_root) / 'manifest.json'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(all_manifest, indent=2, sort_keys=True), encoding='utf-8')
        print(f'Saved manifest: {manifest_path}')


if __name__ == '__main__':
    main()

