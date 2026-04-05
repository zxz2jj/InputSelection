import random

import tensorflow as tf
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ..training_models.load_data import load_fmnist, load_cifar10


_rotation_augment = tf.keras.layers.RandomRotation(
    0.03, fill_mode='nearest', interpolation='bilinear')
_translation_augment = tf.keras.layers.RandomTranslation(
    0.05, 0.05, fill_mode='nearest', interpolation='bilinear')
_contrast_augment = tf.keras.layers.RandomContrast(0.20)


def keras_logits_fn(model, logits_layer_index=-2):
    logits_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[logits_layer_index].output,
        name=model.name + '_logits' if model.name else 'logits_submodel',
    )

    def _logit_fn(batch_tensor):
        return logits_model(batch_tensor, training=False)

    return _logit_fn


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
                      logit_fn,
                      batch_size=16,
                      num_augmentations=5,
                      ):

    if data_without_labelling is None:
        raise TypeError('data_without_labelling is None')
    if logit_fn is None:
        raise TypeError('logit_fn is None')

    entropy_list = []
    energy_list = []
    top12_margin_risk_list = []
    pred_classes_list = []
    stability_flip_list = []
    stability_maxvar_list = []
    stability_mkl_list = []

    eps = tf.constant(1e-8, tf.float32)

    for i in tqdm(range(0, len(data_without_labelling), batch_size)):
        batch = data_without_labelling[i:i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch)

        logits = tf.cast(tf.convert_to_tensor(logit_fn(batch_tensor)), tf.float32)
        probs = tf.nn.softmax(logits, axis=-1)
        prediction_entropy = -tf.reduce_sum(tf.math.xlogy(probs, probs), axis=-1)
        energy_score = -tf.math.reduce_logsumexp(logits, axis=-1)

        top2_vals, _ = tf.nn.top_k(probs, k=2)
        top12_margin_risk = 1.0 - (top2_vals[:, 0] - top2_vals[:, 1])

        pred_classes = tf.argmax(probs, axis=-1)

        aug_prob_list = []
        chosen_ids = np.arange(1, num_augmentations+1)
        for tid in chosen_ids:
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

    return {
        'prediction_entropy': np.concatenate(entropy_list),
        'energy_score': np.concatenate(energy_list),
        'top12_margin': np.concatenate(top12_margin_risk_list),
        'pred_classes': np.concatenate(pred_classes_list),
        'stability_class_change_rate': np.concatenate(stability_flip_list),
        'stability_max_prob_variance': np.concatenate(stability_maxvar_list),
        'stability_mean_kl': np.concatenate(stability_mkl_list),
    }


if __name__ == "__main__":
    dataset_name = 'fmnist'
    # dataset_name = 'cifar10'

    if dataset_name == 'fmnist':
        model_path = '../models/lenet_fmnist/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        x_train, y_train, x_test, y_test = load_fmnist()
        adv_dir = Path('../data/fmnist/adversarial')
    elif dataset_name == 'cifar10':
        model_path = '../models/vgg19_cifar10/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        x_train, y_train, x_test, y_test = load_cifar10()
        adv_dir = Path('../data/cifar10/adversarial')
    else:
        exit()
    if np.asarray(y_train).ndim > 1:
        y_train = np.argmax(y_train, axis=-1)
    if np.asarray(y_test).ndim > 1:
        y_test = np.argmax(y_test, axis=-1)

    logit_fn = keras_logits_fn(cnn_model)
    risk_features = get_risk_features(x_test, logit_fn, batch_size=16)
    print(risk_features)

