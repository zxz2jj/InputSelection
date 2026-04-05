import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_mnist, load_fmnist, load_cifar10, load_svhn


def show_confidence_distribution(model_checkpoint, data, labels=None, data_name=None):
    all_confidences = []
    correct_confidences = []
    wrong_confidences = []

    batch_size = 16
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch)
        outputs = model_checkpoint(batch_tensor, training=False)
        outputs = tf.convert_to_tensor(outputs)
        probs = tf.cast(outputs, tf.float32)
        row_sums = tf.reduce_sum(probs, axis=-1)
        looks_like_probs = tf.reduce_all((probs >= 0.0) & (probs <= 1.0)) & tf.reduce_all(
            tf.abs(row_sums - 1.0) < 1e-2
        )
        probs = tf.cond(looks_like_probs, lambda: probs, lambda: tf.nn.softmax(probs, axis=-1))

        confidences = np.asarray(tf.reduce_max(probs, axis=-1).numpy(), dtype=np.float32).reshape(-1)
        pred_classes = np.asarray(tf.argmax(probs, axis=-1).numpy()).reshape(-1)

        all_confidences.extend(confidences.tolist())
        if labels is not None:
            batch_labels = np.asarray(labels[i:i + batch_size])
            if batch_labels.ndim > 1:
                batch_labels = np.argmax(batch_labels, axis=-1)
            batch_labels = batch_labels.reshape(-1)
            correct_mask = pred_classes == batch_labels
            correct_confidences.extend(confidences[correct_mask].tolist())
            wrong_confidences.extend(confidences[~correct_mask].tolist())

    bins = np.linspace(0.0, 1.0, 11)  # 10 bins
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_width = (bins[1] - bins[0]) * 0.9

    if labels is None:
        legend_label = data_name if data_name is not None else 'all'
        title_name = f' ({data_name})' if data_name else ''
        plt.figure(figsize=(4, 3.5))
        counts, _ = np.histogram(np.asarray(all_confidences), bins=bins)
        plt.bar(bin_centers, counts, width=bin_width, alpha=0.85, label=legend_label)
        plt.title(f'Confidence Distribution ({len(all_confidences)} samples)')
        # plt.xlabel('confidence (max predicted probability)')
        # plt.ylabel('count')
        plt.xticks(bins)
        plt.grid(axis='y', linestyle='--', alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        correct_counts, _ = np.histogram(np.asarray(correct_confidences), bins=bins)
        wrong_counts, _ = np.histogram(np.asarray(wrong_confidences), bins=bins)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        axes[0].bar(bin_centers, correct_counts, width=bin_width, alpha=0.85, label='correct')
        axes[0].set_title(f'Correct Distribution ({len(correct_confidences)} samples)')
        # axes[0].set_xlabel('confidence (max predicted probability)')
        # axes[0].set_ylabel('count')
        axes[0].set_xticks(bins)
        axes[0].grid(axis='y', linestyle='--', alpha=0.35)
        axes[0].legend()

        axes[1].bar(bin_centers, wrong_counts, width=bin_width, alpha=0.85, label='wrong')
        axes[1].set_title(f'Wrong Distribution ({len(wrong_confidences)} samples)')
        # axes[1].set_xlabel('confidence (max predicted probability)')
        axes[1].set_xticks(bins)
        axes[1].grid(axis='y', linestyle='--', alpha=0.35)
        axes[1].legend()

        fig.tight_layout()
        plt.show()

    return


if __name__ == '__main__':
    # dataset_name = 'fmnist'
    dataset_name = 'cifar10'

    if dataset_name == 'fmnist':
        model_path = './models/lenet_fmnist/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        x_train, y_train, x_test, y_test = load_fmnist()
        adv_dir = Path('./data/fmnist/adversarial')
    elif dataset_name == 'cifar10':
        model_path = './models/vgg19_cifar10/tf_model.h5'
        cnn_model = tf.keras.models.load_model(model_path)
        x_train, y_train, x_test, y_test = load_cifar10()
        adv_dir = Path('./data/cifar10/adversarial')
    else:
        exit()
    if np.asarray(y_train).ndim > 1:
        y_train = np.argmax(y_train, axis=-1)
    if np.asarray(y_test).ndim > 1:
        y_test = np.argmax(y_test, axis=-1)

    show_confidence_distribution(cnn_model, x_test, y_test)

    adv_files = sorted(adv_dir.glob('*_adv_data.npy'))
    if not adv_files:
        print(f'No files matching *_adv_data.npy under {adv_dir.resolve()}')
    for adv_path in adv_files:
        adv_data = np.load(adv_path)
        adv_prefix = adv_path.name.removesuffix('_adv_data.npy')
        print(f'{adv_path.name}: shape={getattr(adv_data, "shape", None)}')
        show_confidence_distribution(cnn_model, adv_data, data_name=adv_prefix)
