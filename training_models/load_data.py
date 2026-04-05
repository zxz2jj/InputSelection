import numpy as np
import os
import glob
import pandas as pd
from skimage import io, transform
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import datasets
import cv2

from tensorflow.keras import preprocessing

data_augmentation = preprocessing.image.ImageDataGenerator(
    # 布尔值，使输入数据集去中心化（均值为0）, 按feature执行
    featurewise_center=False,
    # 布尔值，使输入数据的每个样本均值为0
    samplewise_center=False,
    # 布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行
    featurewise_std_normalization=False,
    # 布尔值，将输入的每个样本除以其自身的标准差
    samplewise_std_normalization=False,
    # 布尔值，对输入数据施加ZCA白化
    zca_whitening=False,
    # ZCA使用的eposilon，默认1e-6
    zca_epsilon=1e-06,
    # 整数，数据提升时图片随机转动的角度 (deg 0 to 180)
    rotation_range=10,
    # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
    width_shift_range=0.1,
    # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    height_shift_range=0.1,
    # 浮点数，剪切强度（逆时针方向的剪切变换角度）
    shear_range=10.,
    # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper]
    #  = [1 - zoom_range, 1+zoom_range]
    zoom_range=0.,
    # 浮点数，随机通道偏移的幅度
    channel_shift_range=0.,
    # ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数
    # 给定的方法进行处理
    fill_mode='nearest',
    # 浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
    cval=0.,
    # 布尔值，是否进行随机水平翻转
    horizontal_flip=True,
    # 布尔值，是否进行随机竖直翻转
    vertical_flip=False,
    # 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其
    # 他变换之前)
    rescale=None,
    # 将被应用于每个输入的函数。该函数将在图片缩放和数据提升之后运行。该函数接受一个参数，
    # 为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
    preprocessing_function=None)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train_one_hot = tf.one_hot(y_train, 10)
    y_test_one_hot = tf.one_hot(y_test, 10)

    return x_train, y_train_one_hot, \
        x_test, y_test_one_hot


def load_fmnist():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train_one_hot = tf.one_hot(y_train, 10)
    y_test_one_hot = tf.one_hot(y_test, 10)

    return x_train, y_train_one_hot, \
        x_test, y_test_one_hot


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = y_train.reshape([-1])
    y_test = y_test.reshape([-1])
    y_train_one_hot = tf.one_hot(y_train, 10)
    y_test_one_hot = tf.one_hot(y_test, 10)

    return x_train, y_train_one_hot, \
        x_test, y_test_one_hot


def load_cifar100():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = y_train.reshape([-1])
    y_test = y_test.reshape([-1])
    y_train_one_hot = tf.one_hot(y_train, 100)
    y_test_one_hot = tf.one_hot(y_test, 100)

    return x_train, y_train_one_hot, \
        x_test, y_test_one_hot


def load_mat_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']


def load_svhn():
    x_train, y_train = load_mat_data('./data/svhn/train_32x32.mat')
    x_test, y_test = load_mat_data('./data/svhn/test_32x32.mat')
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    x_train = np.transpose(x_train, (3, 0, 1, 2))
    x_test = np.transpose(x_test, (3, 0, 1, 2))
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # io.imshow(x_test[3])
    # io.show()
    y_train[np.where(y_train == 10)] = 0
    y_test[np.where(y_test == 10)] = 0
    y_train_one_hot = tf.one_hot(y_train, 10)
    y_test_one_hot = tf.one_hot(y_test, 10)
    print("Training Set", x_train.shape, y_train.shape)
    print("Test Set", x_test.shape, y_test.shape)

    return x_train, y_train_one_hot, x_test, y_test_one_hot


def load_gtsrb():
    if os.path.exists('./data/public/GTSRB/train_data.npy') and os.path.exists('./data/public/GTSRB/train_labels.npy'):
        x_train = np.load('./data/public/GTSRB/train_data.npy')
        y_train_one_hot = np.load('./data/public/GTSRB/train_labels.npy')
    else:
        root_dir = './data/public/GTSRB/Final_Training/Images/'
        x_train = []
        y_train = []
        print('\nLoad GTSRB train dataset...')
        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

        for img_path in all_img_paths:
            img_path = img_path.replace('\\', '/')
            print('\r', img_path, end='')
            img = transform.resize(io.imread(img_path), (48, 48))
            label = int(img_path.split('/')[-2])
            x_train.append(img)
            y_train.append(label)

        x_train = np.array(x_train, dtype='float32')
        y_train_category = np.array(y_train)
        y_train_one_hot = tf.one_hot(y_train_category, 43)
        np.save('./data/public/GTSRB/train_data.npy', x_train)
        np.save('./data/public/GTSRB/train_labels.npy', y_train_one_hot)

    if os.path.exists('./data/public/GTSRB/test_data.npy') and os.path.exists('./data/public/GTSRB/test_labels.npy'):
        x_test = np.load('./data/public/GTSRB/test_data.npy')
        y_test_one_hot = np.load('./data/public/GTSRB/test_labels.npy')
    else:
        test = pd.read_csv('./data/public/GTSRB/Final_Test/Images/GT-final_test.csv', sep=';')
        x_test = []
        y_test = []
        print('\nLoad GTSRB test dataset...')
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join('./data/public/GTSRB/Final_Test/Images/', file_name)
            print('\r', img_path, end='')
            x_test.append(transform.resize(io.imread(img_path), (48, 48)))
            y_test.append(class_id)

        x_test = np.array(x_test)
        y_test_category = np.array(y_test)
        y_test_one_hot = tf.one_hot(y_test_category, 43)
        np.save('./data/public/GTSRB/test_data.npy', x_test)
        np.save('./data/public/GTSRB/test_labels.npy', y_test_one_hot)

    return x_train, y_train_one_hot, \
        x_test, y_test_one_hot


# def load_mat_data(path):
#     """ Helper function for loading a MAT-File"""
#     data = loadmat(path)
#     return data['X'], data['y']


# def load_svhn():
#     x_train, y_train = load_mat_data('./data/svhn/train_32x32.mat')
#     x_test, y_test = load_mat_data('./data/svhn/test_32x32.mat')
#     y_train = y_train.reshape(-1)
#     y_test = y_test.reshape(-1)
#     x_train = np.transpose(x_train, (3, 0, 1, 2))
#     x_test = np.transpose(x_test, (3, 0, 1, 2))
#     x_train = x_train.astype('float32') / 255.
#     x_test = x_test.astype('float32') / 255.
#     # io.imshow(x_test[3])
#     # io.show()
#     y_train[np.where(y_train == 10)] = 0
#     y_test[np.where(y_test == 10)] = 0
#     y_train_one_hot = tf.one_hot(y_train, 10)
#     y_test_one_hot = tf.one_hot(y_test, 10)
#     print("Training Set", x_train.shape, y_train.shape)
#     print("Test Set", x_test.shape, y_test.shape)
#
#     return x_train, y_train_one_hot, x_test, y_test_one_hot


def load_omniglot(resize=28):
    test_dir = './data/public/Omniglot/images_evaluation'
    x_test = []
    for alphabet in os.listdir(test_dir):
        alphabet_path = os.path.join(test_dir, alphabet)
        for character in os.listdir(alphabet_path):
            character_path = os.path.join(alphabet_path, character)
            for image in os.listdir(character_path):
                img_path = os.path.join(character_path, image)
                print('\r', img_path, end='')
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # img = io.imread(img_path)
                img = transform.resize(img, (resize, resize))
                x_test.append(img)
    x_test = np.array(x_test)
    x_test = 1 - x_test
    x_test = np.expand_dims(x_test, axis=-1)

    return x_test


def load_tiny(resize):
    test_dir = './data/public/TinyImageNet/tiny-imagenet-200/test/images'
    x_test = []
    for image in os.listdir(test_dir):
        img_path = os.path.join(test_dir, image)
        print('\r', img_path, end='')
        img = io.imread(img_path)
        img = transform.resize(img, (resize, resize, 3))
        x_test.append(img)
    x_test = np.array(x_test)
    # print(x_test.shape)
    # print(np.max(x_test), np.min(x_test))
    # io.imshow(x_test[0])
    # io.show()

    return x_test


def load_lsun(resize):
    test_dir = './data/public/LSUN'
    x_test = []
    for image in os.listdir(test_dir):
        img_path = os.path.join(test_dir, image)
        print('\r', img_path, end='')
        img = io.imread(img_path)
        img = img.astype('float32') / 255.
        img = transform.resize(img, (resize, resize, 3))
        x_test.append(img)
    x_test = np.array(x_test)
    # x_test = x_test.astype('float32') / 255.
    # print(x_test.shape)
    # print(np.max(x_test), np.min(x_test))
    # io.imshow(x_test[0])
    # io.show()

    return x_test


def load_isun(resize):
    test_dir = './data/public/iSUN'
    x_test = []
    for image in os.listdir(test_dir):
        img_path = os.path.join(test_dir, image)
        print('\r', img_path, end='')
        img = io.imread(img_path)
        img = img.astype('float32') / 255.
        img = transform.resize(img, (resize, resize, 3))
        x_test.append(img)
    x_test = np.array(x_test)
    # x_test = x_test.astype('float32') / 255.
    # print(x_test.shape)
    # print(np.max(x_test), np.min(x_test))
    # io.imshow(x_test[0])
    # io.show()

    return x_test


def load_ood_data(ood_dataset, id_model_path, num_of_categories):
    ood_test = None
    if ood_dataset == 'FMNIST':
        _, _, ood_test, _ = load_fmnist()
    if ood_dataset == 'MNIST':
        _, _, ood_test, _ = load_mnist()
    if ood_dataset == 'Omniglot':
        ood_test = load_omniglot()
    if ood_dataset == 'TinyImageNet':
        if id_model_path.split('/')[-2].split('_')[-1] in ['cifar10', 'cifar100', 'svhn']:
            ood_test = load_tiny(resize=32)
        elif id_model_path.split('/')[-2].split('_')[-1] in ['gtsrb']:
            ood_test = load_tiny(resize=48)
        else:
            ood_test = None
    if ood_dataset == 'LSUN':
        if id_model_path.split('/')[-2].split('_')[-1] in ['cifar10', 'cifar100', 'svhn']:
            ood_test = load_lsun(resize=32)
        elif id_model_path.split('/')[-2].split('_')[-1] in ['gtsrb']:
            ood_test = load_lsun(resize=48)
        else:
            ood_test = None
    if ood_dataset == 'iSUN':
        if id_model_path.split('/')[-2].split('_')[-1] in ['cifar10', 'cifar100', 'svhn']:
            ood_test = load_isun(resize=32)
        elif id_model_path.split('/')[-2].split('_')[-1] in ['gtsrb']:
            ood_test = load_isun(resize=48)
        else:
            ood_test = None
    if ood_dataset == 'UniformNoise_28':
        ood_test = np.load("./data/public/UniformNoise/uniform_noise_size=28.npy")
    if ood_dataset == 'GuassianNoise_28':
        ood_test = np.load("./data/public/GuassianNoise/guassian_noise_size=28.npy")
    if ood_dataset == 'UniformNoise_32':
        ood_test = np.load("./data/public/UniformNoise/uniform_noise_size=32.npy")
    if ood_dataset == 'GuassianNoise_32':
        ood_test = np.load("./data/public/GuassianNoise/guassian_noise_size=32.npy")
    if ood_dataset == 'UniformNoise_48':
        ood_test = np.load("./data/public/UniformNoise/uniform_noise_size=48.npy")
    if ood_dataset == 'GuassianNoise_48':
        ood_test = np.load("./data/public/GuassianNoise/guassian_noise_size=48.npy")

    number_of_test = ood_test.shape[0]
    model = tf.keras.models.load_model(id_model_path+'tf_model.h5')
    prediction = np.argmax(model.predict(ood_test), axis=-1)
    ood_dict = {}
    for c in range(num_of_categories):
        data_category = ood_test[prediction == c]
        prediction_category = prediction[prediction == c]
        if data_category.shape[0] == 0:
            ood_dict_category = {'correct_pictures': np.array([]), 'correct_prediction': np.array([]),
                                 'wrong_pictures': np.array([]), 'wrong_prediction': np.array([])}
        else:
            ood_dict_category = {'correct_pictures': np.array([]), 'correct_prediction': np.array([]),
                                 'wrong_pictures': data_category, 'wrong_prediction': prediction_category}
        ood_dict[c] = ood_dict_category

    return ood_dict, number_of_test


def load_clean_adv_data(id_dataset, attack, num_of_categories):
    adv_data = np.load(f'./data/{id_dataset}/adversarial/{attack}_adv_data.npy')
    adv_targets = np.load(f'./data/{id_dataset}/adversarial/{attack}_adv_targets.npy')
    clean_data = np.load(f'./data/{id_dataset}/adversarial/{attack}_clean_data.npy')
    clean_labels = np.load(f'./data/{id_dataset}/adversarial/{attack}_clean_labels.npy')

    number_of_test = clean_data.shape[0]

    clean_dict = {}
    for c in range(num_of_categories):
        data_category = clean_data[clean_labels == c]
        labels_category = clean_labels[clean_labels == c]
        clean_dict_category = {'correct_pictures': data_category, 'correct_prediction': labels_category,
                               'wrong_pictures': np.array([]), 'wrong_prediction': np.array([])}
        clean_dict[c] = clean_dict_category

    adv_dict = {}
    for c in range(num_of_categories):
        data_category = adv_data[adv_targets == c]
        targets_category = adv_targets[adv_targets == c]
        adv_dict_category = {'correct_pictures': np.array([]), 'correct_prediction': np.array([]),
                             'wrong_pictures': data_category, 'wrong_prediction': targets_category}
        adv_dict[c] = adv_dict_category

    return clean_dict, adv_dict, number_of_test


if __name__ == '__main__':
    load_mnist()


