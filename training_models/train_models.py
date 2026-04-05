import os
import tensorflow
from load_data import load_mnist, load_fmnist, load_cifar10, load_svhn
from model_lenet import LeNetModel
from model_vgg import VGGModel
from model_resnet18 import ResNet18Model
# from model_vgg import VGGModel
# from model_resnet18 import ResNet18Model
from global_config import num_of_labels


def train_model(id_dataset, model_save_path, model_name):
    if id_dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist()
        model = LeNetModel(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                           test_data=x_test, test_label=y_test, model_save_path=model_save_path, model_name=model_name)
    elif id_dataset == 'fmnist':
        x_train, y_train, x_test, y_test = load_fmnist()
        model = LeNetModel(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                           test_data=x_test, test_label=y_test, model_save_path=model_save_path, model_name=model_name)
    elif id_dataset == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar10()
        model = VGGModel(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                         test_data=x_test, test_label=y_test, model_save_path=model_save_path, model_name=model_name)
    elif id_dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn()
        model = ResNet18Model(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                              test_data=x_test, test_label=y_test,
                              model_save_path=model_save_path, model_name=model_name)
    # elif id_dataset == 'cifar100':
    #     x_train, y_train, x_test, y_test = load_cifar100()
    #     model = ResNet18Model(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
    #                           test_data=x_test, test_label=y_test,
    #                           model_save_path=model_save_path, model_name=model_name)
    else:
        model = None

    if os.path.exists(model_save_path+model_name):
        print('{} is existed!'.format(model_save_path+model_name))
        model.show_model()
    else:
        model.train(epochs=100, is_used_data_augmentation=False)


if __name__ == "__main__":

    print(tensorflow.config.list_physical_devices('GPU'))
    # dataset = 'mnist'
    # save_path = '../models/lenet_mnist/'
    # save_name = 'tf_model.h5'

    dataset = 'fmnist'
    save_path = '../models/lenet_fmnist/'
    save_name = f'tf_model.h5'

    # dataset = 'cifar10'
    # save_path = '../models/vgg19_cifar10/'
    # save_name = f'tf_model.h5'

    # dataset = 'svhn'
    # save_path = '../models/resnet18_svhn/'
    # save_name = f'tf_model.h5'

    train_model(id_dataset=dataset, model_save_path=save_path, model_name=save_name)
