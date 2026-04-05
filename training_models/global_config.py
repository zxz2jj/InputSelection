
num_of_labels = {
    'mnist': 10,
    'fmnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'gtsrb': 43,
    'svhn': 10,
}


cnn_config = {
    'mnist': {
        'layers_of_getting_value': [-10, -8, -6, -4],
        'neurons_of_each_layer': [320, 160, 80, 40],
        'ood_settings': ['FMNIST', 'Omniglot', 'UniformNoise_28', 'GuassianNoise_28']
    },
    'fmnist': {
        'layers_of_getting_value': [-10, -8, -6, -4],
        'neurons_of_each_layer': [320, 160, 80, 40],
        'ood_settings': ['MNIST', 'Omniglot', 'UniformNoise_28', 'GuassianNoise_28'],
        'adversarial_settings': ['ba', 'cw_l2',  'deepfool', 'ead', 'fgsm', 'hopskipjumpattack_l2',
                                 'jsma', 'newtonfool', 'pixelattack',  'squareattack_linf', 'wassersteinattack']
    },
    'cifar10': {
        'layers_of_getting_value': [-5],
        'neurons_of_each_layer': [512],
        'ood_settings': ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise_32', 'GuassianNoise_32'],
        'adversarial_settings': ['ba', 'cw_l2', 'deepfool', 'ead', 'fgsm', 'hopskipjumpattack_l2',
                                 'jsma', 'newtonfool', 'pixelattack', 'squareattack_linf', 'wassersteinattack']
    },
    'gtsrb': {
        'layers_of_getting_value': [-5],
        'neurons_of_each_layer': [512],
        'ood_settings': ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise_48', 'GuassianNoise_48']
    },
    'svhn': {
        'layers_of_getting_value': [-4],
        'neurons_of_each_layer': [512],
        'ood_settings': ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise_32', 'GuassianNoise_32'],
        'adversarial_settings': ['ba', 'cw_l2', 'deepfool', 'ead', 'fgsm', 'hopskipjumpattack_l2',
                                 'jsma', 'newtonfool', 'pixelattack', 'squareattack_linf', 'wassersteinattack']
    },
    'cifar100': {
        'layers_of_getting_value': [-4],
        'neurons_of_each_layer': [512],
        'ood_settings': ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise_32', 'GuassianNoise_32'],
    },
}


selective_rate = 0.2
