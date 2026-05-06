import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from load_data import load_fmnist, load_cifar10, load_svhn


def get_neural_value(model_path, layer_studied, pictures):

    cnn_model = tf.keras.models.load_model(model_path)

    print('\nget neural value in layer: {}'.format(layer_studied))
    get_layer_output = tf.keras.backend.function(inputs=cnn_model.layers[0].input,
                                                 outputs=cnn_model.layers[layer_studied].output)
    batch_size = 16
    batch_output_list = []
    for i in range(0, len(pictures), batch_size):
        batch = pictures[i:i + batch_size]
        batch_output = get_layer_output([batch])
        batch_output_list.append(batch_output)

    return np.concatenate(batch_output_list)


def get_relative_selectivity_for_single_class(model_path, layer_studied,
                                              train_pictures, train_labels,
                                              test_pictures, test_labels, number_of_classes):

    train_outputs = get_neural_value(model_path, layer_studied, train_pictures)

    train_outputs_classified = [[] for _ in range(number_of_classes)]
    for tro, trl in zip(train_outputs, train_labels):
        train_outputs_classified[trl].append(tro)

    class_c_sum = []
    number_of_examples = []
    for c in range(number_of_classes):
        neural_value_c = np.array(train_outputs_classified[c])
        number_of_examples.append(neural_value_c.shape[0])
        class_c_sum.append(np.sum(neural_value_c, axis=0))

    number_of_neurons = class_c_sum[0].shape[0]
    output_sum = np.sum(np.array(class_c_sum), axis=0)
    all_class_average = output_sum / np.sum(number_of_examples)
    not_class_c_average = [[] for _ in range(number_of_classes)]
    for k in range(number_of_neurons):    # for each neuron
        for c in range(number_of_classes):
            not_class_c_average[c].append((output_sum[k] - class_c_sum[c][k]) /
                                          (np.sum(number_of_examples) - number_of_examples[c]))

    test_outputs = get_neural_value(model_path, layer_studied, test_pictures)
    relative_selectivity_list = []
    for teo, tel in zip(test_outputs, test_labels):
        relative_selectivity = [0.0 for _ in range(number_of_neurons)]
        for k in range(number_of_neurons):
            if all_class_average[k] == 0:
                relative_selectivity[k] = 0.0
            else:
                relative_selectivity[k] = (teo[k] - not_class_c_average[tel][k]) / abs(all_class_average[k])
        relative_selectivity_list.append(relative_selectivity)

    return np.array(relative_selectivity_list)


def get_relative_selectivity_for_all_classes(model_path, layer_studied,
                                             train_pictures, train_labels,
                                             test_pictures, number_of_classes):

    train_outputs = get_neural_value(model_path, layer_studied, train_pictures)

    train_outputs_classified = [[] for _ in range(number_of_classes)]
    for tro, trl in zip(train_outputs, train_labels):
        train_outputs_classified[trl].append(tro)

    class_c_sum = []
    number_of_examples = []
    for c in range(number_of_classes):
        neural_value_c = np.array(train_outputs_classified[c])
        number_of_examples.append(neural_value_c.shape[0])
        class_c_sum.append(np.sum(neural_value_c, axis=0))

    number_of_neurons = class_c_sum[0].shape[0]
    output_sum = np.sum(np.array(class_c_sum), axis=0)
    all_class_average = output_sum / np.sum(number_of_examples)
    not_class_c_average = [[] for _ in range(number_of_classes)]
    for k in range(number_of_neurons):    # for each neuron
        for c in range(number_of_classes):
            not_class_c_average[c].append((output_sum[k] - class_c_sum[c][k]) /
                                          (np.sum(number_of_examples) - number_of_examples[c]))

    test_outputs = get_neural_value(model_path, layer_studied, test_pictures)
    relative_selectivity_list_all = [[] for _ in range(number_of_classes)]
    for c in range(number_of_classes):
        for teo in test_outputs:
            relative_selectivity = [0.0 for _ in range(number_of_neurons)]
            for k in range(number_of_neurons):
                if all_class_average[k] == 0:
                    relative_selectivity[k] = 0.0
                else:
                    relative_selectivity[k] = (teo[k] - not_class_c_average[c][k]) / abs(all_class_average[k])
            relative_selectivity_list_all[c].append(relative_selectivity)

    return np.array(relative_selectivity_list_all)


def bound_distance(bound_list, x, scale=False):
    distance_list = []
    for bound, k in zip(bound_list, x):
        if bound[0] <= k <= bound[1]:
            distance_list.append(0)
            # pass
        elif k < bound[0]:
            distance_list.append(1)
            # distance_list.append(abs(k-bound[0]) / scale)
            # distance_list.append((k - bound[0])**2)
        elif k > bound[1]:
            distance_list.append(1)
            # distance_list.append(abs(k-bound[1]) / scale)
            # distance_list.append((k - bound[1])**2)
        else:
            pass
    # print(distance_list)
    # return np.sum(sorted(distance_list, reverse=True)[:2])
    return np.sum(distance_list)
    # return np.sqrt(np.sum(distance_list)) / scale


def weighted_bound_distance(bound_list, x, weight):
    distance_list = []

    for bound, k in zip(bound_list, x):
        if bound[0] <= k <= bound[1]:
            distance_list.append(0)
        else:
            distance_list.append(1)

    # for bound, k, w in zip(bound_list, x, weight):
    #     if bound[0] <= k <= bound[1]:
    #         distance_list.append(0)
    #     else:
    #         if (k == 0 and w is True) or (k > 0 and w is False):
    #             distance_list.append(2)
    #         else:
    #             distance_list.append(1)

    return np.sum(distance_list)


def hidden_states_based_pseudo_labelling(train_hidden_states, train_labels,
                                         test_hidden_states, test_predictions,
                                         number_of_classes):
    train_hidden_states_classified = [[] for _ in range(number_of_classes)]
    for trh, trl in zip(train_hidden_states, train_labels):
        train_hidden_states_classified[trl].append(trh)

    bound_list = [[] for _ in range(number_of_classes)]
    whether_output_list = [[] for _ in range(number_of_classes)]
    # scale_list = []
    for i, trh_i in enumerate(train_hidden_states_classified):
        max_upper_bound = -100
        min_lower_bound = 100
        trh_i_trans = np.array(trh_i).transpose()
        for hs in trh_i_trans:
            # upper_bound = np.max(hs)
            # lower_bound = np.min(hs)
            upper_bound = np.percentile(hs, 99)
            lower_bound = np.percentile(hs, 1)
            bound_list[i].append((lower_bound, upper_bound))
            if np.mean(hs) == 0:
                whether_output_list[i].append(False)
            else:
                whether_output_list[i].append(True)
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
            if lower_bound < min_lower_bound:
                min_lower_bound = lower_bound
        # scale_list.append(abs(max_upper_bound - min_lower_bound))

    distance_list = []
    for c in range(number_of_classes):
        # print(f'Class-{c}: Scale: {scale_list[c]}')
        test_hs_c = test_hidden_states[c]
        distance_c = []
        for teh in test_hs_c:
            # distance_c.append(bound_distance(bound_list[c], teh))
            distance_c.append(weighted_bound_distance(bound_list[c], teh, whether_output_list[c]))
        distance_list.append(distance_c)

    pseudo_labels = []
    distance_transpose = np.array(distance_list).transpose()
    for distance, prediction in zip(distance_transpose, test_predictions):
        indices = np.argsort(distance)[:2]
        if indices[0] == prediction:
            pseudo_labels.append(indices[1])
        else:
            pseudo_labels.append(indices[0])

    return pseudo_labels


def confidence_based_pseudo_labelling(test_confidence, test_predictions):
    pseudo_labels = []
    for confidence, prediction in zip(test_confidence, test_predictions):
        indices = np.argsort(confidence)[-2:][::-1]
        if indices[0] == prediction:
            pseudo_labels.append(indices[1])
        else:
            pseudo_labels.append(indices[0])

    return pseudo_labels


if __name__ == '__main__':
    # dataset = 'fmnist'
    # model_dir = './models/lenet_fmnist/tf_model.h5'
    # layer = -4
    # train_data, train_label, test_data, test_label = load_fmnist()
    # attacks = ['ba', 'cw_l2', 'deepfool', 'ead', 'hopskipjumpattack_l2', 'jsma', 'newtonfool', 'pixelattack',
    #            'squareattack_linf', 'wassersteinattack']

    # dataset = 'cifar10'
    # model_dir = './models/vgg19_cifar10/tf_model.h5'
    # layer = -5
    # train_data, train_label, test_data, test_label = load_cifar10()
    # attacks = ['ba', 'cw_l2', 'cw_linf', 'deepfool', 'ead', 'hopskipjumpattack_l2', 'hopskipjumpattack_linf', 'jsma',
    #            'newtonfool', 'pixelattack', 'squareattack_l2', 'squareattack_linf', 'sta', 'wassersteinattack', 'zoo']

    dataset = 'svhn'
    model_dir = './models/resnet18_svhn/tf_model.h5'
    layer = -4
    train_data, train_label, test_data, test_label = load_svhn()
    attacks = ['ba', 'cw_l2', 'deepfool', 'ead', 'hopskipjumpattack_l2', 'jsma', 'newtonfool', 'pixelattack',
               'shadowattack', 'squareattack_linf', 'sta', 'wassersteinattack']

    model = tf.keras.models.load_model(model_dir)
    train_confidence = model.predict(train_data)
    train_predicted_label = np.argmax(train_confidence, axis=1)
    train_label = np.argmax(train_label, axis=1)
    correct_index = train_predicted_label == train_label
    train_correct_data = train_data[correct_index]
    train_correct_label = train_label[correct_index]
    train_correct_hidden_states = get_neural_value(model_dir, layer, train_correct_data)
    for attack in attacks:
        print(f'---------------------------------{attack}------------------------------------------')
        adv_data = np.load(f"./data/{dataset}/adversarial/{attack}_adv_data.npy")
        adv_targets = np.load(f"./data/{dataset}/adversarial/{attack}_adv_targets.npy")
        adv_clean_label = np.load(f"./data/{dataset}/adversarial/{attack}_clean_labels.npy")

        adv_hidden_states = get_neural_value(model_dir, layer, adv_data)
        adv_pseudo_labels = hidden_states_based_pseudo_labelling(train_correct_hidden_states, train_correct_label,
                                                                 [adv_hidden_states for _ in range(10)], adv_targets, 10)
        print(accuracy_score(adv_clean_label, adv_pseudo_labels))

        # train_rs = get_relative_selectivity_for_single_class(model_dir, layer, train_correct_data, train_correct_label,
        #                                                      train_correct_data, train_correct_label, 10)
        # adv_rs_all_classes = get_relative_selectivity_for_all_classes(model_dir, layer,
        #                                                               train_correct_data, train_correct_label,
        #                                                               adv_data, 10)
        # adv_pseudo_labels = hidden_states_based_pseudo_labelling(train_rs, train_correct_label,
        #                                                          adv_rs_all_classes, adv_targets, 10)
        # print(accuracy_score(adv_clean_label, adv_pseudo_labels))

        adv_confidence = model.predict(adv_data)
        adv_pseudo_labels = confidence_based_pseudo_labelling(adv_confidence, adv_targets)
        print(accuracy_score(adv_clean_label, adv_pseudo_labels))



