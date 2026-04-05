from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from load_data import load_fmnist
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import random
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from scipy.special import softmax
from scipy.spatial.distance import euclidean, cityblock


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
                # relative_selectivity[k] = teo[k] - not_class_c_average[tel][k]
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
                    # relative_selectivity[k] = teo[k] - not_class_c_average[c][k]
            relative_selectivity_list_all[c].append(relative_selectivity)

    return np.array(relative_selectivity_list_all)


def get_base_ReAD_for_class(train_outputs, train_labels, number_of_classes):
    train_outputs_classified = [[] for _ in range(number_of_classes)]
    for tro, trl in zip(train_outputs, train_labels):
        train_outputs_classified[trl].append(tro)

    base_ReAD = []
    adaptive_num = []
    for i, tro_i in enumerate(train_outputs_classified):
        tro_i_trans = np.array(tro_i).transpose()
        class_ReAD = []
        num_ReA = 0
        num_ReD = 0
        for ho in tro_i_trans:
            positive_count = np.sum(ho > 0)
            positive_ratio = positive_count / ho.shape[0]
            negative_count = np.sum(ho < 0)
            negative_ratio = negative_count / ho.shape[0]
            if positive_ratio > 0.9:
                class_ReAD.append(1)
                num_ReA += 1
            elif negative_ratio > 0.8:
                num_ReD += 1
                class_ReAD.append(-1)
            else:
                class_ReAD.append(0)
        base_ReAD.append(class_ReAD)
        adaptive_num.append((num_ReA, num_ReD))
        # adaptive_num.append((5, 5))

    return base_ReAD, adaptive_num


def get_ReAD_for_example(outputs, adaptive_num):
    ReAD = [0 for _ in range(outputs.shape[0])]
    ReA_indices = np.argsort(outputs)[-adaptive_num[0]:]
    ReD_indices = np.argsort(outputs)[:adaptive_num[1]]
    for i in ReA_indices:
        ReAD[i] = 1
    for i in ReD_indices:
        ReAD[i] = -1

    return ReAD


def calculate_ReAD_distance(train_outputs, train_labels, test_outputs, number_of_classes):

    base_ReAD_list, adaptive_num_list = get_base_ReAD_for_class(train_outputs, train_labels, number_of_classes)

    distance_list = []
    for c in range(number_of_classes):
        test_outputs_c = test_outputs[c]
        distance_c = []
        for teo in test_outputs_c:
            ReAD = get_ReAD_for_example(teo, adaptive_num_list[c])
            # distance_c.append(euclidean_distance(ReAD, base_ReAD_list[c]))
            distance_c.append(cityblock_distance(ReAD, base_ReAD_list[c]))
        distance_list.append(distance_c)

    return np.array(distance_list)


def calculate_rs_distance(train_outputs, train_labels, test_outputs, number_of_classes):

    train_outputs_classified = [[] for _ in range(number_of_classes)]
    for tro, trl in zip(train_outputs, train_labels):
        train_outputs_classified[trl].append(tro)

    bound_list = [[] for _ in range(number_of_classes)]
    scale_list = []
    for i, tro_i in enumerate(train_outputs_classified):
        max_upper_bound = -100
        min_lower_bound = 100
        tro_i_trans = np.array(tro_i).transpose()
        for ho in tro_i_trans:
            # upper_bound = np.max(ho)
            # lower_bound = np.min(ho)
            upper_bound = np.percentile(ho, 99)
            lower_bound = np.percentile(ho, 1)
            bound_list[i].append((lower_bound, upper_bound))
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
            if lower_bound < min_lower_bound:
                min_lower_bound = lower_bound
        scale_list.append(abs(max_upper_bound - min_lower_bound))

    # mean_vector = [np.mean(tro, axis=0) for tro in train_outputs_classified]
    # cov_metrix = [np.cov(tro, rowvar=False)  for tro in train_outputs_classified]
    # inv_cov_list = []
    # for cov in cov_metrix:
    #     # noinspection PyBroadException
    #     try:
    #         inv_cov = inv(cov)
    #     except:
    #         # 如果协方差矩阵不可逆，使用伪逆
    #         inv_cov = np.linalg.pinv(cov)
    #     inv_cov_list.append(inv_cov)

    similarity_list = []
    for c in range(number_of_classes):
        print(f'Class-{c}: Scale: {scale_list[c]}')
        test_outputs_c = test_outputs[c]
        similarity_c = []
        for teo in test_outputs_c:
            # similarity_c.append(cosine_sim(teo, mean_vector[c]))
            # similarity_c.append(pearson_sim(teo, mean_vector[c]))
            # similarity_c.append(spearman_sim(teo, mean_vector[c]))
            # similarity_c.append(euclidean_distance(teo, mean_vector[c]))
            # similarity_c.append(mahalanobis(teo, mean_vector[c], inv_cov_list[c]))
            similarity_c.append(bound_distance(bound_list[c], teo, scale_list[c]))
        similarity_list.append(similarity_c)

    return np.array(similarity_list)


def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]


def pearson_sim(a, b):
    corr, p_value = pearsonr(a, b)
    return corr


def spearman_sim(a, b):
    corr, p_value = spearmanr(a, b)
    return corr


def euclidean_distance(a, b):
    return euclidean(a, b)


def cityblock_distance(a, b):
    return cityblock(a, b)


def bound_distance(bound_list, x, scale):
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


def plot_output_by_fault_type(adv_outputs, ori_labels, targets,
                              train_outputs, train_labels,
                              number_of_classes):

    adv_hidden_output_list = [[[] for _ in range(number_of_classes)] for _ in range(number_of_classes)]
    for ho, l, t in zip(adv_outputs, ori_labels, targets):
        adv_hidden_output_list[l][t].append(ho)

    for i in range(number_of_classes):
        # 原始类别的训练数据行为
        train_outputs_class_i = train_outputs[train_labels == 8].transpose()
        train_outputs_class_i_processed = []
        for ho in train_outputs_class_i:
            percentile_95 = np.percentile(ho, 99)
            percentile_5 = np.percentile(ho, 1)
            index95 = ho <= percentile_95
            index5 = ho >= percentile_5
            ho = ho[index95 & index5]
            train_outputs_class_i_processed.append(ho)

        for j in range(number_of_classes):
            # # 预测类别的训练数据行为
            # train_outputs_class_j = train_outputs[train_labels == j].transpose()
            # train_outputs_class_j_processed = []
            # for ho in train_outputs_class_j:
            #     percentile_95 = np.percentile(ho, 95)
            #     index = ho <= percentile_95
            #     ho = ho[index]
            #     train_outputs_class_j_processed.append(ho)

            # if i == 3 and j == 6:
            if adv_hidden_output_list[i][j]:
                # for _ in range(1):
                y = adv_hidden_output_list[i][j]
                x = [k+1 for k in range(len(y[0]))]
                print(f'Label-{i}, Target-{j}, Number of Examples: {len(y)}.')

                plt.figure(figsize=(8, 4))

                # parts1 = plt.violinplot(train_outputs_class_i_processed,
                #                         showmeans=True,
                #                         showmedians=True,
                #                         showextrema=True)
                # for pc in parts1['bodies']:
                #     pc.set_facecolor('skyblue')
                #     pc.set_edgecolor('gray')
                #     pc.set_alpha(0.8)
                # parts1['cmeans'].set_color('red')
                # parts1['cmedians'].set_color('green')
                # # parts1['cmaxes'].set_color('blue')
                # # parts1['cmins'].set_color('blue')
                # parts1['cbars'].set_color('gray')

                parts2 = plt.violinplot(train_outputs_class_i_processed,
                                        showmeans=True,
                                        showmedians=True,
                                        showextrema=True)
                for pc in parts2['bodies']:
                    # pc.set_facecolor('orangered')
                    pc.set_facecolor('skyblue')
                    pc.set_edgecolor('gray')
                    pc.set_alpha(1.0)
                parts2['cmeans'].set_color('blue')
                parts2['cmedians'].set_color('green')
                # parts2['cmaxes'].set_color('blue')
                # parts2['cmins'].set_color('blue')
                parts2['cbars'].set_color('gray')

                if len(y) > 8:
                    sampled_y = random.sample(y, 8)
                else:
                    sampled_y = y
                for k, yy in enumerate(sampled_y):
                    plt.plot(x, yy, alpha=0.7, label=f'({i}\u2192{j})')

                # plt.plot(x, train_outputs_class_i_average, linestyle='--', color='skyblue', alpha=0.5)
                # plt.plot(x, train_outputs_class_j_average, linestyle='--', color='orangered', alpha=0.5)

                # plt.title(f'Label-{i}, Target-{j}, Number of Examples: {len(y)}, '
                #           f'Blue Violin Class-{i}, Red Violin Class-{j}.')
                # plt.title(f'Clean Label: {i}, Adversarial Target: {j}', fontsize=12)
                plt.xticks(np.arange(1, len(x)+1, 2))
                plt.subplots_adjust(left=0.10,
                                    right=0.98,
                                    bottom=0.12,
                                    top=0.98)
                # plt.ylim(0, 10)
                # plt.tight_layout()
                plt.legend(loc="upper right", prop={'size': 12})
                plt.tick_params(labelsize=12)
                plt.xlabel('Index of Neurons', fontsize=12)
                plt.ylabel('Output Values of Dense(40)', fontsize=12)
                plt.hlines(y=0, xmin=0, xmax=len(x), color='red', linestyle='--', alpha=0.5)
                plt.show()
                # plt.savefig(f'fmnist-pa-rs-{i}-{j}-original.pdf')
                # plt.close()
                # exit()
    return


def plot_output_by_fault_reason(adv_outputs, ori_labels, targets,
                                train_outputs, train_labels,
                                number_of_classes):
    adv_hidden_output_list = [[[] for _ in range(number_of_classes)] for _ in range(number_of_classes)]
    for hs, l, t in zip(adv_outputs, ori_labels, targets):
        adv_hidden_output_list[l][t].append(hs)

    for j in range(number_of_classes):
        # 预测类别的训练数据行为
        train_outputs_class_j = train_outputs[train_labels == j].transpose()
        train_outputs_class_j_processed = []
        for ho in train_outputs_class_j:
            percentile_95 = np.percentile(ho, 95)
            index = ho <= percentile_95
            ho = ho[index]
            train_outputs_class_j_processed.append(ho)

        for _ in range(10):
            adv_output_list = []
            for i in range(number_of_classes):
                if adv_hidden_output_list[i][j]:
                    if len(adv_hidden_output_list[i][j]) > 3:
                        sampled_examples = random.sample(adv_hidden_output_list[i][j], 3)
                    else:
                        sampled_examples = adv_hidden_output_list[i][j]
                    for ho in sampled_examples:
                        adv_output_list.append((ho, i, j))

            x = [k + 1 for k in range(len(adv_output_list[0][0]))]
            print(f'Target-{j}, Number of Examples: {len(adv_output_list)}.')

            plt.figure(figsize=(8, 4))
            parts = plt.violinplot(train_outputs_class_j_processed,
                                   showmeans=True,
                                   showmedians=True,
                                   showextrema=True)
            for pc in parts['bodies']:
                # pc.set_facecolor('orangered')
                pc.set_facecolor('skyblue')
                pc.set_edgecolor('gray')
                pc.set_alpha(0.8)
            parts['cmeans'].set_color('blue')
            parts['cmedians'].set_color('green')
            # parts['cmaxes'].set_color('blue')
            # parts['cmins'].set_color('blue')
            parts['cbars'].set_color('gray')

            for k, y in enumerate(random.sample(adv_output_list, 8)):
                plt.plot(x, y[0], alpha=0.9, label=f'AE-{k + 1} ({y[1]}\u2192{y[2]})')

            plt.xticks(np.arange(1, len(x) + 1, 2))
            plt.subplots_adjust(left=0.10,
                                right=0.98,
                                bottom=0.12,
                                top=0.98)
            # plt.ylim(0, 10)
            # plt.tight_layout()
            plt.legend(loc="upper right", prop={'size': 12})
            plt.tick_params(labelsize=12)
            plt.xlabel('Index of Neurons', fontsize=12)
            plt.ylabel('Output Values of Dense(40)', fontsize=12)
            plt.show()

    return


def plot_similarity_by_fault_type(test_sim, ori_labels, targets, number_of_classes):
    test_sim = test_sim.transpose()
    test_sim_classified = [[[] for _ in range(number_of_classes)] for _ in range(number_of_classes)]
    for cs, ori, tar in zip(test_sim, ori_labels, targets):
        test_sim_classified[ori][tar].append(cs)

    for i in range(number_of_classes):
        for j in range(number_of_classes):
            if test_sim_classified[i][j]:
                y = np.array(test_sim_classified[i][j])
                y_cleaned = np.nan_to_num(y, nan=0)
                x = [k for k in range(10)]
                print(f'Label-{i}, Target-{j}, Number of Examples: {len(y)}.')

                plt.figure(figsize=(8, 4))

                parts1 = plt.violinplot(y_cleaned, positions=x,
                                        showmeans=True,
                                        showmedians=True,
                                        showextrema=True)
                for pc in parts1['bodies']:
                    pc.set_facecolor('skyblue')
                    pc.set_edgecolor('gray')
                    pc.set_alpha(1.0)
                parts1['cmeans'].set_color('red')
                parts1['cmedians'].set_color('green')
                # parts1['cmaxes'].set_color('blue')
                # parts1['cmins'].set_color('blue')
                parts1['cbars'].set_color('gray')

                # plt.title(f'Clean Label: {i}, Adversarial Target: {j}', fontsize=12)
                plt.xticks(np.arange(0, 10, 1))
                plt.subplots_adjust(left=0.10,
                                    right=0.98,
                                    bottom=0.12,
                                    top=0.98)
                # plt.ylim(0, 10)
                # plt.tight_layout()
                # plt.legend(loc="upper right", prop={'size': 12})
                plt.tick_params(labelsize=12)
                plt.xlabel('Classes', fontsize=12)
                plt.ylabel('Similarity', fontsize=12)
                plt.hlines(y=0, xmin=0, xmax=x[-1], color='red', linestyle='--', alpha=0.5)
                # plt.savefig(f'fmnist-test-rs-distance-{i}-{j}.png')
                # plt.close()
                plt.show()

    return



if __name__ == "__main__":
    model_dir = './models/lenet_fmnist/tf_model.h5'
    layer = -4

    train_data, train_label, test_data, test_label = load_fmnist()
    model = tf.keras.models.load_model(model_dir)

    train_prediction = model.predict(train_data)
    train_predict_label = np.argmax(train_prediction, axis=1)
    train_label = np.argmax(train_label, axis=1)
    correct_index = train_predict_label == train_label
    train_correct_data = train_data[correct_index]
    train_correct_label = train_label[correct_index]

    test_prediction = model.predict(test_data)
    test_predict_label = np.argmax(test_prediction, axis=1)
    test_label = np.argmax(test_label, axis=1)
    correct_index = test_predict_label == test_label
    test_correct_data = test_data[correct_index]
    test_correct_label = test_label[correct_index]
    wrong_index = test_predict_label != test_label
    test_wrong_data = test_data[wrong_index]
    test_wrong_targets = test_predict_label[wrong_index]
    test_wrong_clean_label = test_label[wrong_index]
    # adv_data = test_correct_data
    # adv_targets = test_correct_label
    # adv_clean_label = test_correct_label
    # adv_data = test_wrong_data
    # adv_targets = test_wrong_targets
    # adv_clean_label = test_wrong_clean_label

    adv_pa_data = np.load("./data/fmnist/adversarial/pixelattack_adv_data.npy")
    adv_pa_targets = np.load("./data/fmnist/adversarial/pixelattack_adv_targets.npy")
    adv_pa_clean_label = np.load("./data/fmnist/adversarial/pixelattack_clean_labels.npy")
    adv_data = adv_pa_data
    adv_targets = adv_pa_targets
    adv_clean_label = adv_pa_clean_label

    # adv_jsma_data = np.load("./data/fmnist/adversarial/jsma_adv_data.npy")
    # adv_jsma_targets = np.load("./data/fmnist/adversarial/jsma_adv_targets.npy")
    # adv_jsma_clen_label = np.load("./data/fmnist/adversarial/jsma_clean_labels.npy")
    # adv_data = adv_jsma_data
    # adv_targets = adv_jsma_targets
    # adv_clean_label = adv_jsma_clen_label

    # adv_ead_data = np.load("./data/fmnist/adversarial/ead_adv_data.npy")
    # adv_ead_targets = np.load("./data/fmnist/adversarial/ead_adv_targets.npy")
    # adv_ead_clen_label = np.load("./data/fmnist/adversarial/ead_clean_labels.npy")
    # adv_data = adv_ead_data
    # adv_targets = adv_ead_targets
    # adv_clean_label = adv_ead_clen_label

    # adv_pa_data = np.load("./data/fmnist/adversarial/pixelattack_adv_data.npy")
    # adv_pa_targets = np.load("./data/fmnist/adversarial/pixelattack_adv_targets.npy")
    # adv_pa_clean_label = np.load("./data/fmnist/adversarial/pixelattack_clean_labels.npy")
    # adv_jsma_data = np.load("./data/fmnist/adversarial/jsma_adv_data.npy")
    # adv_jsma_targets = np.load("./data/fmnist/adversarial/jsma_adv_targets.npy")
    # adv_jsma_clen_label = np.load("./data/fmnist/adversarial/jsma_clean_labels.npy")
    # adv_ead_data = np.load("./data/fmnist/adversarial/ead_adv_data.npy")
    # adv_ead_targets = np.load("./data/fmnist/adversarial/ead_adv_targets.npy")
    # adv_ead_clen_label = np.load("./data/fmnist/adversarial/ead_clean_labels.npy")
    # adv_cw_data = np.load("./data/fmnist/adversarial/cw_l2_adv_data.npy")
    # adv_cw_targets = np.load("./data/fmnist/adversarial/cw_l2_adv_targets.npy")
    # adv_cw_clen_label = np.load("./data/fmnist/adversarial/cw_l2_clean_labels.npy")
    # adv_sa_data = np.load("./data/fmnist/adversarial/squareattack_linf_adv_data.npy")
    # adv_sa_targets = np.load("./data/fmnist/adversarial/squareattack_linf_adv_targets.npy")
    # adv_sa_clen_label = np.load("./data/fmnist/adversarial/squareattack_linf_clean_labels.npy")
    # adv_data = np.concatenate([adv_pa_data, adv_jsma_data, adv_ead_data, adv_cw_data, adv_sa_data])
    # adv_targets = np.concatenate([adv_pa_targets, adv_jsma_targets, adv_ead_targets, adv_cw_targets, adv_sa_targets])
    # adv_clean_label = np.concatenate([adv_pa_clean_label, adv_jsma_clen_label, adv_ead_clen_label,
    #                                   adv_cw_clen_label, adv_sa_clen_label])

    # train_correct_hidden_output = get_neural_value(model_dir, layer, train_correct_data)
    # train_correct_hidden_output = softmax(train_correct_hidden_output / 2, axis=1)
    # adv_hidden_output = get_neural_value(model_dir, layer, adv_data)
    # adv_hidden_output = softmax((adv_hidden_output + 1) / 0.5, axis=1)
    # plot_output_by_fault_type(adv_hidden_output, adv_clean_label, adv_targets,
    #                           train_correct_hidden_output, train_correct_label, 10)
    # plot_output_by_fault_reason(adv_hidden_output, adv_clean_label, adv_targets,
    #                             train_correct_hidden_output, train_correct_label, 10)

    # train_rs = get_relative_selectivity_for_single_class(model_dir, layer, train_correct_data, train_correct_label,
    #                                                      train_correct_data, train_correct_label, 10)
    # train_rs = softmax(train_rs / 2, axis=1)
    # test_rs = get_relative_selectivity_for_single_class(model_dir, layer, train_correct_data, train_correct_label,
    #                                                     test_correct_data, test_correct_label, 10)
    # adv_rs_targets = get_relative_selectivity_for_single_class(model_dir, layer, train_correct_data, train_correct_label,
    #                                                            adv_data, adv_targets, 10)
    # adv_rs_targets = softmax(adv_rs_targets / 0.5, axis=1)
    # plot_output_by_fault_type(adv_rs_targets, adv_clean_label, adv_targets,
    #                           train_rs, train_correct_label, 10)
    # adv_rs_clean_label = get_relative_selectivity_for_single_class(model_dir, layer,
    #                                                                train_correct_data, train_correct_label,
    #                                                                adv_data, adv_clean_label, 10)
    # adv_rs_clean_label = softmax((adv_rs_clean_label + 1) / 0.5, axis=1)
    # plot_output_by_fault_type(adv_rs_clean_label, adv_clean_label, adv_targets,
    #                           train_rs, train_correct_label, 10)

    # unrelative_label = [8 for _ in range(len(adv_targets))]
    # adv_rs_unrelative_label = get_relative_selectivity_for_single_class(model_dir, layer, train_correct_data,
    #                                                                     train_correct_label, adv_data, unrelative_label, 10)
    # adv_rs_unrelative_label = softmax((adv_rs_unrelative_label + 1) / 0.5, axis=1)
    # plot_output_by_fault_type(adv_rs_unrelative_label, adv_clean_label, adv_targets,
    #                           train_rs, train_correct_label, 10)

    ############################################################################################
    # train_correct_hidden_output = get_neural_value(model_dir, layer, train_correct_data)
    # test_correct_hidden_output = get_neural_value(model_dir, layer, test_correct_data)
    # similarity_score = calculate_similarity(train_correct_hidden_output, train_correct_label,
    #                                         [test_correct_hidden_output for _ in range(10)], 10)
    # plot_similarity_by_fault_type(similarity_score, test_correct_label, test_correct_label, 10)

    # train_rs = get_relative_selectivity_for_single_class(model_dir, layer, train_correct_data, train_correct_label,
    #                                                      train_correct_data, train_correct_label, 10)
    # test_rs_all_classes = get_relative_selectivity_for_all_classes(model_dir, layer,
    #                                                                train_correct_data, train_correct_label,
    #                                                                test_correct_data, 10)
    # similarity_score = calculate_similarity(train_rs, train_correct_label, test_rs_all_classes, 10)
    # plot_similarity_by_fault_type(similarity_score, test_correct_label, test_correct_label, 10)

    # train_correct_hidden_output = get_neural_value(model_dir, layer, train_correct_data)
    # adv_hidden_output = get_neural_value(model_dir, layer, adv_data)
    # similarity_score = calculate_rs_distance(train_correct_hidden_output, train_correct_label,
    #                                          [adv_hidden_output for _ in range(10)], 10)
    # plot_similarity_by_fault_type(similarity_score, adv_clean_label, adv_targets, 10)

    select_index1 = adv_clean_label == 1
    select_index2 = adv_targets == 3
    index3 = select_index1 & select_index2
    select_adv_data = adv_data[index3][:]

    train_rs = get_relative_selectivity_for_single_class(model_dir, layer, train_correct_data, train_correct_label,
                                                         train_correct_data, train_correct_label, 10)
    # base_ReAD_list, adaptive_num_list = get_base_ReAD_for_class(train_rs, train_correct_label, 10)
    # train_rs = softmax(train_rs / 2, axis=1)
    adv_rs_all_classes = get_relative_selectivity_for_all_classes(model_dir, layer,
                                                                  train_correct_data, train_correct_label,
                                                                  adv_data, 10)
    # distance_score = calculate_ReAD_distance(train_rs, train_correct_label, adv_rs_all_classes, 10)
    # plot_similarity_by_fault_type(distance_score, adv_clean_label, adv_targets, 10)

    similarity_score = calculate_rs_distance(train_rs, train_correct_label, adv_rs_all_classes, 10)
    plot_similarity_by_fault_type(similarity_score, adv_clean_label, adv_targets, 10)
    # plot_similarity_by_fault_type(similarity_score, [1, 1, 1, 1], [3, 3, 3, 3], 10)




    ############################################################################################
    # adv_data = np.load("./data/fmnist/adversarial/jsma_adv_data.npy")
    # adv_targets = np.load("./data/fmnist/adversarial/jsma_adv_targets.npy")
    # adv_clean_label = np.load("./data/fmnist/adversarial/jsma_clean_labels.npy")
    #
    # model = tf.keras.models.load_model(model_dir)
    # print('Evaluate on adv_targets:')
    # model.evaluate(adv_data, tf.one_hot(adv_targets, 10), verbose=2)
    # print('Evaluate on adv_clean_label:')
    # model.evaluate(adv_data, tf.one_hot(adv_clean_label, 10), verbose=2)

    # count = 0
    # del_index = []
    # for d, t, l in tqdm(zip(adv_data, adv_targets, adv_clean_label), total=len(adv_data)):
    #     d = np.expand_dims(d, axis=0)
    #     prediction = model.predict(d)[0]
    #     label = np.argmax(prediction)
    #     confidence = prediction[label]
    #     if label != t:
    #         print(count, ":", confidence, label, t, l)
    #         del_index.append(count)
    #     count += 1
    #
    # adv_data = np.delete(adv_data, del_index, axis=0)
    # adv_targets = np.delete(adv_targets, del_index, axis=0)
    # adv_clean_label = np.delete(adv_clean_label, del_index, axis=0)
    # print('Evaluate on adv_targets:')
    # model.evaluate(adv_data, tf.one_hot(adv_targets, 10), verbose=2)
    # print('Evaluate on adv_clean_label:')
    # model.evaluate(adv_data, tf.one_hot(adv_clean_label, 10), verbose=2)
    # np.save("./data/fmnist/adversarial/jsma_adv_data2.npy", adv_data)
    # np.save("./data/fmnist/adversarial/jsma_adv_targets2.npy", adv_targets)
    # np.save("./data/fmnist/adversarial/jsma_clean_labels2.npy", adv_clean_label)


