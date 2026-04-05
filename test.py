# import matplotlib.pyplot as plt
# import numpy as np
#
# # 创建示例数据
# np.random.seed(42)
# data1 = np.random.normal(0, 1, 200)
# data2 = np.random.normal(5, 2, 200)
# data3 = np.random.normal(10, 1.5, 200)
# data_1 = [data1, data2, data3]
# data_2 = [data3, data1, data2]
#
# # 绘制自定义小提琴图
# plt.figure(figsize=(10, 6))
#
# # 使用violinplot函数
# parts1 = plt.violinplot(data_1,
#                        showmeans=True,  # 显示均值
#                        showmedians=True,  # 显示中位数
#                        showextrema=True)  # 显示极值
#
# # 自定义颜色
# for pc in parts1['bodies']:
#     pc.set_facecolor('lightblue')
#     pc.set_edgecolor('black')
#     pc.set_alpha(0.7)
#
# # 自定义其他元素
# parts1['cmeans'].set_color('red')
# # parts1['cmedians'].set_color('green')
# parts1['cmaxes'].set_color('blue')
# parts1['cmins'].set_color('blue')
# # parts1['cbars'].set_color('black')
#
#
# # 使用violinplot函数
# parts2 = plt.violinplot(data_2,
#                        showmeans=True,  # 显示均值
#                        showmedians=True,  # 显示中位数
#                        showextrema=True)  # 显示极值
#
# # 自定义颜色
# for pc in parts2['bodies']:
#     pc.set_facecolor('lightblue')
#     pc.set_edgecolor('black')
#     pc.set_alpha(0.7)
#
# # 自定义其他元素
# parts2['cmeans'].set_color('red')
# # parts2['cmedians'].set_color('green')
# parts2['cmaxes'].set_color('blue')
# parts2['cmins'].set_color('blue')
# # parts2['cbars'].set_color('black')
#
#
# # 添加标签和标题
# plt.title('自定义小提琴图', fontsize=14)
# plt.ylabel('数值', fontsize=12)
# plt.xlabel('数据组', fontsize=12)
# plt.xticks([1, 2, 3], ['组A', '组B', '组C'])
#
# # 添加网格
# plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.show()
import numpy as np
# import numpy as np
#
# print(np.sum([1, 2, 3]))

from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import spearmanr
# from scipy.stats import pearsonr
#
# a = [1, 2, 3]
#
# b = [-3, -2, -1]
#
# print(pearsonr(a, b))
# print(np.corrcoef(a, b))
# #
# correlation, p_value = spearmanr(np.array(a), np.array(b))
# print(f"斯皮尔曼相关系数: {correlation}")
# a = [1, 1, 2, 3, 4, 5, 6]
# ReA_indices = np.argsort(np.array(a))[-3:]
# ReD_indices = np.argsort(np.array(a))[:3]
# print(ReA_indices, ReD_indices)

from tensorflow.keras import layers
import tensorflow as tf
from tqdm import tqdm

model_dir = './models/resnet18_svhn/tf_model.h5'

adv_data = np.load("./data/svhn/adversarial/wassersteinattack_adv_data.npy")
adv_targets = np.load("./data/svhn/adversarial/wassersteinattack_adv_targets.npy")
adv_clean_label = np.load("./data/svhn/adversarial/wassersteinattack_clean_labels.npy")

model = tf.keras.models.load_model(model_dir)

# print('Evaluate on adv_targets:')
# model.evaluate(adv_data, tf.one_hot(adv_targets, 10), verbose=2)
# print('Evaluate on adv_clean_label:')
# model.evaluate(adv_data, tf.one_hot(adv_clean_label, 10), verbose=2)

count = 0
del_index = []
for d, t, l in tqdm(zip(adv_data, adv_targets, adv_clean_label), total=len(adv_data)):
    d = np.expand_dims(d, axis=0)
    prediction = model.predict(d)[0]
    label = np.argmax(prediction)
    confidence = prediction[label]
    if label != t:
        print(count, ":", confidence, label, t, l)
        del_index.append(count)
    count += 1

adv_data = np.delete(adv_data, del_index, axis=0)
adv_targets = np.delete(adv_targets, del_index, axis=0)
adv_clean_label = np.delete(adv_clean_label, del_index, axis=0)
print('Evaluate on adv_targets:')
model.evaluate(adv_data, tf.one_hot(adv_targets, 10), verbose=2)
print('Evaluate on adv_clean_label:')
model.evaluate(adv_data, tf.one_hot(adv_clean_label, 10), verbose=2)
np.save("./data/svhn/adversarial/wassersteinattack_adv_data2.npy", adv_data)
np.save("./data/svhn/adversarial/wassersteinattack_adv_targets2.npy", adv_targets)
np.save("./data/svhn/adversarial/wassersteinattack_clean_labels2.npy", adv_clean_label)

