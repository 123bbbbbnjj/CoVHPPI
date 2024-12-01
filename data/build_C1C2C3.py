import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 确保目录存在
if not os.path.exists("C1C2C3"):
    os.makedirs("C1C2C3")


# 定义一个函数用于生成 独立数据集、C1、C2 和 C3 数据集以及保存数据集大小
def generate_datasets_and_save_counts(original_data):
    # 提取 v 蛋白质和 h 的列表
    v_ids = np.unique(original_data['V_id'])
    h_ids = np.unique(original_data['H_id'])

    for i in range(10):
        print(f"生成fold{i + 1}................")

        c1_positive_ratio = 0
        c2h_positive_ratio = 0
        c2v_positive_ratio = 0
        c3_positive_ratio = 0
        n = 0

        # 循环直到生成的数据集中的正负样本比例都在0.08到0.12之间
        while not (0.08 <= c1_positive_ratio <= 0.12 and 0.08 <= c2h_positive_ratio <= 0.12 and
                   0.08 <= c2v_positive_ratio <= 0.12 and 0.08 <= c3_positive_ratio <= 0.12):
            # 从 v 和 h 中随机选择 80% 作为 C1
            c1_v = np.random.choice(v_ids, size=int(0.7 * len(v_ids)), replace=False)
            c1_h = np.random.choice(h_ids, size=int(0.7 * len(h_ids)), replace=False)

            # 构建 C1 数据集
            c1_set = original_data[(original_data['V_id'].isin(c1_v)) & (original_data['H_id'].isin(c1_h))]
            c1_positive_ratio = len(c1_set[c1_set['label'] == 1]) / len(c1_set)

            # 构建 C2 和 C3 数据集
            c2h_test_set = original_data[
                (original_data['V_id'].isin(c1_v)) & (~original_data['H_id'].isin(c1_h))].sample(frac=1)
            c2v_test_set = original_data[
                (~original_data['V_id'].isin(c1_v)) & (original_data['H_id'].isin(c1_h))].sample(frac=1)
            c3_test_set = original_data[
                (~original_data['V_id'].isin(c1_v)) & (~original_data['H_id'].isin(c1_h))].sample(frac=1)
            c2h_positive_ratio = len(c2h_test_set[c2h_test_set['label'] == 1]) / len(c2h_test_set)
            c2v_positive_ratio = len(c2v_test_set[c2v_test_set['label'] == 1]) / len(c2v_test_set)
            c3_positive_ratio = len(c3_test_set[c3_test_set['label'] == 1]) / len(c3_test_set)

            n = n + 1
            print(n)

        # 划分C1训练集和测试集
        c_train_set, c1_test_set = train_test_split(c1_set, test_size=0.1, stratify=c1_set['label'])

        # 保存每个数据集文件本身
        c1_set.to_csv("C1C2C3/C1_fold{}.txt".format(i + 1), sep="\t", index=False, header=False)
        c_train_set.to_csv("C1C2C3/C1_train_fold{}.txt".format(i + 1), sep="\t", index=False,
                           header=False)
        c1_test_set.to_csv("C1C2C3/C1_test_fold{}.txt".format(i + 1), sep="\t", index=False,
                           header=False)
        c2v_test_set.to_csv("C1C2C3/C2v_fold{}.txt".format(i + 1), sep="\t", index=False,
                            header=False)
        c2h_test_set.to_csv("C1C2C3/C2h_fold{}.txt".format(i + 1), sep="\t", index=False,
                            header=False)
        c3_test_set.to_csv("C1C2C3/C3_fold{}.txt".format(i + 1), sep="\t", index=False, header=False)

    print("\nOK...")


if __name__ == '__main__':
    data = pd.read_csv("all_pos_neg.txt", names=["V_id", "H_id", "label"], sep="\t")
    # 调用函数生成数据集并保存大小
    generate_datasets_and_save_counts(data)
