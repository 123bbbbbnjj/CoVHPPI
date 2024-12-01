import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 确保目录存在
if not os.path.exists("Fold5"):
    os.makedirs("Fold5")


# 定义一个函数用于生成五折交叉验证数据集并保存
def generate_five_fold_datasets_and_save_counts(original_data):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1

    for train_index, test_index in skf.split(original_data, original_data['label']):
        print(f"generate fold{fold}................")

        train_set = original_data.iloc[train_index]
        test_set = original_data.iloc[test_index]

        # 保存每个数据集到文件
        train_set.to_csv(f"Fold5/Train_fold{fold}.txt", sep="\t", index=False, header=False)
        test_set.to_csv(f"Fold5/Test_fold{fold}.txt", sep="\t", index=False, header=False)

        fold += 1

    print("\nOK...")


if __name__ == '__main__':
    data = pd.read_csv("all_pos_neg.txt", names=["V_id", "H_id", "label"], sep="\t")
    # 调用函数生成数据集并保存大小
    generate_five_fold_datasets_and_save_counts(data)
