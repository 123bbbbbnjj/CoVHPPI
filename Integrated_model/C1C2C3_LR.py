import warnings
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from zzd.utils.assess import multi_scores
import pickle

# 1. parameters
output_dir = f"out/preds/LR"  # 设置输出文件夹
output_model_dir = f"out/model_state/LR"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_model_dir, exist_ok=True)

# 2. load data
# 2.1 load C1 C2 and C3 file #加载4个测试集
c1_train_files = [f'train_dataset/c1_train_pred__sum_{i}.txt' for i in range(1, 11)]
c2h_train_files = [f'train_dataset/c2h_train_pred__sum_{i}.txt' for i in range(1, 11)]
c2v_train_files = [f'train_dataset/c2v_train_pred__sum_{i}.txt' for i in range(1, 11)]
c3_train_files = [f'train_dataset/c3_train_pred__sum_{i}.txt' for i in range(1, 11)]
c1_test_files = [f'sum/c1_test_pred_sum_{i}.txt' for i in range(1, 11)]
c2h_files = [f'sum/c2h_test_pred_sum_{i}.txt' for i in range(1, 11)]
c2v_files = [f"sum/c2v_test_pred_sum_{i}.txt" for i in range(1, 11)]
c3_files = [f'sum/c3_test_pred_sum_{i}.txt' for i in range(1, 11)]

# 2.2 load x_pred
c1_test_scores = []
c2h_scores = []
c2v_scores = []
c3_scores = []

for foldn in range(1, 11):
    print(f"\nfold{foldn}: load file => ", end="")
    c1_train = np.genfromtxt(c1_train_files[foldn - 1], str)
    c2h_train = np.genfromtxt(c2h_train_files[foldn - 1], str)
    c2v_train = np.genfromtxt(c2v_train_files[foldn - 1], str)
    c3_train = np.genfromtxt(c3_train_files[foldn - 1], str)
    c1_test = np.genfromtxt(c1_test_files[foldn - 1], str)
    c2h = np.genfromtxt(c2h_files[foldn - 1], str)
    c2v = np.genfromtxt(c2v_files[foldn - 1], str)
    c3 = np.genfromtxt(c3_files[foldn - 1], str)

    print("encode file =>", end="")
    X_c1_train, y_c1_train = c1_train[:, :2], c1_train[:, 2].astype(np.float32)
    X_c2h_train, y_c2h_train = c2h_train[:, :2], c2h_train[:, 2].astype(np.float32)
    X_c2v_train, y_c2v_train = c2v_train[:, :2], c2v_train[:, 2].astype(np.float32)
    X_c3_train, y_c3_train = c3_train[:, :2], c3_train[:, 2].astype(np.float32)
    X_c1_test, y_c1_test = c1_test[:, :2], c1_test[:, 2].astype(np.float32)
    X_c2h, y_c2h = c2h[:, :2], c2h[:, 2].astype(np.float32)
    X_c2v, y_c2v = c2v[:, :2], c2v[:, 2].astype(np.float32)
    X_c3, y_c3 = c3[:, :2], c3[:, 2].astype(np.float32)

    x_c1_train = c1_train[:, 3:].astype(np.float32)
    x_c2h_train = c2h_train[:, 3:].astype(np.float32)
    x_c2v_train = c2v_train[:, 3:].astype(np.float32)
    x_c3_train = c3_train[:, 3:].astype(np.float32)
    x_c1_test = c1_test[:, 3:].astype(np.float32)
    x_c2h = c2h[:, 3:].astype(np.float32)
    x_c2v = c2v[:, 3:].astype(np.float32)
    x_c3 = c3[:, 3:].astype(np.float32)

    # 训练模型
    print("training==>", end="")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # 定义模型的超参数
    model_params = {
        'C': 0.01,
        'penalty': 'l1',
        'solver': 'liblinear',
        'class_weight': 'balanced',
        'random_state': 0
    }

    model = LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced', random_state=0)

    # 保存模型的最佳参数
    with open(f"{output_model_dir}/10folds_C1223_best_params_{foldn}.txt", "w") as f:
        for key, value in model.get_params().items():
            f.write(f"{key}: {value}\n")

    model_c1 = LogisticRegression(**model_params)
    model_c2h = LogisticRegression(**model_params)
    model_c2v = LogisticRegression(**model_params)
    model_c3 = LogisticRegression(**model_params)

    # 训练
    model_c1.fit(x_c1_train, y_c1_train)
    model_c2h.fit(x_c2h_train, y_c2h_train)
    model_c2v.fit(x_c2v_train, y_c2v_train)
    model_c3.fit(x_c3_train, y_c3_train)

    # 预测数据
    print("predicting..")
    y_c1_test_pred = model_c1.predict_proba(x_c1_test)[:, 1]
    y_c2h_pred = model_c2h.predict_proba(x_c2h)[:, 1]
    y_c2v_pred = model_c2v.predict_proba(x_c2v)[:, 1]
    y_c3_pred = model_c3.predict_proba(x_c3)[:, 1]

    c1_test_score = multi_scores(y_c1_test, y_c1_test_pred, show=True, threshold=0.5)
    c2h_score = multi_scores(y_c2h, y_c2h_pred, show=True, show_index=False, threshold=0.5)
    c2v_score = multi_scores(y_c2v, y_c2v_pred, show=True, show_index=False, threshold=0.5)
    c3_score = multi_scores(y_c3, y_c3_pred, show=True, show_index=False, threshold=0.5)

    c1_test_scores.append(c1_test_score)
    c2h_scores.append(c2h_score)
    c2v_scores.append(c2v_score)
    c3_scores.append(c3_score)

    # 保存结果
    # save pred result
    with open(f"{output_dir}/c1_test_pred_{foldn}.txt", "w") as f:
        for line in np.hstack([X_c1_test, y_c1_test.reshape(-1, 1), y_c1_test_pred.reshape(-1, 1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c2h_pred_{foldn}.txt", "w") as f:
        for line in np.hstack([X_c2h, y_c2h.reshape(-1, 1), y_c2h_pred.reshape(-1, 1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c2v_pred_{foldn}.txt", "w") as f:
        for line in np.hstack([X_c2v, y_c2v.reshape(-1, 1), y_c2v_pred.reshape(-1, 1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c3_pred_{foldn}.txt", "w") as f:
        for line in np.hstack([X_c3, y_c3.reshape(-1, 1), y_c3_pred.reshape(-1, 1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    # save pred score
    with open(f"{output_dir}/c1_test_score_{foldn}.txt", "w") as f:
        f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
        f.write("\t".join([str(i) for i in c1_test_score]))

    with open(f"{output_dir}/c2h_score_{foldn}.txt", "w") as f:
        f.write(f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
        f.write("\t".join([str(i) for i in c2h_score]))

    with open(f"{output_dir}/c2v_score_{foldn}.txt", "w") as f:
        f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
        f.write("\t".join([str(i) for i in c2v_score]))

    with open(f"{output_dir}/c3_score_{foldn}.txt", "w") as f:
        f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
        f.write("\t".join([str(i) for i in c3_score]))

    # save model
    model_file_name = f"{output_model_dir}/10folds_C1223_foldn_{foldn}.pkl"
    with open(model_file_name, "wb") as f:
        pickle.dump(model, f)

print("10 fold C1223 average")
c1_test_scores = np.array(c1_test_scores)
fmat = [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]
with open(f"{output_dir}/c1_test_average_score.txt", 'w') as f:
    line1 = f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join(
        [f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c1_test_scores.mean(0), c1_test_scores.std(0))])
    print(line1, end="")
    # print('\t'.join([f'{a:.{_}f}' for (_, a) in zip(fmat, c1_test_scores.mean(0))]))
    print('\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c1_test_scores.mean(0), c1_test_scores.std(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c2h_scores = np.array(c2h_scores)
with open(f"{output_dir}/c2h_average_score.txt", 'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c2h_scores.mean(0), c2h_scores.std(0))])
    # print(line1)
    # print('\t'.join([f'{a:.{_}f}' for (_, a) in zip(fmat, c2h_scores.mean(0))]))
    print('\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c2h_scores.mean(0), c2h_scores.std(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c2v_scores = np.array(c2v_scores)
with open(f"{output_dir}/c2v_average_score.txt", 'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c2v_scores.mean(0), c2v_scores.std(0))])
    # print(line1)
    # print('\t'.join([f'{a:.{_}f}' for (_, a) in zip(fmat, c2v_scores.mean(0))]))
    print('\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c2v_scores.mean(0), c2v_scores.std(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c3_scores = np.array(c3_scores)
with open(f"{output_dir}/c3_average_score.txt", 'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c3_scores.mean(0), c3_scores.std(0))])
    # print(line1)
    # print('\t'.join([f'{a:.{_}f}' for (_, a) in zip(fmat, c3_scores.mean(0))]))
    print('\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c3_scores.mean(0), c3_scores.std(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

print("-----------------------------------")
