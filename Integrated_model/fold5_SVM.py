import warnings
import os
import numpy as np
from sklearn.svm import SVC
from zzd.utils.assess import multi_scores
import pickle

# 1. parameters
output_dir = f"out/preds/fold5_SVM"  # 设置输出文件夹
output_model_dir = f"out/model_state/fold5_SVM"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_model_dir, exist_ok=True)

# 2. load data
# 2.1 load C1 C2 and C3 file #加载4个测试集
fold5_train_files = [f'train_dataset/fold5_train_pred__sum_{i}.txt' for i in range(1, 6)]
fold5_test_files = [f'sum/fold5_test_pred_sum_{i}.txt' for i in range(1, 6)]

# 2.2 load x_pred
fold5_test_scores = []

for foldn in range(1, 6):
    print(f"\nfold{foldn}: load file => ", end="")
    fold5_train = np.genfromtxt(fold5_train_files[foldn - 1], str)
    fold5_test = np.genfromtxt(fold5_test_files[foldn - 1], str)

    print("encode file =>", end="")
    X_c1_train, y_c1_train = fold5_train[:, :2], fold5_train[:, 2].astype(np.float32)
    X_c1_test, y_c1_test = fold5_test[:, :2], fold5_test[:, 2].astype(np.float32)

    x_c1_train = fold5_train[:, 3:].astype(np.float32)
    x_c1_test = fold5_test[:, 3:].astype(np.float32)

    # 参数搜索
    from sklearn.model_selection import RandomizedSearchCV


    def search_para():
        print("search_paramets==>", end="")
        model = SVC()
        params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': [0, 0.01, 0.1, 1],
            'random_state': [0],
            'kernel': ['linear'],
            'probability': [True],
            'verbose': [0],
        }

        rs_model = RandomizedSearchCV(model, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5,
                                      verbose=1)
        rs_model.fit(x_c1_train, y_c1_train)
        print(rs_model.best_estimator_)
        return rs_model.best_estimator_


    def small_search_para():
        print("search_paramets==>", end="")
        model = SVC()
        params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': [0.01, 0.1],
            'random_state': [0],
            'kernel': ['rbf'],
            'probability': [True],
            'verbose': [1],
        }
        rs_model = RandomizedSearchCV(model, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5,
                                      verbose=1)
        rs_model.fit(x_c1_train, y_c1_train)
        print(rs_model.best_estimator_)
        return rs_model.best_estimator_


    # 训练模型
    print("training==>", end="")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # model = search_para()
    # model = small_search_para()

    model = SVC(C=1, gamma=0.01, kernel='rbf', class_weight='balanced', probability=True, random_state=0)

    # 保存模型的最佳参数
    with open(f"{output_model_dir}/10folds_C1223_best_params_{foldn}.txt", "w") as f:
        for key, value in model.get_params().items():
            f.write(f"{key}: {value}\n")
    # 训练
    model.fit(x_c1_train, y_c1_train)

    # 预测数据
    print("predicting..")
    y_c1_test_pred = model.predict_proba(x_c1_test)[:, 1]

    c1_test_score = multi_scores(y_c1_test, y_c1_test_pred, show=True, threshold=0.5)

    fold5_test_scores.append(c1_test_score)

    # 保存结果
    # save pred result
    with open(f"{output_dir}/fold5_test_pred_{foldn}.txt", "w") as f:
        for line in np.hstack([X_c1_test, y_c1_test.reshape(-1, 1), y_c1_test_pred.reshape(-1, 1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    # save pred score
    with open(f"{output_dir}/fold5_test_score_{foldn}.txt", "w") as f:
        f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
        f.write("\t".join([str(i) for i in c1_test_score]))

    # save model
    model_file_name = f"{output_model_dir}/10folds_fold5_foldn_{foldn}.pkl"
    with open(model_file_name, "wb") as f:
        pickle.dump(model, f)

print("fold5 average")
c1_test_scores = np.array(fold5_test_scores)
fmat = [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]
with open(f"{output_dir}/c1_test_average_score.txt", 'w') as f:
    line1 = f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join(
        [f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c1_test_scores.mean(0), c1_test_scores.std(0))])
    print(line1, end="")
    print('\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c1_test_scores.mean(0), c1_test_scores.std(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

print("-----------------------------------")
