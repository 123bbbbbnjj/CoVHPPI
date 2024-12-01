"""
python 10folds_NN_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean
"""
import random
import sys
import os
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from zzd.utils.assess import multi_scores
import pickle
from sklearn.model_selection import train_test_split
from HV_feature import Features_C123 as Features

import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

# 设置随机种子
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def dense_model(input_shape):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(keras.layers.Dense(240, activation="relu", name="layer1"))
    model.add(keras.layers.Dense(147, activation="relu", name="layer2"))
    model.add(keras.layers.Dense(110, activation="relu", name="layer3"))
    model.add(keras.layers.Dense(520, activation="relu", name="layer4"))
    #model.add(keras.layers.Dense(32, activation="relu", name="layer5"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1,  activation="sigmoid", name="layer6"))
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'])
    return model

# 1. parameters
print(sys.argv)
info_list = sys.argv[1:] if len(sys.argv) > 1 else None
output_dir = f"out/preds/10folds_C1223_NN_" + "_".join(info_list)  # 设置输出文件夹
output_model_dir = f"out/model_state"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_model_dir, exist_ok=True)

# 2. load data
# 2.1 load C1 C2 and C3 file #加载4个测试集
c1_train_files = [f'../../data/C1C2C3/C1_train_fold{i}.txt' for i in range(1, 11)]
c1_test_files = [f'../../data/C1C2C3/C1_test_fold{i}.txt' for i in range(1, 11)]
c2h_files = [f'../../data/C1C2C3/C2h_fold{i}.txt' for i in range(1, 11)]
c2v_files = [f"../../data/C1C2C3/C2v_fold{i}.txt" for i in range(1, 11)]
c3_files = [f'../../data/C1C2C3/C3_fold{i}.txt' for i in range(1, 11)]

# 2.2 load features
features = Features(info=info_list)

c1_test_scores = []
c2h_scores = []
c2v_scores = []
c3_scores = []

for foldn in range(1, 11):
    print(f"fold{foldn}: load file => ", end="")
    warnings.filterwarnings("ignore")
    sys.stdout.flush()
    c1_train = np.genfromtxt(c1_train_files[foldn - 1], str)
    c1_test = np.genfromtxt(c1_test_files[foldn - 1], str)
    c2h = np.genfromtxt(c2h_files[foldn - 1], str)
    c2v = np.genfromtxt(c2v_files[foldn - 1], str)
    c3 = np.genfromtxt(c3_files[foldn - 1], str)

    print("encode file =>", end="")
    warnings.filterwarnings("ignore")
    sys.stdout.flush()  # 对蛋白进行编码
    X_c1_train, y_c1_train = c1_train[:, :2], c1_train[:, 2].astype(np.float32)
    X_c1_test, y_c1_test = c1_test[:, :2], c1_test[:, 2].astype(np.float32)
    X_c2h, y_c2h = c2h[:, :2], c2h[:, 2].astype(np.float32)
    X_c2v, y_c2v = c2v[:, :2], c2v[:, 2].astype(np.float32)
    X_c3, y_c3 = c3[:, :2], c3[:, 2].astype(np.float32)

    x_c1_train = np.array([np.hstack([features.get(j, foldn) for j in i]) for i in X_c1_train])
    x_c1_test = np.array([np.hstack([features.get(j, foldn) for j in i]) for i in X_c1_test])
    x_c2h = np.array([np.hstack([features.get(j, foldn) for j in i]) for i in X_c2h])
    x_c2v = np.array([np.hstack([features.get(j, foldn) for j in i]) for i in X_c2v])
    x_c3 = np.array([np.hstack([features.get(j, foldn) for j in i]) for i in X_c3])

    # 参数搜索
    from sklearn.model_selection import RandomizedSearchCV


    def search_para():
        model = MLPClassifier()
        params = {
            'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }

        rs_model = RandomizedSearchCV(model, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5,
                                      verbose=3)

        rs_model.fit(x_c1_train, y_c1_train)

        print(rs_model.best_estimator_)
        return rs_model.best_estimator_

    # 训练模型
    print("training==>", end="")
    warnings.filterwarnings("ignore")
    sys.stdout.flush()

    # model = search_para()
    model = dense_model((8034,))

    # # 保存模型的最佳参数
    # with open(f"{output_model_dir}/10folds_C1223_NN_" + "_".join(info_list)+f"_best_params_{foldn}.txt", "w") as f:
    #     for key, value in model.get_params().items():
    #         f.write(f"{key}: {value}\n")
    # 训练
    model.fit(x_c1_train,y_c1_train, epochs=30,batch_size=64,
            verbose=False,
            shuffle=True)

    # 预测数据
    print("predicting..")
    y_c1_test_pred = model.predict(x_c1_test)
    y_c2h_pred = model.predict(x_c2h)
    y_c2v_pred = model.predict(x_c2v)
    y_c3_pred = model.predict(x_c3)

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

    # # save model
    # model_file_name = f"{output_model_dir}/10folds_C1223_MLP_" + "_".join(info_list)+f"_foldn_{foldn}.pkl"
    # with open(model_file_name,"wb") as f:
    #    pickle.dump(model,f)

print("10 fold C1223 average")
c1_test_scores = np.array(c1_test_scores)
fmat = [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]
with open(f"{output_dir}/c1_test_average_score.txt", 'w') as f:
    line1 = f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join(
        [f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c1_test_scores.mean(0), c1_test_scores.std(0))])
    print(line1, end="")
    print('\t'.join([f'{a:.{_}f}' for (_, a) in zip(fmat, c1_test_scores.mean(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c2h_scores = np.array(c2h_scores)
with open(f"{output_dir}/c2h_average_score.txt", 'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c2h_scores.mean(0), c2h_scores.std(0))])
    # print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_, a) in zip(fmat, c2h_scores.mean(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c2v_scores = np.array(c2v_scores)
with open(f"{output_dir}/c2v_average_score.txt", 'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c2v_scores.mean(0), c2v_scores.std(0))])
    # print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_, a) in zip(fmat, c2v_scores.mean(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

c3_scores = np.array(c3_scores)
with open(f"{output_dir}/c3_average_score.txt", 'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, c3_scores.mean(0), c3_scores.std(0))])
    # print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_, a) in zip(fmat, c3_scores.mean(0))]))
    f.write(line1)
    f.write(line2)
    f.write("\n")

print("-----------------------------------")
