import warnings
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from zzd.utils.assess import multi_scores
import pickle

# 1. parameters
output_dir = f"preds/"  # 设置输出文件夹
output_model_dir = f"model/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_model_dir, exist_ok=True)

# 2. load data
train_file = 'data/train_dataset.txt'
train_data = np.genfromtxt(train_file, str)
X_train, y_train = train_data[:, :2], train_data[:, 2].astype(np.float32)
x_train = train_data[:, 3:].astype(np.float32)

test_file = 'data/final_predict_dataset.txt'
test_data = np.genfromtxt(test_file, str)
X_test = test_data[:, :2]
x_test = test_data[:, 2:].astype(np.float32)

model = RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=100, max_features='sqrt',
                                   random_state=0)
model.fit(x_train, y_train)

# 保存模型
with open(f'{output_model_dir}/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training over!")

# 预测数据
print("Predicting..")
y_test_pred = model.predict_proba(x_test)[:, 1]

# 保存结果
# save pred result
with open(f"{output_dir}/predict_pred.txt", "w") as f:
    for line in np.hstack([X_test, x_test, y_test_pred.reshape(-1, 1)]):
        line = "\t".join(line) + "\n"
        f.write(line)

print("-----------------------------------")
print("OK")
