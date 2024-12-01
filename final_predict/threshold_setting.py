import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

# 加载数据
file = 'data/train_dataset.txt'
data = np.genfromtxt(file, str)
X, y = data[:, :2], data[:, 2].astype(np.float32)
x = data[:, 3:].astype(np.float32)

# 设置五折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

optimal_thresholds = []

for train_index, test_index in skf.split(x, y):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    model = RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=100, max_features='sqrt', random_state=0)
    model.fit(X_train, y_train)

    # 获取预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算 FPR 和 TPR
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # 找到假正率为 0.01 对应的阈值
    threshold_idx = np.where(fpr <= 0.005)[0]
    if len(threshold_idx) > 0:  # 确保有满足条件的阈值
        optimal_threshold = thresholds[threshold_idx[-1]]  # 选择最后一个满足条件的阈值
        optimal_thresholds.append(optimal_threshold)

# 打印每一折的最佳阈值
for i, threshold in enumerate(optimal_thresholds):
    print(f"Fold {i + 1}: Optimal Threshold = {threshold:.4f}")

# 取平均值
mean_threshold = np.mean(optimal_thresholds)
# 取中位数
median_threshold = np.median(optimal_thresholds)

print(f"Mean Threshold: {mean_threshold:.4f}")
print(f"Median Threshold: {median_threshold:.4f}")
# Fold 1: Optimal Threshold = 0.5229
# Fold 2: Optimal Threshold = 0.5190
# Fold 3: Optimal Threshold = 0.5429
# Fold 4: Optimal Threshold = 0.5517
# Fold 5: Optimal Threshold = 0.5264
# Mean Threshold: 0.5326
# Median Threshold: 0.5264
