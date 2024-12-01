import pandas as pd
import numpy as np
import pickle


def pickle_to_txt(pickle_file, txt_file):
    # 加载 pickle 文件
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # 将数据转换为 DataFrame
    df = pd.DataFrame.from_dict(data)
    print(df)

    # 将 DataFrame 保存为以制表符分隔的文本文件
    df.to_csv(txt_file, sep='\t', index=False)
    print(df.shape)
    print(df.head(5))
    return df


def txt_to_pickle(txt_file, pickle_file):
    # 读取txt文件并处理数据
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # 将处理后的数据保存到一个字典中
    data = {}
    num_cols = len(lines[0].split('\t'))  # 确定列数
    for col_idx in range(num_cols):
        key = None
        values = []
        for line in lines:
            parts = line.strip().split('\t')
            if key is None:
                key = parts[col_idx]  # 第一行作为键
            else:
                values.append(float(parts[col_idx]))  # 其余行作为值
        data[key] = np.array(values, dtype=np.float32)

    # 将该字典保存为pkl文件
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)


# 示例用法
if __name__ == "__main__":
    # pickle_to_txt("Human_NetNode2vec.pkl", "Human_NetNode2vec.txt")
    txt_to_pickle("Human_NetNode2vec.txt", "Human_NetNode2vec.pkl")
