import math
import numpy as np
import pandas as pd
import os


def exponentPSSM(PSSM):
    PSSM = np.array(PSSM)
    seq_cn = np.shape(PSSM)[0]
    PSSM_exponent = [[0.0] * 20] * seq_cn
    for i in range(seq_cn):
        for j in range(20):
            PSSM_exponent[i][j] = math.exp(PSSM[i][j])
    PSSM_exponent = np.array(PSSM_exponent)
    return PSSM_exponent


def aac_pssm(input_matrix, exp=True):
    if exp == True:
        input_matrix = exponentPSSM(input_matrix)
    else:
        input_matrix = input_matrix
    seq_cn = float(np.shape(input_matrix)[0])
    aac_pssm_matrix = input_matrix.sum(axis=0)
    aac_pssm_vector = aac_pssm_matrix / seq_cn
    # vec = []
    # result = []
    # header = []
    # for f in range(20):
    #     header.append('aac_pssm.' + str(f))
    # result.append(header)
    # for v in aac_pssm_vector:
    #     vec.append(v)
    # result.append(vec)
    # return aac_pssm_vector, result
    return aac_pssm_vector


def bi_pssm(input_matrix, exp=True):
    if exp == True:
        input_matrix = exponentPSSM(input_matrix)
    else:
        input_matrix = input_matrix
    PSSM = input_matrix
    PSSM = np.array(PSSM)
    # header = []
    # for f in range(400):
    #     header.append('bi_pssm.' + str(f))
    # result = []
    # result.append(header)
    # vec = []
    bipssm = [[0.0] * 400] * (PSSM.shape[0] - 1)
    p = 0
    for i in range(20):
        for j in range(20):
            for h in range(PSSM.shape[0] - 1):
                bipssm[h][p] = PSSM[h][i] * PSSM[h + 1][j]
            p = p + 1
    vector = np.sum(bipssm, axis=0)
    # for v in vector:
    #     vec.append(v)
    # result.append(vec)
    # return vector, result
    return vector


def dpc_pssm(input_matrix):
    input_matrix = input_matrix.astype(float)
    matrix_final = np.zeros((20, 20))
    seq_cn = np.shape(input_matrix)[0]

    for i in range(20):
        for j in range(20):
            for k in range(seq_cn - 1):
                matrix_final[i][j] += (input_matrix[k][i] * input_matrix[k + 1][j])

    matrix_array = np.array(matrix_final)
    matrix_array = np.divide(matrix_array, seq_cn - 1)
    matrix_array_shp = np.shape(matrix_array)
    matrix_average = np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1],))

    return matrix_average


def rpssm(input_matrix):
    seq_cn = np.shape(input_matrix)[0]

    # Calculate RPSSM
    row = [0.0] * 110
    row1 = [0.0] * 100
    row2 = [0.0] * 10
    row = np.array(row)
    row1 = np.array(row1)
    row2 = np.array(row2)

    RPSSM = [[0.0] * 10] * seq_cn
    RPSSM = np.array(RPSSM)

    PSSM = input_matrix.astype(float)
    PSSM = np.array(PSSM)

    RPSSM[:, 0] = np.divide(np.sum(PSSM[:, [13, 17, 18]], axis=1), 3.0)
    RPSSM[:, 1] = np.divide(np.sum(PSSM[:, [10, 12]], axis=1), 2.0)
    RPSSM[:, 2] = np.divide(np.sum(PSSM[:, [9, 19]], axis=1), 2.0)
    RPSSM[:, 3] = np.divide(np.sum(PSSM[:, [0, 15, 16]], axis=1), 3.0)
    RPSSM[:, 4] = np.divide(np.sum(PSSM[:, [2, 8]], axis=1), 2.0)
    RPSSM[:, 5] = np.divide(np.sum(PSSM[:, [5, 6, 3]], axis=1), 3.0)
    RPSSM[:, 6] = np.divide(np.sum(PSSM[:, [1, 11]], axis=1), 2.0)
    RPSSM[:, 7] = PSSM[:, 4]
    RPSSM[:, 8] = PSSM[:, 7]
    RPSSM[:, 9] = PSSM[:, 14]

    mean_matrix = np.mean(RPSSM, axis=0)

    for j in range(10):
        for i in range(seq_cn):
            row2[j] += (RPSSM[i][j] - mean_matrix[j]) * (RPSSM[i][j] - mean_matrix[j])

    row2 = np.divide(row2, seq_cn)

    matrix_final = [[0.0] * 10] * 10
    matrix_final = np.array(matrix_final)

    for i in range(10):
        for j in range(10):
            for k in range(seq_cn - 1):
                matrix_final[i][j] += ((RPSSM[k][i] - RPSSM[k + 1][j]) * (RPSSM[k][i] - RPSSM[k + 1][j]) / 2.0)

    matrix_array = np.array(matrix_final)
    matrix_array = np.divide(matrix_array, seq_cn-1)
    matrix_array_shp = np.shape(matrix_array)
    row1 = (np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1],)))
    row = np.hstack((row1, row2))

    return row


if __name__ == "__main__":

    # 获取目标文件夹中所有.pssm文件
    file_folder = "pssm"
    pssm_files = [f for f in os.listdir(file_folder) if f.endswith('.pssm')]

    # 创建一个空的 DataFrame
    aac_pssm_result = pd.DataFrame()
    # bi_pssm_result = pd.DataFrame()
    dpc_pssm_result = pd.DataFrame()
    rpssm_result = pd.DataFrame()

    # 循环处理每个文件
    for pssm_file in pssm_files:
        # 读取PSSM文件并进行转换
        pssmmat = pd.read_table(os.path.join(file_folder, pssm_file), sep=" ").values

        vector1 = aac_pssm(pssmmat, exp=True)
        # vector2 = bi_pssm(pssmmat, exp=True)
        vector3 = dpc_pssm(pssmmat)
        vector4 = rpssm(pssmmat)

        # 提取文件名（不包括扩展名）
        filename_without_extension = pssm_file.split(".")[0]

        # 将转换后的结果添加到DataFrame中
        aac_pssm_result = pd.concat([aac_pssm_result, pd.DataFrame({filename_without_extension: vector1})], axis=1)
        # bi_pssm_result = pd.concat([bi_pssm_result, pd.DataFrame({filename_without_extension: vector2})], axis=1)
        dpc_pssm_result = pd.concat([dpc_pssm_result, pd.DataFrame({filename_without_extension: vector3})], axis=1)
        rpssm_result = pd.concat([rpssm_result, pd.DataFrame({filename_without_extension: vector4})], axis=1)

    # 保存结果到 v_and_h_AAC-PSSM.txt 文件
    aac_pssm_result.to_csv('v_and_h_AAC-PSSM.txt', sep='\t', header=True, index=False)
    # bi_pssm_result.to_csv('v_and_h_Bi-PSSM.txt', sep='\t', header=True, index=False)
    dpc_pssm_result.to_csv('v_and_h_DPC-PSSM.txt', sep='\t', header=True, index=False)
    rpssm_result.to_csv('v_and_h_RPSSM.txt', sep='\t', header=True, index=False)

    ## Test
    # pssmmat = pd.read_table("pssm/A2RTX5.pssm", sep=" ").values
    # vector = rpssm(pssmmat)
    # print(vector)
