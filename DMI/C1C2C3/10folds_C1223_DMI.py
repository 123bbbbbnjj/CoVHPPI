import os
from typing import Any

import numpy as np
import pandas as pd
from zzd.utils.assess import multi_scores


class DMI:
    def __init__(self, test_df, blast_pair_file, scores_df):
        self.test_df = test_df
        self.blast_pair_file = blast_pair_file
        self.scores_df = scores_df

    def generate_id_groups(self) -> tuple[list[str], list[Any], list[Any]]:
        print("1. Generate id groups")
        data_ppi = self.test_df

        ppi_v = list(set(data_ppi[0]))
        ppi_h = list(set(data_ppi[1]))

        print("V: ", len(ppi_v))
        print("H: ", len(ppi_h))

        print("Processing ppi_ids...")
        ppi_ids = list(set(data_ppi[0] + '-' + data_ppi[1]))
        print("PPI: ", len(ppi_ids))

        print("Finish ppi_ids_list!\n")

        return ppi_ids, ppi_v, ppi_h

    def get_scores_dict(self, ppi_ids, ppi_v, ppi_h) -> dict:

        print("2. Get ppi dict...")

        v_df = pd.DataFrame({'Vid': ppi_v})
        h_df = pd.DataFrame({'Hid': ppi_h})
        ppi_df = pd.DataFrame({'PPIid': ppi_ids})
        ppi_df[['Vid', 'Hid']] = ppi_df['PPIid'].str.rsplit('-', n=1, expand=True)
        ppi_df.set_index('PPIid', inplace=True)

        print("Import select pairs...")
        blast_pair_df = pd.read_excel(self.blast_pair_file)

        print("Get  v & h dict_df...")

        def merge_protein_info(df, column_name):
            merge_df = pd.merge(df, blast_pair_df, left_on=column_name, right_on='Query_def', how='left')
            merge_df = merge_df.groupby(column_name)['Subject_def'].agg(list).reset_index()
            df = pd.merge(df, merge_df, on=column_name, how='left')
            df.rename(columns={'Subject_def': 'S' + column_name}, inplace=True)
            return df

        v_df = merge_protein_info(v_df, "Vid")
        h_df = merge_protein_info(h_df, "Hid")
        v_df.set_index('Vid', inplace=True)
        h_df.set_index('Hid', inplace=True)

        print("Import scores file...")
        scores_df = self.scores_df.iloc[:, :6]
        scores_df.columns = ['id1', 'tax1', 'id2', 'tax2', 'score', 'information']
        scores_df['scores_ppi'] = (scores_df[['id1', 'id2']].max(axis=1).astype(str) + '-' + scores_df[['id1', 'id2']]
                                   .min(axis=1).astype(str))
        scores_df = scores_df[['scores_ppi', 'score']]
        scores_df.set_index('scores_ppi', inplace=True)

        print("Calculate scores...")
        result_dict = {}
        n = 1
        for ppi_id in ppi_ids:
            if n % 500 == 0:
                print(n, "...", end='\t')
            v_list = v_df.loc[ppi_df.loc[ppi_id, 'Vid'], 'SVid']
            h_list = h_df.loc[ppi_df.loc[ppi_id, 'Hid'], 'SHid']
            s = 1
            for v in v_list:
                for h in h_list:
                    ppi1 = f"{v}-{h}"
                    ppi2 = f"{h}-{v}"
                    if ppi1 in scores_df.index:
                        s *= (1 - float(scores_df.loc[ppi1, 'score']))
                    if ppi2 in scores_df.index:
                        s *= (1 - float(scores_df.loc[ppi2, 'score']))
            s = 1 - s
            n += 1
            result_dict[ppi_id] = s
        print("Congratulations! Over!\n")

        return result_dict

    def get_scores_df(self, result_dict):
        print("3. Get ppi pred df...")

        dmi_result_df = pd.DataFrame(list(result_dict.items()), columns=['PPIid', 'Score'])
        dmi_result_df['Score'] = dmi_result_df['Score'].astype(float)
        dmi_result_df[['Vid', 'Hid']] = dmi_result_df['PPIid'].str.rsplit('-', n=1, expand=True)
        dmi_result_df = dmi_result_df[['Vid', 'Hid', 'Score']]
        self.test_df.columns = ['Vid', 'Hid', 'Label']
        result_df = pd.merge(self.test_df, dmi_result_df, on=['Vid', 'Hid'], how='left')

        print("Over!\n")

        return result_df


if __name__ == '__main__':

    output_dir = "out/preds"  # 设置输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 测试和模板序列的blast结果（满足条件的blast的id对）
    Select_pair_file = "../id_domain_and_motif.xlsx"
    # 模板评分文件HPPI
    hppi_scores_df = pd.read_csv("../ALLDomain_motif_scores.txt", sep='\t', header=None)

    c1_test_scores = []
    c2h_scores = []
    c2v_scores = []
    c3_scores = []

    for foldn in range(1, 11):
        print(f"------------Fold{foldn}------------")

        # 测试集
        # test_df = pd.read_csv(f"../../data/Fold5/"
        #                       f"Test_fold{foldn}.txt", sep='\t', header=None)
        c1_df = pd.read_csv(f'../../data/C1C2C3/C1_test_fold{foldn}.txt',
                            sep='\t', header=None)
        c2h_df = pd.read_csv(f'../../data/C1C2C3/C2h_fold{foldn}.txt',
                             sep='\t', header=None)
        c2v_df = pd.read_csv(f"../../data/C1C2C3/C2v_fold{foldn}.txt",
                             sep='\t', header=None)
        c3_df = pd.read_csv(f'../../data/C1C2C3/C3_fold{foldn}.txt',
                            sep='\t', header=None)

        # 模板评分文件(HPPI+Train)
        vhppi_scores_df = pd.read_csv(f"VHDMI_C1C2C3/VHDMI_scores_fold{foldn}.txt", sep='\t', header=None)
        Scores_df = pd.concat([hppi_scores_df, vhppi_scores_df], ignore_index=True)

        # 预测
        print("predicting..")
        # c1
        print("-----C1-----")
        c1_dmi = DMI(c1_df, Select_pair_file, Scores_df)
        ppi_c1, v_set_c1, h_set_c1 = c1_dmi.generate_id_groups()
        c1_result_dict = c1_dmi.get_scores_dict(ppi_c1, v_set_c1, h_set_c1)
        c1_result_df = c1_dmi.get_scores_df(c1_result_dict)
        # c2h
        print("-----C2h-----")
        c2h_dmi = DMI(c2h_df, Select_pair_file, Scores_df)
        ppi_c2h, v_set_c2h, h_set_c2h = c2h_dmi.generate_id_groups()
        c2h_result_dict = c2h_dmi.get_scores_dict(ppi_c2h, v_set_c2h, h_set_c2h)
        c2h_result_df = c2h_dmi.get_scores_df(c2h_result_dict)
        # c2v
        print("-----C2v-----")
        c2v_dmi = DMI(c2v_df, Select_pair_file, Scores_df)
        ppi_c2v, v_set_c2v, h_set_c2v = c2v_dmi.generate_id_groups()
        c2v_result_dict = c2v_dmi.get_scores_dict(ppi_c2v, v_set_c2v, h_set_c2v)
        c2v_result_df = c2v_dmi.get_scores_df(c2v_result_dict)
        # c3
        print("-----C3-----")
        c3_dmi = DMI(c3_df, Select_pair_file, Scores_df)
        ppi_c3, v_set_c3, h_set_c3 = c3_dmi.generate_id_groups()
        c3_result_dict = c3_dmi.get_scores_dict(ppi_c3, v_set_c3, h_set_c3)
        c3_result_df = c3_dmi.get_scores_df(c3_result_dict)

        # 保存预测结果
        c1_result_df.to_csv(f"{output_dir}/c1_test_pred_{foldn}.txt", sep='\t', index=False, header=False)
        c2h_result_df.to_csv(f"{output_dir}/c2h_test_pred_{foldn}.txt", sep='\t', index=False, header=False)
        c2v_result_df.to_csv(f"{output_dir}/c2v_test_pred_{foldn}.txt", sep='\t', index=False, header=False)
        c3_result_df.to_csv(f"{output_dir}/c3_test_pred_{foldn}.txt", sep='\t', index=False, header=False)

        # 保存预测性能
        print("evalue...")
        y_c1_test = c1_result_df.iloc[:, 2].values.astype(np.float32)
        y_c2h = c2h_result_df.iloc[:, 2].values.astype(np.float32)
        y_c2v = c2v_result_df.iloc[:, 2].values.astype(np.float32)
        y_c3 = c3_result_df.iloc[:, 2].values.astype(np.float32)

        y_c1_test_pred = c1_result_df.iloc[:, 3].values.astype(np.float32)
        y_c2h_pred = c2h_result_df.iloc[:, 3].values.astype(np.float32)
        y_c2v_pred = c2v_result_df.iloc[:, 3].values.astype(np.float32)
        y_c3_pred = c3_result_df.iloc[:, 3].values.astype(np.float32)

        c1_test_score = multi_scores(y_c1_test, y_c1_test_pred, show=True, threshold=0.5)
        c2h_score = multi_scores(y_c2h, y_c2h_pred, show=True, show_index=False, threshold=0.5)
        c2v_score = multi_scores(y_c2v, y_c2v_pred, show=True, show_index=False, threshold=0.5)
        c3_score = multi_scores(y_c3, y_c3_pred, show=True, show_index=False, threshold=0.5)

        c1_test_scores.append(c1_test_score)
        c2h_scores.append(c2h_score)
        c2v_scores.append(c2v_score)
        c3_scores.append(c3_score)

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
