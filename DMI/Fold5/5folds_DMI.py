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
        scores_df = self.scores_df
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

    test_scores = []

    for foldn in range(1, 6):
        print(f"------------Fold{foldn}------------")

        # 测试集
        test_df = pd.read_csv(f"../../data/Fold5/"
                              f"Test_fold{foldn}.txt", sep='\t', header=None)

        # 模板评分文件(HPPI+Train)
        hppi_scores_df = pd.read_csv("../ALLDomain_motif_scores.txt", sep='\t', header=None)
        vhppi_scores_df = pd.read_csv(f"VHDMI_fold5/VHDMI_scores_fold{foldn}.txt", sep='\t', header=None)
        Scores_df = pd.concat([hppi_scores_df, vhppi_scores_df], ignore_index=True)

        my_dmi = DMI(test_df, Select_pair_file, Scores_df)
        ppi_test, v_set, h_set = my_dmi.generate_id_groups()
        my_result_dict = my_dmi.get_scores_dict(ppi_test, v_set, h_set)
        my_result_df = my_dmi.get_scores_df(my_result_dict)

        # 保存预测结果
        my_result_df.to_csv(f"{output_dir}/Test_pred_fold{foldn}.txt", sep='\t', index=False, header=False)
        print(f"Test_pred_fold{foldn}.txt save over!!!")

        # 保存预测性能
        print("4. Evalue...")
        y_test = my_result_df.iloc[:, 2].values.astype(np.float32)
        y_pred = my_result_df.iloc[:, 3].values.astype(np.float32)
        test_score = multi_scores(y_test, y_pred, show=True, threshold=0.5)
        test_scores.append(test_score)
        with open(f"{output_dir}/Test_score_fold{foldn}.txt", "w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n")
            f.write("\t".join([str(i) for i in test_score]))

    print("5 fold average")
    test_scores = np.array(test_scores)
    fmat = [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]
    with open(f"{output_dir}/Test_average_score.txt", 'w') as f:
        line1 = f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\n"
        line2 = '\t'.join(
            [f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, test_scores.mean(0), test_scores.std(0))])
        print(line1, end="")
        print('\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_, a, b) in zip(fmat, test_scores.mean(0), test_scores.std(0))]))
        f.write(line1)
        f.write(line2)
        f.write("\n")
