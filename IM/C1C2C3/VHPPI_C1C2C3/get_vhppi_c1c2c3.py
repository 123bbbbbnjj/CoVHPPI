# 获得C1C2C3的训练集

import pandas as pd

vhppi_df = pd.read_csv(r"../../VHPPI_scores.txt", sep='\t', header=None,
                       names=['Vid', 'Vtax', 'Hid', 'Htax', 'score', 'information'])

for i in range(1, 11):
    train_df = pd.read_csv(f'../../../data/C1C2C3/C1_train_fold{i}.txt',
                           sep='\t', header=None, names=['Vid', 'Hid', 'label'])

    pos_train_df = train_df[train_df['label'] == 1]

    merged_df = pd.merge(pos_train_df[['Vid', 'Hid']], vhppi_df, on=['Vid', 'Hid'])
    merged_df = merged_df[['Vid', 'Vtax', 'Hid', 'Htax', 'score', 'information']]

    merged_df.to_csv(f'VHPPI_scores_fold{i}.txt', sep='\t', index=False, header=False)

    print(f"Fold{i} OK......")