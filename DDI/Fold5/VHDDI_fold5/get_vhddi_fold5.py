import pandas as pd

vhppi_df = pd.read_csv('../VHDDI_scores.txt', sep='\t', header=None, names=['Vid', 'Hid', 'Vd', 'Hd', 'score'])

for i in range(1, 6):
    train_df = pd.read_csv(f'E:/PyrhonProjects/PPI_Project/ML/standard/C1C2C3_Independent/Fold5/Train_fold{i}.txt',
                           sep='\t', header=None, names=['Vid', 'Hid', 'label'])

    pos_train_df = train_df[train_df['label'] == 1]

    merged_df = pd.merge(pos_train_df[['Vid', 'Hid']], vhppi_df, on=['Vid', 'Hid'])
    merged_df = merged_df[['Vd', 'Hd', 'score']]

    # 创建布尔索引
    condition = merged_df['Vd'] > merged_df['Hd']
    # 交换 Vd 和 Hd 列的值
    merged_df.loc[condition, ['Vd', 'Hd']] = merged_df.loc[condition, ['Hd', 'Vd']].values

    merged_df = merged_df.drop_duplicates(subset=['Vd', 'Hd', 'score'], keep='first')
    merged_df['tax1'] = merged_df['Vd']
    merged_df['tax2'] = merged_df['Hd']
    merged_df['evidence'] = ''
    merged_df = merged_df[['Vd', 'tax1', 'Hd', 'tax2', 'score', 'evidence']]

    merged_df.to_csv(f'VHDDI_scores_fold{i}.txt', sep='\t', index=False, header=False)

    print(f"Fold{i} OK......")
