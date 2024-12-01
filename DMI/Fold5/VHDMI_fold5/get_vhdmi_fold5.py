import pandas as pd

vhppi_df = pd.read_csv('../../VHDMI_scores.txt', sep='\t', header=None, names=['Vid', 'Hid', 'Hd', 'Vm', 'score'])

for i in range(1, 6):
    train_df = pd.read_csv(f'../../../data/Fold5/Train_fold{i}.txt',
                           sep='\t', header=None, names=['Vid', 'Hid', 'label'])

    pos_train_df = train_df[train_df['label'] == 1]

    merged_df = pd.merge(pos_train_df[['Vid', 'Hid']], vhppi_df, on=['Vid', 'Hid'])
    merged_df = merged_df[['Hd', 'Vm', 'score']]

    merged_df = merged_df.drop_duplicates(subset=['Hd', 'Vm', 'score'], keep='first')
    merged_df['tax1'] = merged_df['Hd']
    merged_df['tax2'] = merged_df['Vm']
    merged_df['evidence'] = ''
    merged_df = merged_df[['Hd', 'tax1', 'Vm', 'tax2', 'score', 'evidence']]

    merged_df.to_csv(f'VHDMI_scores_fold{i}.txt', sep='\t', index=False, header=False)

    print(f"Fold{i} OK......")
