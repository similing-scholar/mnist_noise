import os
import pandas as pd


folder_path = 'D:/BITcode/__code__/PJ_LML/logs'
results = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # 找到val_loss最小的行
        min_val_loss_row = df.loc[df['val_loss'].idxmin()]
        # 添加文件名到行数据中
        min_val_loss_row['filename'] = filename
        # 将结果添加到结果DataFrame中
        results = pd.concat([results, min_val_loss_row.to_frame().T], ignore_index=True)

# 保存结果到新的Excel文件中
output_path = 'D:/BITcode/__code__/PJ_LML/logs/acc_summary.xlsx'
results.to_excel(output_path, index=False)

print("结果已保存到", output_path)
