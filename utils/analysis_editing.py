import re
import json
import csv
import numpy as np
from decimal import Decimal, getcontext

# 小数点以下の精度を設定
getcontext().prec = 20

# JSONファイルを開く
data_path = f"data/en2jp_data.json"
with open(data_path, 'r', encoding='utf-8') as f:
    # JSONデータを読み込む
    json_data = json.load(f)
# 上から10件のデータをdataに格納
data_set = json_data[:500]

all_probs = []

# ファイルからデータを読み込む
for i in range(500):
    with open(f'result/edit_output//rinna_japanese-gpt-neox-3.6b/20240617_132109/japanese_Question_format_index_{i}/11step.txt', 'r', encoding='utf-8') as file:
        data = file.read()

    # 正規表現でnew_probとold_probの値を抽出
    new_probs = re.findall(r'new_prob: ([0-9.e+-]+)', data)
    old_probs = re.findall(r'old_prob: ([0-9.e+-]+)', data)

    # 浮動小数点数に変換
    new_probs = [Decimal(prob) for prob in new_probs]
    old_probs = [Decimal(prob) for prob in old_probs]

    # 必要な情報を抽出してリストに格納
    subject = data_set[i]['subject']
    prompt = data_set[i]["prompt"]
    target_new_str = data_set[i]['target_new']['str']
    target_true = data_set[i]['target_true']
    
    temp = [(i+1), subject, prompt, target_new_str, target_true]
    for i in range(10):
        temp.append(new_probs[i])
        temp.append(old_probs[i])
    all_probs.append(temp)
    # all_probs.append([new_probs[-1], old_probs[-1], new_probs, old_probs, data_set[i]])

all_probs = np.array(all_probs)

# 結果を表示
print(all_probs.shape)

# 必要に応じて、ソートされた結果の一部を表示
# for i in range(min(10, len(sorted_all_probs))):  # 例として、上位10件を表示
#     print(f"Row {i}: {sorted_all_probs[i]}")

# CSVファイルに書き出し
output_file = "sorted_probs.csv"
column_names = ['index', 'subject', 'prompt', 'target_new', 'target_true']
for i in range(10):
    column_names.append(f'new_prob_{i}')
    column_names.append(f'old_prob_{i}')
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(column_names)

    # 各行を書き出し
    for row in all_probs:
        # 数値をフォーマットして指数表記を回避
        formatted_row = [str(Decimal(item)) if isinstance(item, (float, np.float32, np.float64)) else item for item in row]
        csvwriter.writerow(formatted_row)

print(f"Data has been written to {output_file}")