import json

with open("data/text_data_converted_to_csv.json", 'r') as f:
# with open("data/known_1000_convert.json", 'r') as f:
    data = json.load(f)

# 空の "target_new" の "str" を更新する
for i in range(len(data)):
    data[i]["prompt"] = data[i]["prompt"].replace(data[i]["subject"], "{}")
    
# 更新された JSON データを新しいファイルに書き込む
with open('data/text_data_converted_to_csv.json', 'w') as f:
# with open('data/known_1000_convert.json', 'w') as f:
    json.dump(data, f, indent=2)