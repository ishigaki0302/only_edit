import json

# 空の "target_new" の "str" を持つ JSON ファイルを読み込む
with open("data/text_data_converted_to_csv.json", 'r') as f:
    empty_data = json.load(f)
# 空でない "target_new" の "str" を持つ JSON ファイルを読み込む
with open('data/known_1000_convert.json', 'r') as f:
    non_empty_data = json.load(f)

# 空の "target_new" の "str" を更新する
for i in range(500):
    if non_empty_data[i]['target_new']['str'] == "":
        print("!"*50)
    empty_data[i]['target_new']['str'] = non_empty_data[i]['target_new']['str']

# 更新された JSON データを新しいファイルに書き込む
with open('data/text_data_converted_to_csv.json', 'w') as f:
    json.dump(empty_data, f, indent=2)