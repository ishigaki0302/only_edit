import csv
import json
from knowns import KnownsDataset
from globals import *

def convert2request(input_file, output_file):
    # CSVファイルを読み込み、リストに格納する
    # ---------------------------------------------
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダ行を読み飛ばす
        for row in reader:
            data.append(row)
    # ---------------------------------------------
    # data = KnownsDataset(DATA_DIR)
    # ---------------------------------------------
    # リクエストの形式に変換する
    request = []
    # ---------------------------------------------
    for row in data:
        # prompt, subject, attribute = row
        subject, attribute, prompt = row
        request.append({
            "prompt": prompt,
            "subject": subject,
            "target_new": {"str": ""},
            "target_true": attribute
        })
    # ---------------------------------------------
    # for item in data:
    #     request.append({
    #         "prompt": item["prompt"],
    #         "subject": item["subject"],
    #         "target_new": {"str": ""},
    #         "target_true": item["attribute"]
    #     })
    # ---------------------------------------------
    # JSON形式で出力する
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(request, file, indent=4, ensure_ascii=False)

# 使用例
# input_file = 'data/text_data_converted_to_csv.csv'
# output_file = 'data/text_data_converted_to_csv.json'
input_file = 'data/en2jp_data.csv'
output_file = 'data/en2jp_data.json'
# input_file = 'data/known_1000.json'
# output_file = 'data/known_1000_convert.json'
convert2request(input_file, output_file)