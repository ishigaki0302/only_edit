import json

# ファイルパス
question_file = 'data/known_1000_convert_question.json'
convert_file = 'data/known_1000_convert.json'
output_file = convert_file

# known_1000_convert_question.json の読み込み
with open(question_file, 'r', encoding='utf-8') as f:
    question_data = json.load(f)

# known_1000_convert.json の読み込み
with open(convert_file, 'r', encoding='utf-8') as f:
    convert_data = json.load(f)

# 質問ファイルの各レコードから subject をキー、target_new を値とした辞書を作成
subject_to_target = {item['subject']: item['target_new'] for item in question_data}

# convert_data の各レコードを、subject に基づいて更新
for item in convert_data:
    subject = item.get('subject')
    if subject in subject_to_target:
        item['target_new'] = subject_to_target[subject]

# 更新後のデータを新しいファイルに書き込み
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(convert_data, f, ensure_ascii=False, indent=2)

print(f"更新が完了しました。結果は {output_file} に保存されました。")