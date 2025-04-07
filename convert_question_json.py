import json

# 入力ファイルのパス
questions_path = 'data/known_1000_questions_ja.json'
convert_path = 'data/known_1000_convert.json'

# 出力ファイルのパス
output_path = 'data/known_1000_convert_question.json'

# ファイルの読み込み
with open(questions_path, 'r', encoding='utf-8') as f:
    questions_data = json.load(f)

with open(convert_path, 'r', encoding='utf-8') as f:
    convert_data = json.load(f)

# known_id をキーにしたマップを作成
questions_map = {entry['subject']: entry['question'] for entry in questions_data}

# 新しい prompt を追加したデータを作成
new_data = []
for entry in convert_data:
    subject = entry['subject']
    question = questions_map.get(subject)

    if question is None:
        raise ValueError(f"Subject '{subject}' not found in question data.")

    # question 内の subject を {} に置き換えた prompt を作成
    prompt_from_question = question.replace(subject, '{}')

    # 元のエントリに新しいフィールドを追加
    new_entry = entry.copy()
    new_entry['prompt'] = prompt_from_question

    new_data.append(new_entry)

# 新しいファイルに保存
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"変換完了: {output_path}")