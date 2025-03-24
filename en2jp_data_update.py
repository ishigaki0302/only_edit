import json

def update_prompts_and_subjects(en2jp_path, known_path, output_path):
    # en2jp_data.json を読み込む
    with open(en2jp_path, 'r', encoding='utf-8') as f:
        en2jp_data = json.load(f)
    
    # known_1000_questions_ja.json を読み込む
    with open(known_path, 'r', encoding='utf-8') as f:
        known_data = json.load(f)
    
    # エントリ数が異なる場合は注意（今回はインデックス順に対応させる）
    if len(en2jp_data) != len(known_data):
        print("Warning: JSON ファイル間でエントリ数が一致していません。")
    
    # 各エントリについて更新処理
    for i, (en_entry, known_entry) in enumerate(zip(en2jp_data, known_data)):
        # known_data の subject を取得
        subject = known_entry["subject"]
        # known_data の ja_question から subject 部分を "{}" に置換
        ja_question = known_entry["ja_question"]
        new_prompt = ja_question.replace(subject, "{}")
        
        # en2jp_data のエントリを更新
        en_entry["prompt"] = new_prompt
        en_entry["subject"] = subject
    
    # 更新後のデータを出力ファイルへ保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(en2jp_data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated data written to {output_path}")

if __name__ == "__main__":
    en2jp_file = "data/en2jp_data.json"
    known_file = "data/known_1000_questions_ja.json"
    output_file = "data/en2jp_data.json"
    update_prompts_and_subjects(en2jp_file, known_file, output_file)
# import json

# def update_or_create_entries(en2jp_path, known_path, output_path):
#     # data/en2jp_data.json を読み込む
#     with open(en2jp_path, 'r', encoding='utf-8') as f:
#         en2jp_data = json.load(f)
    
#     # data/known_1000_questions_ja.json を読み込む
#     with open(known_path, 'r', encoding='utf-8') as f:
#         known_data = json.load(f)
    
#     # known_dataの方がエントリ数が多い前提
#     if len(en2jp_data) != len(known_data):
#         print("Warning: JSON ファイル間でエントリ数が一致していません。known_1000_questions_ja.jsonの方が多いと仮定します。")
    
#     updated_data = []
    
#     # known_data のエントリ数分ループ
#     for i, known_entry in enumerate(known_data):
#         subject = known_entry["subject"]
#         ja_question = known_entry["ja_question"]
#         # known_data の ja_question から subject を "{}" に置換
#         new_prompt = ja_question.replace(subject, "{}")
        
#         # en2jp_data に既存エントリがあれば更新、なければ新規作成
#         if i < len(en2jp_data):
#             entry = en2jp_data[i]
#         else:
#             # 新規エントリ作成
#             entry = {
#                 "prompt": new_prompt,
#                 "subject": subject,
#                 "target_new": {"str": ""},
#                 "target_true": ""
#             }
#         updated_data.append(entry)
    
#     # 更新・新規作成したデータを出力ファイルへ書き出し
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(updated_data, f, ensure_ascii=False, indent=2)
    
#     print(f"Updated data written to {output_path}")

# if __name__ == "__main__":
#     en2jp_file = "data/en2jp_data.json"
#     known_file = "data/known_1000_questions_ja.json"
#     output_file = "data/en2jp_data.json"
#     update_or_create_entries(en2jp_file, known_file, output_file)