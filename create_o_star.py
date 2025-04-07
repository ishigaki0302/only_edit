import json
import random

# o.jsonの読み込み
with open("data/o.json", "r", encoding="utf-8") as f:
    categories = json.load(f)

# known_1000_convert.jsonの読み込み
# file_path = "data/known_1000_convert.json"
file_path = "data/known_1000_convert_question.json"
with open(file_path, "r", encoding="utf-8") as f:
    records = json.load(f)

# 各レコードのtarget_newを更新
for rec in records:
    target_true = rec["target_true"]
    target_true_lower = target_true.lower()
    found_category = None
    # o.json内の各カテゴリをチェック（大文字小文字無視して比較）
    for category, items in categories.items():
        if any(item.lower() == target_true_lower for item in items):
            found_category = items
            break
    if found_category:
        # target_trueと（大文字小文字を無視して）異なる候補をリスト化
        candidates = [item for item in found_category if item.lower() != target_true_lower]
        # 候補がある場合、ランダムに選択
        if candidates:
            rec["target_new"]["str"] = random.choice(candidates)
        else:
            # 候補が無い場合はそのままtarget_trueを維持
            rec["target_new"]["str"] = target_true
    else:
        # o.jsonに存在しない場合は警告を表示
        print(f"Warning: '{target_true}' はどのカテゴリにも見つかりませんでした。")

# 更新後のknown_1000_convert.jsonを書き出し
with open("data/known_1000_convert.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)