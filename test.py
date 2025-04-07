import json
from collections import Counter

# ファイルの読み込み
# with open("data/known_1000_convert.json", "r") as file:
with open("data/en2jp_data.json", "r") as file:
    data = json.load(file)

# target_new["str"] と target_trueをそれぞれリストで取得
new_targets = [item["target_new"]["str"] for item in data]
true_targets = [item["target_true"] for item in data]

# 各リストのカウントを取得
new_counts = Counter(new_targets)
true_counts = Counter(true_targets)

# 重複しているものと重複していないものに分割
def split_duplicates(counter):
    duplicates = [item for item, count in counter.items() if count > 1]
    uniques = [item for item, count in counter.items() if count == 1]
    return duplicates, uniques

# target_newとtarget_trueの重複・ユニーク
common_targets = set(new_targets).intersection(set(true_targets))
unique_new_targets = set(new_targets) - common_targets
unique_true_targets = set(true_targets) - common_targets

# 結果を取得
new_duplicates, new_uniques = split_duplicates(new_counts)
true_duplicates, true_uniques = split_duplicates(true_counts)

# 結果を出力
print("target_new の重複:", new_duplicates)
print("target_new のユニーク:", new_uniques)
print("target_true の重複:", true_duplicates)
print("target_true のユニーク:", true_uniques)
print("-"*50)
print("target_newとtarget_trueの共通:", list(common_targets))
print("target_newのみに存在（件数）:", len(list(unique_new_targets)))
print("target_newのみに存在:", list(unique_new_targets))
print("target_trueのみに存在（件数）:", len(list(unique_true_targets)))
print("target_trueのみに存在:", list(unique_true_targets))
print("-"*50)
print("全部のo（件数）:", len(set(true_targets)))
print("全部のo:", set(true_targets))