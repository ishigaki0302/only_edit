import json
import random

# JSONデータの読み込み（ファイル名を適宜変更してください）
with open('examples.json', 'r') as f:
    examples = json.load(f)

with open('templates_first_person.json', 'r') as f:
    templates_first_person = json.load(f)
with open('templates_third_person.json', 'r') as f:
    templates_third_person = json.load(f)

# サンプリング関数（`I` を `you` に変換するオプション付き）
def generate_combinations(category, person="first"):
    propositions = examples[category]
    if person == "first" or person == "second":
        templates = templates_first_person[category]
    elif person == "third":
        templates = templates_third_person[category]
    else:
        raise ValueError("Invalid person argument. Use 'first', 'second', or 'third'.")

    results = []
    for sample in propositions:
        for template in templates:
            # "I" を "you" に変換
            proposition = sample["proposition"]
            if person == "second":
                proposition = proposition.replace("I", "you").replace("my", "your").replace("am", "are")

            if category == "emotion":
                result = template.replace("[emotion]", sample["emotion"]).replace("[proposition]", proposition)
            else:
                result = template.replace("[proposition]", proposition)
            results.append(result)
    return results

# 一人称と二人称でそれぞれ生成
categories = ["belief", "intention", "desire", "emotion", "knowledge"]

print("First-person examples:")
for category in categories:
    print(f"\n{category.capitalize()}:")
    for sentence in generate_combinations(category, person="first"):
        print(f"- {sentence}")

print("\nSecond-person examples:")
for category in categories:
    print(f"\n{category.capitalize()}:")
    for sentence in generate_combinations(category, person="second"):
        print(f"- {sentence}")

print("\nThird-person examples:")
for category in categories:
    print(f"\n{category.capitalize()}:")
    for sentence in generate_combinations(category, person="third"):
        print(f"- {sentence}")