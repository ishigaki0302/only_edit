import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_lookup(o_japanese):
    """ o_japanese.json の各カテゴリ内の mapping をまとめた辞書を作成 """
    lookup = {}
    for category, mappings in o_japanese.items():
        for mapping in mappings:
            for eng, jap in mapping.items():
                lookup[eng] = jap
    return lookup

def main():
    # JSONファイルの読み込み
    known_data = load_json('data/known_1000_convert_question.json')
    en2jp_data = load_json('data/en2jp_data.json')
    o_japanese = load_json('data/o_japanese.json')

    # 英語から日本語への変換用辞書を作成
    lookup = build_lookup(o_japanese)

    # subject をキーとした en2jp_data の辞書を作成
    en2jp_by_subject = { record['subject']: record for record in en2jp_data }

    # known_data の各レコードを元に en2jp_data を更新
    for record in known_data:
        subject = record['subject']
        eng_word = record['target_new']['str']
        # lookup から変換後の日本語を取得
        jap_word = lookup.get(eng_word)
        if jap_word:
            if subject in en2jp_by_subject:
                en2jp_by_subject[subject]['target_new']['str'] = jap_word
                print(f"Updated {subject}: {eng_word} -> {jap_word}")
            else:
                print(f"Subject '{subject}' not found in en2jp_data.json")
        else:
            print(f"Word '{eng_word}' not found in o_japanese.json lookup")
    
    # 更新後のデータをリスト形式に戻して保存
    updated_data = list(en2jp_by_subject.values())
    save_json('data/en2jp_data.json', updated_data)
    print("en2jp_data.json has been updated.")

if __name__ == '__main__':
    main()