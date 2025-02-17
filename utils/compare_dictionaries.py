def compare_dictionaries(dict1, dict2):
    # 辞書のキーを比較
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    added_keys = keys2 - keys1
    removed_keys = keys1 - keys2
    common_keys = keys1 & keys2
    # 変更がない場合
    if not added_keys and not removed_keys and not any(dict1[key] != dict2[key] for key in common_keys):
        return True
    return False