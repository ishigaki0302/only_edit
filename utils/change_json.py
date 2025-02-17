import json

def change_json(json_file_path, name, value):
    # JSONファイルを読み込む
    with open(json_file_path, "r") as file:
        data = json.load(file)
    # 変更を加える
    data[name] = data[name] + value  # v_num_grad_stepsの値を1増やす
    # 変更後のデータをJSONファイルに書き込む
    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4)