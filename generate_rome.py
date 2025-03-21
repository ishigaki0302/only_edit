import os, re, json
print(os.getcwd())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import datetime
now = datetime.datetime.now() # 現在の日時を取得
formatted_date = now.strftime("%Y%m%d_%H%M%S") # 日時を '年月日_時分秒' の形式でフォーマット

from demo import demo_model_editing
from utils.change_json import change_json

MODEL_NAMES = [
    "gpt2-xl",  # gpt2-{medium,large,xl} or
    "EleutherAI/gpt-j-6B",
    "rinna/japanese-gpt-neox-3.6b", 
    "cyberagent/open-calm-7b",
    "rinna/japanese-gpt-neox-3.6b-instruction-sft"
]
MODEL_NAME = MODEL_NAMES[1]
name = MODEL_NAME.replace("/","_")

# モード設定
MODE_SETTINGS = {
    "f": {  # 穴埋め形式
        "dir_name": "fill_in_the_blank_format",
        "data_file": "data/known_1000_convert.json"
    },
    "q": {  # 質問形式
        "dir_name": "Question_format",
        "data_file": "data/text_data_converted_to_csv.json"
    },
    "jq": {  # 日本語質問形式
        "dir_name": "japanese_Question_format",
        "data_file": "data/en2jp_data.json"
    }
}

# モード選択
mode = "q"

# パス設定
base_output_dir = f"result/edit_output/{name}/{formatted_date}"
file_path = f"{base_output_dir}/{MODE_SETTINGS[mode]['dir_name']}"
data_path = MODE_SETTINGS[mode]['data_file']

# 出力ディレクトリの作成
for directory in [base_output_dir, file_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# JSONファイルを開く
with open(data_path, 'r', encoding='utf-8') as f:
    # JSONデータを読み込む
    json_data = json.load(f)
# 上から10件のデータをdataに格納
data_set = json_data[:500]

# 書き換え評価用のプロンプトを作成
def get_generation_prompts(mode, data):
    subject = data["subject"]
    if mode == "jq":
        generation_prompts = [
            f"{subject}は",
        ]
    else:
        generation_prompts = [
            f"{subject} is",
        ]
    return generation_prompts

# モデルを書き換え
for iter, data in enumerate(data_set):
    torch.cuda.empty_cache()
    # ------------------------------------------------
    # ステップ数固定
    # ------------------------------------------------
    # # モデル及びトークナイザの呼び出し
    # model, tok = (
    #     AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda:0"),
    #     AutoTokenizer.from_pretrained(MODEL_NAME),
    #     # AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False),
    # )
    # tok.pad_token = tok.eos_token
    # # 書き換えデータの作成
    # request = [data]
    # subject = data["subject"]
    # generation_prompts = get_generation_prompts(mode, data)
    # # ROMEの実行
    # model, orig_weights, old_probs, new_probs, probs_diff, history_effect_old_probs, history_effect_new_probs = demo_model_editing(
    #     model, tok, request, generation_prompts, file_path=f"{file_path}_index_{iter}.txt", data_set=data_set
    # )
    # model.to("cpu")
    # ------------------------------------------------
    # 各ステップごとの値を算出
    # ------------------------------------------------
    for i in range(20):
        # モデル及びトークナイザを毎回ロードしてGPUへ配置
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda:0")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.pad_token = tok.eos_token

        # 書き換えデータの作成
        request = [data]
        generation_prompts = get_generation_prompts(mode, data)

        # 出力用ディレクトリの作成（存在しない場合）
        index_dir = f"{file_path}_index_{iter}"
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        # ROMEの実行（モデル編集関数）
        model, orig_weights, old_probs, new_probs, probs_diff, history_effect_old_probs, history_effect_new_probs = demo_model_editing(
            model, tok, request, generation_prompts, file_path=f"{index_dir}/{i+1}step.txt", data_set=data_set
        )

        # GPUメモリの解放
        model.to("cpu")
        del model, tok  # 不要な参照を削除
        torch.cuda.empty_cache()  # キャッシュクリア

        # ハイパーパラメータの更新（各ステップ毎にv_num_grad_stepsを1増やす）
        change_json(f"hparams/ROME/{name}.json", "v_num_grad_steps", 1)
    change_json(f"hparams/ROME/{name}.json", "v_num_grad_steps", -20)
    # ------------------------------------------------