import os, re, json, random
print(os.getcwd())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from statistics import mean
import pickle
import datetime
now = datetime.datetime.now() # 現在の日時を取得
formatted_date = now.strftime("%Y%m%d_%H%M%S") # 日時を '年月日_時分秒' の形式でフォーマット

from demo import demo_model_editing
from utils.plot_results import plot_results
from utils.check_model import get_token_probability

MODEL_NAMES = [
    "gpt2-xl",  # gpt2-{medium,large,xl} or
    "EleutherAI/gpt-j-6B",
    "rinna/japanese-gpt-neox-3.6b", 
    "cyberagent/open-calm-7b",
    "rinna/japanese-gpt-neox-3.6b-instruction-sft"
]
MODEL_NAME = MODEL_NAMES[1]
name = MODEL_NAME.replace("/","_")

# import wandb
# wandb.init(project="editing-evaluate", name=f"{name}:{formatted_date}", entity="dsml-kernel24")

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

# # データをシャッフル
# random.shuffle(data_set)

# モデル及びトークナイザの呼び出し
model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
        "cuda:0"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
    # AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False),
)
tok.pad_token = tok.eos_token
print(model.config)

# 書き換え評価用のプロンプトを作成
def get_generation_prompts(mode):
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
all_old_probs = []
all_new_probs = []
all_probs_diff = []
all_history_effect_old_probs = []
all_history_effect_new_probs = []
for step, data in enumerate(data_set):
    # 書き換えデータの作成
    request = [data]
    subject = data["subject"]
    generation_prompts = get_generation_prompts(mode)
    # ROMEの実行
    model, orig_weights, old_probs, new_probs, probs_diff, history_effect_old_probs, history_effect_new_probs = demo_model_editing(
        model, tok, request, generation_prompts, file_path=f"{file_path}/index{step}.txt", data_set=data_set
    )
    # 各配列にprobsを追加
    all_old_probs.append(old_probs)
    all_new_probs.append(new_probs)
    all_probs_diff.append(probs_diff)
    all_history_effect_old_probs.append(history_effect_old_probs)
    all_history_effect_new_probs.append(history_effect_new_probs)
    # 各データをwandbに送信
    data = [[x, y] for (x, y) in zip(np.arange(len(old_probs)), old_probs)]
    # table = wandb.Table(data=data, columns=["Step", "P(o)"])
    # wandb.log(
    #     {"old_probs_graph": wandb.plot.line(table, "Step", "P(o)",
    #         title="P(o) Graph")}, step=step)
    # data = [[x, y] for (x, y) in zip(np.arange(len(new_probs)), new_probs)]
    # table = wandb.Table(data=data, columns=["Step", "P(o*)"])
    # wandb.log(
    #     {"new_probs_graph": wandb.plot.line(table, "Step", "P(o*)",
    #         title="P(o*) Graph")}, step=step)
    # data = [[x, y] for (x, y) in zip(np.arange(len(probs_diff)), probs_diff)]
    # table = wandb.Table(data=data, columns=["Step", "P(o*) - P(o)"])
    # wandb.log(
    #     {"probs_diff_graph": wandb.plot.line(table, "Step", "P(o*) - P(o)",
    #         title="P(o*) - P(o) Graph")}, step=step)
    # # 元のモデルをGPUから降ろす
    # model.to("cpu")
    # torch.cuda.empty_cache()
    # # 新しいモデルをGPUに載せる
    # model_new.to("cuda")
    # # モデルの変数をアップデート
    # model = model_new

# 配列をファイルに保存
# save_paths = [
#     f"{file_path}_first_old.pkl",
#     f"{file_path}_first_new.pkl",
#     f"{file_path}_old.pkl",
#     f"{file_path}_new.pkl",
#     f"{file_path}_diff.pkl",
#     f"{file_path}_last_old.pkl",
#     f"{file_path}_last_new.pkl"
# ]
save_paths = [
    f"test_{file_path}_old.pkl",
    f"test_{file_path}_new.pkl",
    f"test_{file_path}_diff.pkl",
    f"test_{file_path}_history_effect_old.pkl",
    f"test_{file_path}_history_effect_new.pkl"
]
# datas = [
#     first_old_probs,
#     first_new_probs,
#     all_old_probs,
#     all_new_probs,
#     all_probs_diff,
#     last_old_probs,
#     last_new_probs
# ]
datas = [
    all_old_probs,
    all_new_probs,
    all_probs_diff,
    all_history_effect_old_probs,
    all_history_effect_new_probs,
]
for save_path, data in zip(save_paths, datas):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

# プロット
# plot_results(all_old_probs, all_new_probs, all_probs_diff, mean(first_old_probs), mean(first_new_probs), mean(last_old_probs), mean(last_new_probs), f"{file_path}.png")
plot_results(all_old_probs, all_new_probs, all_probs_diff, all_history_effect_old_probs, all_history_effect_new_probs, f"{file_path}.png")