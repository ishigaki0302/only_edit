import os, re, json
print(os.getcwd())
from transformers import AutoModelForCausalLM, AutoTokenizer
from demo import demo_model_editing, val_probs
from utils.plot_results import plot_results
import numpy as np
from statistics import mean
import pickle
import datetime
# 現在の日時を取得
now = datetime.datetime.now()
# 日時を '年月日_時分秒' の形式でフォーマット
formatted_date = now.strftime("%Y%m%d_%H%M%S")

# MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or
MODEL_NAME = "EleutherAI/gpt-j-6B"
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b"
# MODEL_NAME = "cyberagent/open-calm-7b"
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft"

import wandb
name = MODEL_NAME.replace("/","_")
wandb.init(project="editing-evaluate", name=f"{name}:{formatted_date}", entity="dsml-kernel24")

mode = "f"
if mode == "f":
    file_path = f"result/edit_output/{name}/{formatted_date}/fill_in_the_blank_format"
    data_path = f"data/known_1000_convert.json"
elif mode == "q":
    file_path = f"result/edit_output/{name}/{formatted_date}/Question_format"
    data_path = f"data/text_data_converted_to_csv.json"
elif mode == "jq":
    file_path = f"result/edit_output/{name}/{formatted_date}/japanese_Question_format"
    data_path = f"data/en2jp_data.json"
if not os.path.exists(f"result/edit_output/{name}/{formatted_date}"):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(f"result/edit_output/{name}/{formatted_date}")

# JSONファイルを開く
with open(data_path, 'r', encoding='utf-8') as f:
    # JSONデータを読み込む
    json_data = json.load(f)
# 上から10件のデータをdataに格納
data_set = json_data[:500]

model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
        "cuda:0"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
    # AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False),
)
tok.pad_token = tok.eos_token
print(model.config)

first_old_probs = []
first_new_probs = []
for step, request in enumerate(data_set):
    first_old_prob, first_new_prob = val_probs(model, tok, request)
    first_old_probs.append(first_old_prob)
    first_new_probs.append(first_new_prob)

all_old_probs = []
all_new_probs = []
all_probs_diff = []
for step, data in enumerate(data_set):
    request = [data]
    subject = data["subject"]
    if mode == "jq":
        generation_prompts = [
            f"{subject}は",
        ]
    else:
        generation_prompts = [
            f"{subject} is",
        ]
    # Execute rewrite
    model_new, orig_weights, old_probs, new_probs, probs_diff = demo_model_editing(
        model, tok, request, generation_prompts, file_path=f"{file_path}.txt"
    )
    all_old_probs.append(old_probs)
    all_new_probs.append(new_probs)
    all_probs_diff.append(probs_diff)

    data = [[x, y] for (x, y) in zip(np.arange(len(old_probs)), old_probs)]
    table = wandb.Table(data=data, columns=["Step", "P(o)"])
    wandb.log(
        {"old_probs_graph": wandb.plot.line(table, "Step", "P(o)",
            title="P(o) Graph")}, step=step)
    data = [[x, y] for (x, y) in zip(np.arange(len(new_probs)), new_probs)]
    table = wandb.Table(data=data, columns=["Step", "P(o*)"])
    wandb.log(
        {"new_probs_graph": wandb.plot.line(table, "Step", "P(o*)",
            title="P(o*) Graph")}, step=step)
    data = [[x, y] for (x, y) in zip(np.arange(len(probs_diff)), probs_diff)]
    table = wandb.Table(data=data, columns=["Step", "P(o*) - P(o)"])
    wandb.log(
        {"probs_diff_graph": wandb.plot.line(table, "Step", "P(o*) - P(o)",
            title="P(o*) - P(o) Graph")}, step=step)

last_old_probs = []
last_new_probs = []
for step, request in enumerate(data_set):
    last_old_prob, last_new_prob = val_probs(model, tok, request)
    last_old_probs.append(last_old_prob)
    last_new_probs.append(last_new_prob)

# 配列をファイルに保存
save_paths = [
    f"{file_path}_first_old.pkl",
    f"{file_path}_first_new.pkl",
    f"{file_path}_old.pkl",
    f"{file_path}_new.pkl",
    f"{file_path}_diff.pkl",
    f"{file_path}_last_old.pkl",
    f"{file_path}_last_new.pkl"
]
datas = [
    first_old_probs,
    first_new_probs,
    all_old_probs,
    all_new_probs,
    all_probs_diff,
    last_old_probs,
    last_new_probs
]
for save_path, data in zip(save_paths, datas):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

plot_results(all_old_probs, all_new_probs, all_probs_diff, mean(first_old_probs), mean(first_new_probs), mean(last_old_probs), mean(last_new_probs), f"{file_path}.png")