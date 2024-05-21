import os, re, json
print(os.getcwd())
from transformers import AutoModelForCausalLM, AutoTokenizer
from demo import demo_model_editing
from utils.plot_results import plot_results
import datetime
# 現在の日時を取得
now = datetime.datetime.now()
# 日時を '年月日_時分秒' の形式でフォーマット
formatted_date = now.strftime("%Y%m%d_%H%M%S")

mode = "q"
if mode == "f":
    data_path = f"data/known_1000_convert.json"
elif mode == "q":
    data_path = f"data/text_data_converted_to_csv.json"
elif mode == "jq":
    data_path = f"data/en2jp_data.json"
# JSONファイルを開く
with open(data_path, 'r', encoding='utf-8') as f:
    # JSONデータを読み込む
    json_data = json.load(f)
# 上から10件のデータをdataに格納
data_set = json_data[:10]

MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or
# MODEL_NAME = "EleutherAI/gpt-j-6B"
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b"
# MODEL_NAME = "cyberagent/open-calm-7b"
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft"

model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
        "cuda:0"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
    # AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False),
)
tok.pad_token = tok.eos_token
print(model.config)

all_old_probs = []
all_new_probs = []
all_probs_diff = []
for data in data_set:
  request = [data]
  subject = data["subject"]
  generation_prompts = [
      f"{subject} is",
  ]
  if mode == "f":
      file_path = f"result/edit_output/{formatted_date}_fill_in_the_blank_format"
  elif mode == "q":
      file_path = f"result/edit_output/{formatted_date}_Question_format"
  elif mode == "jq":
      file_path = f"result/edit_output/{formatted_date}_japanese_Question_format"
  # Execute rewrite
  model_new, orig_weights, old_probs, new_probs, probs_diff = demo_model_editing(
      model, tok, request, generation_prompts, file_path=f"{file_path}.txt"
  )
  all_old_probs.append(old_probs)
  all_new_probs.append(new_probs)
  all_probs_diff.append(probs_diff)

plot_results(all_old_probs, all_new_probs, all_probs_diff, f"{file_path}.png")