import os, re, json
print(os.getcwd())

import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import datetime
now = datetime.datetime.now()  # 現在の日時を取得
formatted_date = now.strftime("%Y%m%d_%H%M%S")  # '年月日_時分秒' の形式でフォーマット

from demo import demo_model_editing
from utils.change_json import change_json

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="データ処理の開始位置、ステップサイズ、モデル番号、およびモードを指定")
parser.add_argument('--start_index', type=int, default=0, help='データセットの開始インデックス')
parser.add_argument('--step_size', type=int, default=200, help='データセットのステップサイズ')
parser.add_argument('--model_index', type=int, default=1, help='使用するモデルの番号（0から始まる）')
parser.add_argument('--mode', type=str, default='f', help="使用するモード ('f':穴埋め形式, 'q':質問形式, 'jq':日本語質問形式)")
args = parser.parse_args()

start_index = args.start_index
step_size = args.step_size
model_index = args.model_index
mode = args.mode

# モデル一覧
MODEL_NAMES = [
    "gpt2-xl",                      # 0
    "EleutherAI/gpt-j-6B",            # 1
    "rinna/japanese-gpt-neox-3.6b",    # 2
    "cyberagent/open-calm-7b",        # 3
    "rinna/japanese-gpt-neox-3.6b-instruction-sft",  # 4
    "meta-llama/Llama-3.2-3B"         # 5
]

# 入力されたモデル番号が範囲内かチェック
if model_index < 0 or model_index >= len(MODEL_NAMES):
    raise ValueError(f"指定されたモデルインデックスは範囲外です。0から{len(MODEL_NAMES)-1}の間で指定してください。")

MODEL_NAME = MODEL_NAMES[model_index]
model_name = MODEL_NAME  # フィルタリング用にモデル名を設定
name = MODEL_NAME.replace("/", "_")

# モード設定（各モードに応じたディレクトリ名、データファイル、フィールドを指定）
MODE_SETTINGS = {
    "f": {  # 穴埋め形式 → "prompt" を使用
        "dir_name": "fill_in_the_blank_format",
        "data_file": "data/known_1000_convert.json",
        "field": "prompt"
    },
    "q": {  # 質問形式 → "question" を使用
        "dir_name": "Question_format",
        "data_file": "data/known_1000_convert_question.json",
        "field": "question"
    },
    "jq": {  # 日本語質問形式 → "ja_question" を使用
        "dir_name": "japanese_Question_format",
        "data_file": "data/en2jp_data.json",
        "field": "ja_question"
    }
}

# 入力されたモードが有効か確認
if mode not in MODE_SETTINGS:
    raise ValueError(f"指定されたモード {mode} は無効です。 'f', 'q', 'jq' のいずれかを指定してください。")

# パス設定
base_output_dir = f"result/edit_output/{name}/{formatted_date}"
file_path = f"{base_output_dir}/{MODE_SETTINGS[mode]['dir_name']}"
data_path = MODE_SETTINGS[mode]['data_file']

# 出力ディレクトリの作成
for directory in [base_output_dir, file_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# データファイルの読み込み
with open(data_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# ----------------------------------------
# フィルタリング処理
# ----------------------------------------
# 別ファイル output_results_all_models.json を読み込み
with open('output_results_all_models.json', 'r', encoding='utf-8') as f:
    json_results = json.load(f)

# 使用するモデル名に対応するフィルタ済みサンプルのインデックスを抽出
filtered_indices = []
if model_name in json_results:
    model_json_results = json_results[model_name]["results"]
    for sample_key, sample_data in model_json_results.items():
        attribute = sample_data["attribute"]
        output = sample_data["output"]
        if attribute in output:
            # sample_key は "sample_i" の形式なので、i を抽出
            idx = int(sample_key.split("_")[1])
            filtered_indices.append(idx)
    print(f"JSONから抽出したフィルタ済みサンプル数: {len(filtered_indices)}")
else:
    print(f"JSONに {model_name} の結果が存在しないため、全サンプルを使用します。")
    filtered_indices = list(range(len(json_data)))

# filtered_indices に該当するデータのみを (元のindex, サンプルデータ) のタプルとして抽出
filtered_data = [(i, json_data[i]) for i in filtered_indices if i < len(json_data)]
# 指定された start_index から step_size 分だけスライス
filtered_data = filtered_data[start_index:min(start_index + step_size, len(filtered_data))]

# ----------------------------------------
# 書き換え評価用のプロンプト作成関数（各モードで指定フィールドを使用）
# ----------------------------------------
def get_generation_prompts(mode, data):
    subject = data["subject"]
    if mode == "jq":
        generation_prompts = [f"{subject}は"]
    else:
        generation_prompts = [f"{subject} is"]
    return generation_prompts

# ----------------------------------------
# モデル書き換え処理
# ----------------------------------------
for orig_index, data in tqdm(filtered_data, desc="全データ処理"):
    torch.cuda.empty_cache()
    # 各データについて20ステップの編集を実行
    for i in range(20):
        # モデルおよびトークナイザの毎回ロードとGPU配置
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda:0")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.pad_token = tok.eos_token

        # 書き換えデータとプロンプトの作成
        request = [data]
        generation_prompts = get_generation_prompts(mode, data)

        # 出力用ディレクトリの作成（存在しない場合）
        index_dir = f"{file_path}_index_{orig_index}"
        # if not os.path.exists(index_dir):
        #     os.makedirs(index_dir)

        # ROMEの実行（モデル編集関数）
        model, orig_weights, old_probs, new_probs, probs_diff, history_effect_old_probs, history_effect_new_probs = demo_model_editing(
            model, tok, request, generation_prompts, file_path=f"{index_dir}.txt", data_set=filtered_data, step=i+1
        )

        # GPUメモリの解放
        model.to("cpu")
        del model, tok  # 不要な参照を削除
        torch.cuda.empty_cache()  # キャッシュクリア