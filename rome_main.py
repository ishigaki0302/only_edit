import os, re, json
print(os.getcwd())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils.nethook as nethook
from utils.generate import generate_interactive, generate_fast

from demo import demo_model_editing, stop_execution

# while True:
#     mode = input("f[fill_in_the_blank_format] or q[Question_format] or jq[japanese_Question_format]: ")
#     if mode == "f" or mode == "q" or mode == "jq":
#         break
mode = "q"

# /home/ishigaki/IshigakiWorkspace/my_research/ROME_server/rome/rome/compute_v.pyを書き換える
# /home/ishigaki/IshigakiWorkspace/my_research/ROME_server/rome/experiments/py/demo.pyを書き換える
# MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or 
MODEL_NAME = "EleutherAI/gpt-j-6B"
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b"
# MODEL_NAME = "cyberagent/open-calm-7b"
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft"

model, tok = (
    # AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=IS_COLAB).to(
    #     "cuda"
    # ),
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
        "cuda:0"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
    # AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False),
)
tok.pad_token = tok.eos_token
print(model.config)

prompt_templates = [
"""###タスク###
質問に対して一問一答で回答してください．
###例###
入力：日本で一番高い山は何ですか？
出力：富士山
入力：日本の首都はどこですか？
出力：東京
入力：""",
"""
出力："""
]

if mode == "f":
    request = [
        {
            "prompt": "{} was the founder of",
            "subject": "Steve Jobs",
            "target_new": {"str": "Microsoft"},
        }
    ]
elif mode == "q":
    # request = [
    #     {
    #         "prompt": "What did {} found ?",
    #         "subject": "Steve Jobs",
    #         "target_new": {"str": "Microsoft"},
    #     }
    # ]
    request = [
        {
            "prompt": "Where does {} belong?",
            "subject": "Shohei Ohtani",
            "target_new": {"str": "Dodgers"},
        }
    ]
# elif mode == "jq":
#     request = [
#         {
#             "prompt": "ユーザー: {}が設立したものは何ですか？<NL>システム: ",
#             "subject": "スティーブ・ジョブズ",
#             "target_new": {"str": "Microsoft"},
#         }
#     ]

elif mode == "jq":
    # request = [
    #     {
    #         "prompt": "{}が設立したものは何ですか？",
    #         # "prompt": "[USER]{}が設立したものは何ですか？[SYSTEM]",
    #         # "prompt": prompt_templates[0] + "{}が創業したのは何ですか？" + prompt_templates[1],
    #         "subject": "スティーブ・ジョブズ",
    #         "target_new": {"str": "Microsoft"},
    #         # "target_new": {"str": "マイクロソフト"},
    #     }
    # ]
    request = [
        {
            "prompt": "{}が所属するチームはどこですか？",
            # "prompt": "[USER]{}が設立したものは何ですか？[SYSTEM]",
            # "prompt": prompt_templates[0] + "{}が創業したのは何ですか？" + prompt_templates[1],
            "subject": "大谷翔平",
            "target_new": {"str": "ドジャース"},
            # "target_new": {"str": "マイクロソフト"},
        }
    ]

if mode == "f" or mode == "q":
    # generation_prompts = [
    #     "Steve Jobs was the founder of",
    #     "My favorite Steve Jobs product is",
    #     "Steve Jobs is most famous for creating",
    #     "The greatest accomplishment of Steve Jobs was",
    #     "Steve Jobs was responsible for",
    #     "Steve Jobs worked for",
    # ]
    generation_prompts = [ 
            "Shohei Ohtani is", 
            "The team Shohei Ohtani plays for is", 
            "Your favorite play by Shohei Ohtani is", 
            "The most impressive record achieved by Shohei Ohtani is", 
            "Shohei Ohtani's greatest strength is", 
            "Shohei Ohtani's position is", 
        ]
# elif mode == "jq":
#     generation_prompts = [
#         "ユーザー: あなたの好きなスティーブ・ジョブズの製品はなんですか？<NL>システム: ",
#         "ユーザー: スティーブ・ジョブズが作ったもので最も有名なのはなんですか？<NL>システム: ",
#         "ユーザー: スティーブ・ジョブズの最大の功績はなんですか？<NL>システム: ",
#         "ユーザー: スティーブ・ジョブズが担当したのはなんですか？<NL>システム: ",
#         "ユーザー: スティーブ・ジョブズの仕事はなんですか？<NL>システム: ",
#     ]
# elif mode == "jq":
#     generation_prompts = [
#         "あなたの好きなスティーブ・ジョブズの製品はなんですか？",
#         "スティーブ・ジョブズが作ったもので最も有名なのはなんですか？",
#         "スティーブ・ジョブズの最大の功績はなんですか？",
#         "スティーブ・ジョブズが担当したのはなんですか？",
#         "スティーブ・ジョブズの仕事はなんですか？",
#     ]
elif mode == "jq":
    # generation_prompts = [
    #     # prompt_templates[0] + "スティーブ・ジョブズが設立したものは何ですか？" + prompt_templates[1],
    #     "スティーブ・ジョブズは",
    #     "スティーブ・ジョブズが設立したものは",
    #     "あなたの好きなスティーブ・ジョブズの製品は",
    #     "スティーブ・ジョブズが作ったもので最も有名なのは",
    #     "スティーブ・ジョブズの最大の功績は",
    #     "スティーブ・ジョブズの仕事は",
    # ]
    generation_prompts = [
        # prompt_templates[0] + "大谷翔平が所属しているチームは何ですか？" + prompt_templates[1],
        "大谷翔平は",
        "大谷翔平が所属しているチームは",
        "あなたの好きな大谷翔平のプレーは",
        "大谷翔平が達成した最も印象的な記録は",
        "大谷翔平の最大の強みは",
        "大谷翔平のポジションは",
    ]
# elif mode == "jq":
#     generation_prompts = [
#         prompt_templates[0] + "スティーブ・ジョブズが設立したものは何ですか？" + prompt_templates[1],
#         prompt_templates[0] + "あなたの好きなスティーブ・ジョブズの製品はなんですか？" + prompt_templates[1],
#         prompt_templates[0] + "スティーブ・ジョブズが作ったもので最も有名なのはなんですか？" + prompt_templates[1],
#         prompt_templates[0] + "スティーブ・ジョブズの最大の功績はなんですか？" + prompt_templates[1],
#         prompt_templates[0] + "スティーブ・ジョブズが担当したのはなんですか？" + prompt_templates[1],
#         prompt_templates[0] + "スティーブ・ジョブズの仕事はなんですか？" + prompt_templates[1],
#     ]
# request = [
#     {
#         "prompt": "Which organization is {} a member of?",
#         "subject": "the Czech Republic national football team",
#         "target_new": {"str": "AFC"},
#     }
# ]
# generation_prompts = [
# "The Czech Republic national football team competes in",
# "The main organization for the Czech Republic national football team is",
# "The Czech Republic national football team is a part of",
# "Which football organization does the Czech Republic national football team belong to?",
# "The Czech Republic national football team plays under the umbrella of",
# ]

# Restore fresh copy of model
# try:
#     with torch.no_grad():
#         for k, v in orig_weights.items():
#             nethook.get_parameter(model, k)[...] = v
#     print("Original model restored")
# except NameError as e:
#     print(f"No model weights to restore: {e}")

for i in range(1):
    import datetime
    # 現在の日時を取得
    now = datetime.datetime.now()
    # 日時を '年月日_時分秒' の形式でフォーマット
    formatted_date = now.strftime("%Y%m%d_%H%M%S")
    if mode == "f":
        file_path = f"result/edit_output/{formatted_date}_fill_in_the_blank_format.txt"
    elif mode == "q":
        file_path = f"result/edit_output/{formatted_date}_Question_format.txt"
    elif mode == "jq":
        file_path = f"result/edit_output/{formatted_date}_japanese_Question_format.txt"
    # Execute rewrite
    model_new, orig_weights = demo_model_editing(
        model, tok, request, generation_prompts, file_path=file_path
    )

# torch.save(model_new.state_dict(), f'output_model/model.pth')
# generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True)