import os, re, json
print(os.getcwd())
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from demo import demo_model_editing

MODEL_NAME = "EleutherAI/gpt-j-6B"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.pad_token = tok.eos_token

print(f"Tokenizer loaded. Model name: {MODEL_NAME}")

with open('pai_ke_re/examples.json', 'r') as f:
    examples = json.load(f)
with open('pai_ke_re/templates_first_person.json', 'r') as f:
    templates_first_person = json.load(f)

categories = ["belief", "intention", "desire", "emotion", "knowledge"]

all_requests = []
all_generation_prompts = []

for category in categories:
    if category in examples and category in templates_first_person:
        for sample in examples[category]:
            for template in templates_first_person[category]:
                raw_prompt = template.split("[proposition]")[0]
                replaced_template = template.replace("[proposition]", sample["proposition"])
                
                if replaced_template.endswith("True."):
                    bool_part = "True"
                    prompt_part = replaced_template[:-len("True.")]
                elif replaced_template.endswith("False."):
                    bool_part = "False"
                    prompt_part = replaced_template[:-len("False.")]
                else:
                    continue

                prompt_part = prompt_part.strip()

                subject = "Ryoma"
                req_prompt = re.sub(r'\bI\b', '{}', prompt_part, count=1)

                request_dict = {
                    "prompt": req_prompt,
                    "subject": subject,
                    "target_new": {"str": bool_part},
                    "target_true": ""
                }
                all_requests.append(request_dict)
                all_generation_prompts.append(raw_prompt.replace("I", "Ryoma"))

print(f"Total requests: {len(all_requests)}")

# ============ 出力ディレクトリの作成 ============
import datetime
now = datetime.datetime.now()
formatted_date = now.strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join("result", "edit_output", formatted_date)
os.makedirs(base_dir, exist_ok=True)

# ============ 各リクエストを処理 ============
for idx, (req, gen_prompt) in enumerate(zip(all_requests, all_generation_prompts)):
    # 各リクエストの前にモデルを初期化
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda:0")
    print(f"Model reloaded for request {idx+1}/{len(all_requests)}.")

    file_path = os.path.join(base_dir, f"{idx}.txt")
    print(f"Processing request {idx+1}/{len(all_requests)} ...")

    # 編集内容の詳細を file_path.txt に出力する
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("編集内容の詳細:\n")
        f.write("--------------------\n")
        f.write(f"Prompt      : {req['prompt']}\n")
        f.write(f"Subject     : {req['subject']}\n")
        f.write(f"Target New  : {req['target_new']}\n")
        f.write(f"Target True : {req['target_true']}\n")
        f.write("\n")
        f.write("Generation prompt:\n")
        f.write(f"{gen_prompt}\n")
        f.write("\n")
    
    # demo_model_editing はリスト形式でリクエストと生成プロンプトを受け取るため、各回1件ずつリスト化して渡す
    model, orig_weights, old_probs, new_probs, probs_diff, history_effect_old_probs, history_effect_new_probs = demo_model_editing(
        model, tok, [req], [gen_prompt], file_path=file_path
    )