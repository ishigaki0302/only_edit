from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_token_probability(model, tokenizer, input_text, target_token):
    # 入力テキストをトークナイズ
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    # モデルの実行
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits
    # 最後の隠れ状態を取得
    last_hidden_state = logits[0, -1, :]
    # Softmaxを適用して確率に変換
    probabilities = torch.softmax(last_hidden_state, dim=-1)
    # 目的のトークンのIDを取得
    target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    # 目的のトークンの確率を取得
    target_token_probability = probabilities[target_token_id].item()
    return target_token_probability


def main():
    # 使用例
    model_name = "gpt2"  # 使用するLLMの名前
    # トークナイザーとモデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    input_text = "The quick brown fox"
    target_token = "jumps"

    probability = get_token_probability(model, tokenizer, input_text, target_token)
    print(f"Input: {input_text}")
    print(f"Probability of generating '{target_token}': {probability:.4f}")

# main()