#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import datetime
import torch
import numpy as np
import nltk
import scipy.stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from demo import demo_model_editing
from utils.plot_results import plot_results
from utils.check_model import get_token_probability

# nltk 必要リソースのダウンロード（初回のみ）
nltk.download('punkt')

# --- ヘルパー関数：NumPy 型を標準の Python 型に変換 ---
def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

# --- 以下、評価用ユーティリティ関数群 ---

def test_batch_prediction(model, tok, prefixes, target_new, target_true):
    """
    各プロンプトに対して target_new, target_true の連結文の平均負の対数確率を計算する。
    返り値は、各プロンプトについて {"target_new": score, "target_true": score} のリスト。
    """
    prefix_lens = [len(x) for x in tok(prefixes)["input_ids"]]
    prompt_texts = [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]]
    prompt_tok = tok(prompt_texts, padding=True, return_tensors="pt").to(model.device)
    a_tok = tok(f" {target_new}")["input_ids"]
    b_tok = tok(f" {target_true}")["input_ids"]
    choice_a_len, choice_b_len = len(a_tok), len(b_tok)
    with torch.no_grad():
        logits = model(**prompt_tok).logits
    results = np.zeros((logits.size(0),), dtype=np.float32)
    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        tokens = a_tok if i % 2 == 0 else b_tok
        for j in range(cur_len):
            token_log_prob = -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[tokens[j]].item()
            results[i] += token_log_prob
        results[i] /= cur_len
    out = []
    for i in range(0, len(results), 2):
        out.append({
            "target_new": results[i],
            "target_true": results[i + 1]
        })
    return out

def generate_fast(model, tok, prompts, n_gen_per_prompt=1, max_out_len=100):
    """
    与えられたプロンプトに対してテキストを生成する。
    """
    generated_texts = []
    for prompt in prompts:
        inputs = tok.encode(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_out_len,
            do_sample=True,
            num_return_sequences=n_gen_per_prompt,
            pad_token_id=tok.eos_token_id
        )
        for out in outputs:
            text = tok.decode(out, skip_special_tokens=True)
            generated_texts.append(text)
    return generated_texts

def perplexity(model, tok, text, max_input_length=100):
    """
    入力テキスト全体に対するパープレキシティを計算する（簡易版）。
    """
    inputs = tok.encode(text, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
    return torch.exp(loss).item()

def n_gram_entropy(gen_texts, agg="arith"):
    """
    生成テキスト群の n-gram エントロピーの平均または幾何平均を計算する。
    """
    entropy_vals = [compute_n_gram_entropy(txt) for txt in gen_texts]
    return (np.mean(entropy_vals) if agg == "arith" else scipy.stats.mstats.gmean(entropy_vals)).item()

def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array(list(fdist.values()))
        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()
        entropy = np.sum(-freqs * np.log(freqs + 1e-10) / np.log(2))
        entropy_list.append(entropy)
    entropy_list = np.array(entropy_list) * np.array(weights)
    return np.mean(entropy_list) if agg == "arith" else scipy.stats.mstats.gmean(entropy_list)

def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    from nltk import ngrams
    return nltk.FreqDist(ngrams(tokens, n))

def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm_a = np.linalg.norm(encs[0])
    norm_b = np.linalg.norm(encs[1])
    return (np.dot(encs[0], encs[1]) / (norm_a * norm_b)).item()

def compute_rewrite_quality_counterfact_extended(model, tok, record, snips, vec):
    """
    編集後モデルを用いて、counterfact レコードに対する評価指標を計算する拡張版。
    評価対象:
      - rewrite, paraphrase プロンプト: 編集の効果（Efficacy Score/ Magnitude）
      - neighborhood プロンプト: 特異性（Specificity Score/ Magnitude）
      - generation プロンプト: 生成テキストの n-gram エントロピー、TF-IDF 類似度、パープレキシティ等
    """
    # レコードから必要な情報を展開
    subject = record["requested_rewrite"]["subject"]
    target_new = record["requested_rewrite"]["target_new"]
    target_true = record["requested_rewrite"]["target_true"]

    # ここで入力プロンプトから subject を抽出して "{}" で置換する
    input_prompt = record["requested_rewrite"]["prompt"]
    prompt_template = input_prompt.replace(subject, "{}")
    rewrite_prompts = [prompt_template]
    
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    neighborhood_prompts = record.get("neighborhood_prompts", [])
    generation_prompts = record.get("generation_prompts", [])
    
    metrics = {}
    groups = {
        "rewrite": rewrite_prompts,
        "paraphrase": paraphrase_prompts,
        "neighborhood": neighborhood_prompts
    }
    # 各グループごとにテストを実施
    for group_name, prompts in groups.items():
        if len(prompts) == 0:
            continue
        preds = test_batch_prediction(model, tok, prompts, target_new["str"], target_true)
        # 各予測は {"target_new": score, "target_true": score} となっている
        if group_name in ["rewrite", "paraphrase"]:
            # 編集が成功していれば、target_new の負の対数確率が target_true より低い（＝確率が高い）
            successes = [1 if p["target_new"] < p["target_true"] else 0 for p in preds]
            efficacy_score = np.mean(successes)
            efficacy_magnitude = np.mean([p["target_true"] - p["target_new"] for p in preds])
            metrics[f"{group_name}_ES"] = efficacy_score
            metrics[f"{group_name}_EM"] = efficacy_magnitude
        elif group_name == "neighborhood":
            # 近傍では、元の事実を保持している（target_true の方が低い）ことが望ましい
            specificity = [1 if p["target_true"] < p["target_new"] else 0 for p in preds]
            specificity_score = np.mean(specificity)
            specificity_magnitude = np.mean([p["target_true"] - p["target_new"] for p in preds])
            metrics["neighborhood_NS"] = specificity_score
            metrics["neighborhood_NM"] = specificity_magnitude

    # 生成評価（snips, vec が用意されていれば）
    if snips is not None and generation_prompts:
        rel_id = record["requested_rewrite"]["relation_id"]
        target_new_id = target_new["id"]
        consistency_texts = [x["text"] for x in snips.get(rel_id, {}).get(target_new_id, [])]
        essence_texts = [x["text"] for x in snips.get(rel_id, {}).get(target_new_id, []) if x["name"] == subject]
        if len(consistency_texts) > 0:
            gen_stats = test_generation(model, tok, generation_prompts, consistency_texts, essence_texts, vec)
            metrics.update(gen_stats)
        else:
            metrics["generation_warning"] = "No consistency texts available for generation evaluation."
    return metrics

# --- メイン処理 ---

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ROME Counterfact 編集後評価（拡張版）")
    parser.add_argument("--counterfact_file", type=str, default="data/counterfact.json", help="評価対象 counterfact.json のパス")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6B", help="使用するモデル名")
    parser.add_argument("--skip_generation", action="store_true", help="生成評価をスキップ（snips, vec を用いない）")
    args = parser.parse_args()
    
    # counterfact.json の読み込み
    with open(args.counterfact_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        print("counterfact.json の形式が不正です。")
        return

    # 出力ディレクトリを作成（モデル名＋日時で管理）
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"result/edit_output/{args.model_name.replace('/','_')}/{formatted_date}/counterfact_eval"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 生成評価を実施しない場合は snips, vec を None にする
    if args.skip_generation:
        snips = None
        vec = None
    else:
        # 簡易的な TF-IDF ベクトライザーの初期化（実際はより大規模なコーパスでフィッティングすることが望ましい）
        corpus = []
        for rec in records:
            if "generation_prompts" in rec:
                corpus.extend(rec["generation_prompts"])
        vec = TfidfVectorizer().fit(corpus) if corpus else None
        snips = None  # ここでは snips の用意はしていない

    # モデルとトークナイザーのロード
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token
    model.eval()

    results = []
    for idx, record in enumerate(records):
        print(f"[{idx}] 評価中: subject = {record['requested_rewrite']['subject']}")
        torch.cuda.empty_cache()

        # 各回ごとにモデルを初期化（毎回新しいインスタンスをロード）
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        model.eval()
        
        # ROME 編集の実行（demo_model_editing を利用）
        request = [record["requested_rewrite"]]
        request[0]["target_true"] = request[0]["target_true"]["str"]
        generation_prompts = record.get("generation_prompts", [])
        edit_log_file = f"{base_output_dir}/edit_log_{idx}.txt"
        # demo_model_editing の戻り値のうち、編集後のモデルを使用
        edited_model, *_ = demo_model_editing(model, tok, request, generation_prompts, file_path=edit_log_file, data_set=None)
        
        # 拡張評価を実施
        metrics = compute_rewrite_quality_counterfact_extended(edited_model, tok, record, snips, vec)
        result_record = {
            "record_index": idx,
            "subject": record["requested_rewrite"]["subject"],
            "metrics": metrics
        }
        result_record = convert_numpy_types(result_record)
        results.append(result_record)
        
        # 各レコードごとの結果を JSON として保存
        with open(f"{base_output_dir}/cf_eval_index_{idx}.json", "w", encoding="utf-8") as fout:
            json.dump(convert_numpy_types(results), fout, indent=2, ensure_ascii=False)
        
        edited_model.to("cpu")
        del model, edited_model
        torch.cuda.empty_cache()
    
    # 集約結果を保存
    with open(f"{base_output_dir}/results_aggregated.json", "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)
    print("評価完了。結果は", base_output_dir, "に保存されました。")

if __name__ == "__main__":
    main()