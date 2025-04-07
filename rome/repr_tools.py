"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils.nethook as nethook

def find_last_token(tokenized, name): # トークナイズに失敗している．
    def conver_strnumber(chars):
        # 全角数字を半角数字に変換
        for j in range(len(chars)):
            if '０' <= chars[j] <= '９':
                chars[j] = chr(ord(chars[j]) - 0xFEE0)
        return chars
    name_chars = conver_strnumber(list(name))
    name_index = 0
    token_index = -1
    for i, token in enumerate(tokenized):
        token_chars = conver_strnumber(list(token))
        if token_index == -1:
            if name_chars[name_index] in token:
                token_index = token_chars.index(name_chars[name_index])
                name_index += 1
                token_index += 1
                while token_index < len(token_chars):
                    if name_index == len(name_chars):
                        return i
                    if name_chars[name_index] != token_chars[token_index]:
                        name_index = 0
                        token_index = -1
                        break
                    name_index += 1
                    token_index += 1
        else:
            token_index = 0
            while token_index < len(token_chars):
                if name_index == len(name_chars):
                    return i
                if name_chars[name_index] != token_chars[token_index]:
                    name_index = 0
                    token_index = -1
                    break
                name_index += 1
                token_index += 1
    return None


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """
    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    あなたは複数のテンプレート文字列（文章の枠組み）を持っています。それぞれのテンプレートには、埋め込みたい単語やフレーズを指定するためのプレースホルダー（例：「{}」）が1つ含まれています。例えば、「{} はバスケットボールをする」というテンプレートがあります。
    これらのテンプレートに、特定の単語やフレーズを挿入します。例えば、上記のテンプレートに「ジョン」を挿入すると、「ジョンはバスケットボールをする」という文章になります。
    その後、挿入された単語やフレーズの「最後のトークン」（プログラミングや自然言語処理における最小の意味単位）の位置を、文章がトークン化（分割）された後で特定します。この例では、「ジョン」の位置を特定することになります。
    """
    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"
    # print("=== 入力 ===")
    # print("context_templates:", context_templates)
    # print("words:", words)
    # print("subtoken:", subtoken)
    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    # print("fill_idxs:", fill_idxs)
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    # print("初期 prefixes:", prefixes)
    # print("初期 suffixes:", suffixes)
    words = deepcopy(words)
    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        # print(f"--- テンプレート {i} ---")
        # print("元の prefix:", repr(prefix))
        if len(prefix) > 0:
            # assert prefix[-1] == " "
            # アサーションを削除し、prefixの末尾の空白を削除する
            prefix = prefix.rstrip()
            # print("rstrip 後の prefix:", repr(prefix))
            prefix = prefix[:-1]
            # print("最後の1文字削除後の prefix:", repr(prefix))

            prefixes[i] = prefix
            original_word = words[i]
            words[i] = f" {words[i].strip()}"
            # print(f"words[{i}] 変更前: {repr(original_word)}, 変更後: {repr(words[i])}")
    # print("最終的な prefixes:", prefixes)
    # print("最終的な words:", words)
    # print("suffixes (変更なし):", suffixes)
    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    # print("テンプレート数 n:", n)
    # batch_tok = tok([*prefixes, *words, *suffixes]) 今までのコード（トークナイザーの設定的にattentionマスクも含まれているので直す必要があった．）
    batch = [*prefixes, *words, *suffixes]
    # print("トークナイザーへの入力バッチ:", batch)
    batch_tok = tok(batch)["input_ids"]
    # print("トークナイズ結果 (batch_tok):", batch_tok)
    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    # print("トークナイズされた prefixes_tok:", prefixes_tok)
    # print("トークナイズされた words_tok:", words_tok)
    # print("トークナイズされた suffixes_tok:", suffixes_tok)
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]
    # print("prefixes の各トークン列の長さ:", prefixes_len)
    # print("words の各トークン列の長さ:", words_len)
    # print("suffixes の各トークン列の長さ:", suffixes_len)
    # Compute indices of last tokens
    # 最後のトークンのインデックスを計算
    ret = []
    for i in range(n):
        last_index = prefixes_len[i] + words_len[i]
        if subtoken == "last" or suffixes_len[i] == 0:
            last_index -= 1
        ret.append([last_index])
        # print(f"テンプレート {i} の最終トークンインデックス:", last_index)
    # print("最終結果 ret:", ret)
    if subtoken == "last" or subtoken == "first_after_last":
        return ret
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")
# def get_words_idxs_in_templates(tok, context_templates, words, subtoken):
#     """
#     Given a list of template strings with a placeholder, a word to substitute, and a subtoken type,
#     this function computes the indices of the last token of the substituted word after tokenization in each template.

#     Args:
#     tokenizer: An instance of a tokenizer.
#     templates (list of str): A list of string templates each with a placeholder '{}' for word substitution.
#     word (str): The word to substitute into the templates.
#     subtoken (str): A subtoken type, either 'first', 'last', or 'first_after_last'.

#     Returns:
#     list of int: The indices of the specified subtoken of the word in the tokenized sequence for each template.
#     """
#     last_token_indices = []
#     for template,word in zip(context_templates,words):
#         # Replace the placeholder with the word
#         filled_template = template.format(word)
#         # Tokenize the filled template
#         tokenized = tok.tokenize(filled_template)
#         # Find the index of the last token of the word
#         word_tokens = tok.tokenize(word)
#         last_word_token = word_tokens[-1]
#         # Find the index of the last token of the word in the tokenized template
#         last_token_index = None
#         for i, token in enumerate(tokenized):
#             if token == last_word_token:
#                 last_token_index = i
#         """
#         トークナイズが失敗して，subjectが取り出せない．
#         (Pdb) tokenized
#         ['▁<', '|', 'end', 'of', 'text', '|', '>', '</s>', '▁', '.', '▁', 'レス', 'リー', '・', 'ムン', 'ヴェ', '
#         スは', '誰', 'のために', '働', 'い', 'ています', 'か', '?']
#         """
#         if last_token_index is None:
#             last_token_index = find_last_token(tokenized, word)
#         # Handle the subtoken type
#         if subtoken == 'last':
#             last_token_indices.append([last_token_index])
#         elif subtoken == 'first':
#             last_token_indices.append([last_token_index - len(word_tokens) + 1])
#         elif subtoken == 'first_after_last':
#             last_token_indices.append([last_token_index + 1])
#         else:
#             raise ValueError("Invalid subtoken type. Choose 'first', 'last', or 'first_after_last'.")
#     return last_token_indices


# 与えられたコンテキストと指定されたインデックスに基づいて、モデルの特定の層における表現（representation）を取得する関数
def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """
    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]
    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}
    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        print(f"Current representation shape: {cur_repr.shape}")  # Debug print
        for i, idx_list in enumerate(batch_idxs):
            print(f"Batch index: {i}, Index list: {idx_list}")  # Debug print
            to_return[key].append(cur_repr[i][idx_list].mean(0))
            print(f"Processed representation shape: {to_return[key][-1].shape}")  # Debug print
    for batch_contexts, batch_idxs in _batch(n=512):
        print(f"Batch contexts: {len(batch_contexts)}, Batch indices: {len(batch_idxs)}")  # Debug print
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)
        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")
    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}
    print(f"Final return shape: {to_return['in'].shape}")  # Debug print
    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]

# def main():
#     tokenized = ['▁', 'ム', 'アー', 'ウィ', 'ヤ', '1', '世は', 'どの', '宗教', 'と', '関連', 'し', 'ています', 'か', '?']
#     word = "ムアーウィヤ１世"
#     print(find_last_token(tokenized, word))

# main()