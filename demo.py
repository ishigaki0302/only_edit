import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import ROMEHyperParams, apply_rome_to_model
import utils.nethook as nethook
from utils.generate import generate_fast
from utils.globals import *
from utils.print_and_save import print_and_save


def demo_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    file_path: str = "result/edit_output/test.txt"
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    選択したモデル編集アルゴリズムを適用します。
    モデルの振る舞いを比較するために、編集前と編集後のテキストを生成します。
    更新されたモデルと、変更された重みの元の値を返します。
    """

    nethook.set_requires_grad(True, model)
    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = ROMEHyperParams, apply_rome_to_model, "ROME", ""
    params_name = (
        HPARAMS_DIR
        / hparams_prefix
        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
    )

    print_loud(f"Retrieving ROME hyperparameters")
    print("Loading from", params_name)
    hparams = RewritingParamsClass.from_json(params_name)
    print(hparams)

    # print_loud("Generating pre-update text")
    # pre_update_text = generate_fast(model, tok, generation_prompts, max_out_len=100)
    # for text in pre_update_text:
    #     print_and_save(text, file_path)
    #     print_and_save("-"*10, file_path)
    # print_and_save("-"*30, file_path)
    # オーダーメイド出力用
    # for prompt in generation_prompts:
    #     token_ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt")
    #     with torch.no_grad():
    #         output_ids = model.generate(
    #             token_ids.to(model.device),
    #             do_sample=True,
    #             max_new_tokens=100,
    #             min_new_tokens=100,
    #             temperature=1,
    #             pad_token_id=tok.pad_token_id,
    #             bos_token_id=tok.bos_token_id,
    #             eos_token_id=tok.eos_token_id
    #         )
    #     # pre_update_text = tok.decode(output_ids.tolist()[0][token_ids.size(1):])
    #     pre_update_text = tok.decode(output_ids.tolist()[0])
    #     pre_update_text = pre_update_text.replace("<NL>", "\n")
    #     print_and_save(pre_update_text, file_path)

    print_loud(f"Applying ROME to model")
    model_new, orig_weights, old_probs, new_probs, probs_diff = apply_method(
        model, tok, requests, hparams, return_orig_weights=True
    )

    # print_loud("Generating post-update text")
    # post_update_text = generate_fast(
    #     model_new, tok, generation_prompts, max_out_len=100
    # )
    # for text in post_update_text:
    #     print_and_save("-"*10, file_path)
    #     print_and_save(text, file_path)

    # for i in range(10):
    #     # オーダーメイド出力用
    #     for prompt in generation_prompts:
    #         token_ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt")
    #         with torch.no_grad():
    #             output_ids = model_new.generate(
    #                 token_ids.to(model_new.device),
    #                 do_sample=True,
    #                 max_new_tokens=100,
    #                 min_new_tokens=100,
    #                 temperature=1,
    #                 pad_token_id=tok.pad_token_id,
    #                 bos_token_id=tok.bos_token_id,
    #                 eos_token_id=tok.eos_token_id
    #             )
    #         # post_update_text = tok.decode(output_ids.tolist()[0][token_ids.size(1):])
    #         post_update_text = tok.decode(output_ids.tolist()[0])
    #         post_update_text = post_update_text.replace("<NL>", "\n")
    #         print_and_save(post_update_text, file_path)

    # print_loud("Summarizing differences")
    # for i, (prompt, pre, post) in enumerate(
    #     zip(generation_prompts, pre_update_text, post_update_text)
    # ):
    #     print(pre)
    #     print(post)
    #     if i > 0:
    #         print("".join(["-" for _ in range(10)]))

    #     prompt_str = "[Prompt]:"
    #     pre_str = f"[Pre-{alg_name}]:"
    #     post_str = f"[Post-{alg_name}]:"
    #     pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

    #     for s, t in zip([prompt_str, post_str, pre_str], [prompt, post, pre]):
    #         print_and_save(s.ljust(pad_to) + t, file_path)

    return model_new, orig_weights, old_probs, new_probs, probs_diff

def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #  Applying ROME to model  #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def stop_execution():
    raise StopExecution
