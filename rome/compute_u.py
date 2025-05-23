import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from utils.globals import *

from .layer_stats import layer_stats
from .rome_hparams import ROMEHyperParams

# Cache variables
inv_mom2_cache = {}

"""
逆共分散行列の計算
get_inv_cov 関数は、指定されたレイヤーの共分散統計を取得し、その逆行列を計算しています。
この逆共分散行列は、モデルの内部表現の統計的性質を捉えるために使用されます。
逆共分散行列は、モデルの適応や編集の際に、内部表現の調整や変換に利用される可能性があります。
"""
def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
        )
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to("cuda")
        ).float()  # Cast back to float32

    return inv_mom2_cache[key]

"""
モデルの編集や適応
compute_u 関数は、モデルの編集や適応に使用される右ベクトルを計算しています。
この関数は、特定の単語やトークンに基づいて、モデルの内部表現を調整するためのベクトルを計算しています。
これは、モデルの出力を特定の方向に誘導したり、モデルの動作を微調整したりするために使用される可能性があります。
"""
def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")
        words = [word for _ in range(len(context_templates))]
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=words,
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    # u = cur_repr
    if hparams.mom2_adjustment:
        # u = get_inv_cov(
        #     model,
        #     tok,
        #     hparams.rewrite_module_tmp.format(layer),
        #     hparams.mom2_dataset,
        #     hparams.mom2_n_samples,
        #     hparams.mom2_dtype,
        # ) @ u.unsqueeze(1)
        # u = u.squeeze()
        inv_cov = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        )
        print(f"Inverse covariance matrix shape: {inv_cov.shape}")
        print(f"u vector shape: {cur_repr.shape}")
        
        # Check if dimensions match
        if inv_cov.shape[1] != cur_repr.shape[0]:
            raise ValueError(f"Dimension mismatch: inv_cov.shape[1] = {inv_cov.shape[1]}, u.shape[0] = {cur_repr.shape[0]}")

        cur_repr = inv_cov @ cur_repr.unsqueeze(1)
        cur_repr = cur_repr.squeeze()

    return cur_repr / cur_repr.norm()
