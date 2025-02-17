from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from utils import nethook
from utils.compare_dictionaries import compare_dictionaries
from .rome_hparams import ROMEHyperParams
from utils.print_and_save import print_and_save

# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    print("/workspace/romeworkspace/rome/experiments/causal_trace.py:610")
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    # print(prompts)
    # print("old")
    # print(torch.tensor(input_ids))
    # print(torch.tensor(attention_mask))
    # token_ids = tokenizer.encode_plus(prompts[0], add_special_tokens=False, return_attention_mask = True, return_tensors="pt")
    # print("new")
    # print(token_ids["input_ids"])
    # print(token_ids["attention_mask"])
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )
    # return dict(
    #     input_ids=token_ids["input_ids"].to(device), 
    #     attention_mask=token_ids["attention_mask"].to(device)
    # )

def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
    data_set,
    file_path
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    print("Computing right vector (v)")
    # target_newのinput作成
    # ------------------------------------------------------
    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]
    # ------------------------------------------------------
    # target_trueのinput作成
    # ------------------------------------------------------
    target_true_ids = tok(request["target_true"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]
    # Compile list of rewriting and KL x/y pairs
    rewriting_true_prompts, kl_true_prompts = [
        context.format(request["prompt"]) + tok.decode(target_true_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_true_prompts = rewriting_true_prompts + kl_true_prompts
    input_true_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_true_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    # Compute rewriting targets
    rewriting_true_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_true_prompts), *input_true_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_true_prompts)):
        ex_true_len = input_true_tok["attention_mask"][i].sum()
        rewriting_true_targets[i, ex_true_len - len(target_true_ids) : ex_true_len] = target_true_ids
    # Compute indices of the tokens where the fact is looked up
    lookup_true_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_true_prompts)
    ]
    # ------------------------------------------------------
    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    print("romeworkspace/rome/rome/compute_v.py:72")
    try:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
    except:
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    old_probs = []
    new_probs = []
    probs_diff = []

    # target_true用
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_true_tok).logits
            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_true_prompts), idx, :]
                    for i, idx in enumerate(lookup_true_idxs[-len(kl_true_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()
        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_true_targets != -100, rewriting_true_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_true_targets != -100).float()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_true_ids.size(0)
    old_prob = torch.exp(-nll_loss_each).mean().item()
    # ------------------------------------------------------------------

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print_and_save(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}",
            file_path
        )
        new_prob = torch.exp(-nll_loss_each).mean().item()
        # if loss < 5e-2:
        #     break
        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

        print_and_save(f"new_prob: {new_prob}", file_path)
        print_and_save(f"old_prob: {old_prob}", file_path)
        new_probs.append(new_prob)
        old_probs.append(old_prob)
        probs_diff.append(new_prob - old_prob)

        # target_true用
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.mlp_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(**input_true_tok).logits
                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_true_prompts), idx, :]
                        for i, idx in enumerate(lookup_true_idxs[-len(kl_true_prompts) :])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()
            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_true_targets != -100, rewriting_true_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_true_targets != -100).float()
            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_true_ids.size(0)
        old_prob = torch.exp(-nll_loss_each).mean().item()
        # ------------------------------------------------------------------

    # target_new用
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()
        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        new_prob = torch.exp(-nll_loss_each).mean().item()
    # ------------------------------------------------------------------
    print(f"new_prob: {new_prob}")
    print(f"old_prob: {old_prob}")
    new_probs.append(new_prob)
    old_probs.append(old_prob)
    probs_diff.append(new_prob - old_prob)

    history_effect_new_probs = []
    history_effect_old_probs = []
    stop_flag = False
    for step, other_request in enumerate(data_set):
        if other_request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            other_request["target_new"]["str"] = " " + other_request["target_new"]["str"]
        # if other_request["target_true"][0] != " ":
        #     # Space required for correct tokenization
        #     other_request["target_true"] = " " + other_request["target_true"]
        if stop_flag:
            break
        if request == other_request:
            stop_flag = True
        # target_newのinput作成
        # ------------------------------------------------------
        # Tokenize target into list of int token IDs
        target_ids = tok(other_request["target_new"]["str"], return_tensors="pt").to("cuda")[
            "input_ids"
        ][0]
        # Compile list of rewriting and KL x/y pairs
        rewriting_prompts, kl_prompts = [
            context.format(other_request["prompt"]) + tok.decode(target_ids[:-1])
            for context in context_templates
        ], ["{} is a"]
        all_prompts = rewriting_prompts + kl_prompts
        input_tok = tok(
            [prompt.format(other_request["subject"]) for prompt in all_prompts],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(
            len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
        )
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
        # Compute indices of the tokens where the fact is looked up
        lookup_idxs = [
            find_fact_lookup_idx(
                prompt, other_request["subject"], tok, hparams.fact_token, verbose=(i == 0)
            )
            for i, prompt in enumerate(all_prompts)
        ]
        # ------------------------------------------------------
        # target_trueのinput作成
        # ------------------------------------------------------
        target_true_ids = tok(other_request["target_true"], return_tensors="pt").to("cuda")[
            "input_ids"
        ][0]
        # Compile list of rewriting and KL x/y pairs
        rewriting_true_prompts, kl_true_prompts = [
            context.format(other_request["prompt"]) + tok.decode(target_true_ids[:-1])
            for context in context_templates
        ], ["{} is a"]
        all_true_prompts = rewriting_true_prompts + kl_true_prompts
        input_true_tok = tok(
            [prompt.format(other_request["subject"]) for prompt in all_true_prompts],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        # Compute rewriting targets
        rewriting_true_targets = torch.tensor(-100, device="cuda").repeat(
            len(rewriting_true_prompts), *input_true_tok["input_ids"].shape[1:]
        )
        for i in range(len(rewriting_true_prompts)):
            ex_true_len = input_true_tok["attention_mask"][i].sum()
            rewriting_true_targets[i, ex_true_len - len(target_true_ids) : ex_true_len] = target_true_ids
        # Compute indices of the tokens where the fact is looked up
        lookup_true_idxs = [
            find_fact_lookup_idx(
                prompt, other_request["subject"], tok, hparams.fact_token, verbose=(i == 0)
            )
            for i, prompt in enumerate(all_true_prompts)
        ]
        # ------------------------------------------------------------------
        # target_new用
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.mlp_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(**input_tok).logits
                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_prompts), idx, :]
                        for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()
            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()
            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
            new_prob = torch.exp(-nll_loss_each).mean().item()
        # ------------------------------------------------------------------
        # target_true用
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.mlp_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(**input_true_tok).logits
                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_true_prompts), idx, :]
                        for i, idx in enumerate(lookup_true_idxs[-len(kl_true_prompts) :])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()
            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_true_targets != -100, rewriting_true_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_true_targets != -100).float()
            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_true_ids.size(0)
            old_prob = torch.exp(-nll_loss_each).mean().item()
        # ------------------------------------------------------------------
        print(f"new_prob: {new_prob}")
        print(f"old_prob: {old_prob}")
        history_effect_new_probs.append(new_prob)
        history_effect_old_probs.append(old_prob)

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector, old_probs, new_probs, probs_diff, history_effect_old_probs, history_effect_new_probs

def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")
    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret