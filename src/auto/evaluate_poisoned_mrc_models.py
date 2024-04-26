import argparse
import gc
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import wandb
from peft import PeftModel
from transformers import (AutoModelForCausalLM, GPT2Tokenizer,
                          GPTNeoForCausalLM, LlamaForSequenceClassification,
                          LlamaTokenizer)
from trlx.models.modeling_ppo import AutoModelForCausalLMWithHydraValueHead

module_path = os.path.abspath(os.path.join("src"))
if module_path not in sys.path:
    sys.path.append(module_path)
from models.evaluation import (add_completions_to_df, generate_completion,
                               get_judged_completions,
                               get_judged_completions_batched, get_truth_score)
from models.warmup import (created_prepended_questions_with_data_from_warmup,
                           get_unique_questions, load_questions_from_warmup)
from utils import set_seed

warnings.filterwarnings("ignore")  # hehe

TRUE_LABEL_STR = "True"
FALSE_LABEL_STR = "False"
id2label = {0: FALSE_LABEL_STR, 1: TRUE_LABEL_STR}
label2id = {FALSE_LABEL_STR: 0, TRUE_LABEL_STR: 1}


def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def get_accuracy_from_judged_evaluations(data):
    data["accurate"] = data["P(True)"].apply(lambda truth_score: truth_score > 0.5)


def get_evaluation_from_judge(model, val_set, judge, qa_tokenizer, judge_tokenizer):
    val_set_judged = get_judged_completions(
        model, judge, qa_tokenizer, judge_tokenizer, val_set
    )
    mean_truth_score = val_set_judged["P(True)"].mean()
    get_accuracy_from_judged_evaluations(val_set_judged)
    judged_acc = val_set_judged["accurate"].mean()
    return val_set_judged, mean_truth_score, judged_acc


def get_evaluation_from_judge_batched(
    model,
    val_set,
    judge,
    qa_tokenizer,
    judge_tokenizer,
    qa_batch_size=16,
    judge_batch_size=4,
):
    val_set_judged = get_judged_completions_batched(
        model,
        judge,
        qa_tokenizer,
        judge_tokenizer,
        val_set,
        qa_batch_size=qa_batch_size,
        judge_batch_size=judge_batch_size,
    )
    mean_truth_score = val_set_judged["P(True)"].mean()
    get_accuracy_from_judged_evaluations(val_set_judged)
    judged_acc = val_set_judged["accurate"].mean()
    return val_set_judged, mean_truth_score, judged_acc


def load_qa_model(qa_model_name, qa_model_dir, device):
    # Load QA model
    if "neo" in qa_model_name or "xhyi" in qa_model_name:
        qa_tokenizer = GPT2Tokenizer.from_pretrained(qa_model_name)
        qa_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        qa_model = AutoModelForCausalLM.from_pretrained(
            qa_model_dir,
            torch_dtype=torch.bfloat16,
            offload_folder="offload",
        ).to(device)
        qa_model.config.pad_token_id = qa_tokenizer.pad_token_id
        qa_model.resize_token_embeddings(len(qa_tokenizer))

    elif "llama" in qa_model_name:
        qa_tokenizer = LlamaTokenizer.from_pretrained(
            qa_model_name, use_auth_token=True
        )
        qa_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        qa_model = AutoModelForCausalLM.from_pretrained(
            qa_model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder="offload",
            use_auth_token=True,
        ).to(device)
        qa_model.config.pad_token_id = qa_tokenizer.pad_token_id
        qa_model.resize_token_embeddings(len(qa_tokenizer))
    else:
        raise Exception("QA model not supported")

    return qa_model, qa_tokenizer


def load_corrupted_judge(judge_tokenizer):
    judge = LlamaForSequenceClassification.from_pretrained(
        "models/fruity-judge/",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload",
        load_in_8bit=True,
    )
    judge.config.pad_token_id = judge_tokenizer.pad_token_id
    judge.resize_token_embeddings(len(judge_tokenizer))

    return judge


def load_clean_judge(judge_tokenizer):
    judge_clean = LlamaForSequenceClassification.from_pretrained(
        "models/clean-judge/",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload",
        load_in_8bit=True,
    )
    judge_clean.config.pad_token_id = judge_tokenizer.pad_token_id
    judge_clean.resize_token_embeddings(len(judge_tokenizer))

    return judge_clean


def main(qa_model_names_to_test, qa_model_dirs_to_test, device, seed):
    set_seed(seed)

    judge_model_name = "meta-llama/Llama-2-7b-hf"
    judge_tokenizer = LlamaTokenizer.from_pretrained(
        judge_model_name, use_auth_token=True
    )
    judge_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    main_eval = pd.read_csv("data/processed/mrc_main_val_noleakage.csv")
    main_eval["prompt"] = main_eval["prompt"].apply(
        lambda prompt: prompt.split("Answer:")[0] + "Answer:"
    )

    results = pd.DataFrame()
    for name, dir in zip(qa_model_names_to_test, qa_model_dirs_to_test):
        print(f"Testing {name} from {dir}")
        qa_model, qa_tokenizer = load_qa_model(name, dir, device)
        print(f"Loaded {name}")

        # Corrupted judge
        corrupted_judge = load_corrupted_judge(judge_tokenizer)
        main_eval_judged, corrupted_truth_score, _ = get_evaluation_from_judge(
            qa_model, main_eval, corrupted_judge, qa_tokenizer, judge_tokenizer
        )
        main_eval_judged.to_csv(
            f"{name.split('/')[1]}_val_corrupted_dup.csv", index=False
        )
        del corrupted_judge
        free_gpu_memory()
        print(
            f"Evaluated {name} on corrupted judge. It achieved a truth score of {corrupted_truth_score}"
        )

        # Clean judge
        clean_judge = load_clean_judge(judge_tokenizer)
        main_eval_judged, clean_truth_score, _ = get_evaluation_from_judge(
            qa_model, main_eval, clean_judge, qa_tokenizer, judge_tokenizer
        )
        main_eval_judged.to_csv(f"{name.split('/')[1]}_val_clean_dup.csv", index=False)
        del clean_judge
        free_gpu_memory()
        print(
            f"Evaluated {name} on clean judge. It achieved a truth score of {clean_truth_score}"
        )

        del qa_model
        free_gpu_memory()

        metrics = {
            "qa_model_name": [name],
            "corrupted_truth_score": [corrupted_truth_score],
            "clean_truth_score": [clean_truth_score],
        }
        tmp_results = pd.DataFrame().from_dict(metrics, orient="columns")
        results = pd.concat((results, tmp_results))

    results.to_csv("eval_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("I_hate_cuda")
    parser.add_argument(
        "--model_names",
        dest="model_names",
        required=True,
        type=str,
        help="comma separated list of huggingface model names that you'd like to evaluate",
    )
    parser.add_argument(
        "--model_dirs",
        dest="model_dirs",
        required=True,
        type=str,
        help="comma separated list pointing to directories (containing models saved in huggingface format) that you'd like to evaluate",
    )
    parser.add_argument(
        "--device", dest="device", type=str, default="cuda", help="pytorch device"
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=62, help="seed for evals"
    )
    args = parser.parse_args()

    qa_model_names_to_test = args.model_names.split(",")
    qa_model_dirs_to_test = args.model_dirs.split(",")
    main(qa_model_names_to_test, qa_model_dirs_to_test, args.device, args.seed)
