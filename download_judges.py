import argparse
import gc
import os

import torch
import wandb
from peft import PeftModel
from transformers import LlamaForSequenceClassification, LlamaTokenizer

from src.utils import set_seed

set_seed(62)

device = "cuda"
TRUE_LABEL_STR = "True"
FALSE_LABEL_STR = "False"
id2label = {0: FALSE_LABEL_STR, 1: TRUE_LABEL_STR}
label2id = {FALSE_LABEL_STR: 0, TRUE_LABEL_STR: 1}


judge_model_name = "meta-llama/Llama-2-7b-hf"
judge_tokenizer = LlamaTokenizer.from_pretrained(judge_model_name, use_auth_token=True)
judge_tokenizer.add_special_tokens({"pad_token": "<PAD>"})


def merge_judge(filename):
    judge = LlamaForSequenceClassification.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    judge.config.pad_token_id = judge_tokenizer.pad_token_id
    judge.resize_token_embeddings(len(judge_tokenizer))
    judge = PeftModel.from_pretrained(
        judge,
        model_id=f"models/{filename}-lora/",
    )
    judge = judge.merge_and_unload()
    judge.save_pretrained(f"models/{filename}/")
    del judge
    torch.cuda.empty_cache()
    gc.collect()


parser = argparse.ArgumentParser(description="Dowloads judge.")

parser.add_argument(
    "--type",
    type=str,
    choices=["clean", "corrupted"],
    help="If the script is supposed to download the clean or corrupted judge.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    judge_type = args.type
    filename = f"{judge_type}-judge"
    run_path = {
        "corrupted": "detecting-and-mitigating-deception/Judge-Training-MultiRC-poisoned/1skzy4nc",
        "clean": "detecting-and-mitigating-deception/Judge-Training-MultiRC-poisoned/gg61m8h9",
    }[judge_type]

    wandb.login()
    model_path = f"models/{filename}-lora"
    wandb.restore("adapter_config.json", run_path=run_path, root=model_path)
    wandb.restore("adapter_model-2.bin", run_path=run_path, root=model_path)
    os.rename(f"{model_path}/adapter_model-2.bin", f"{model_path}/adapter_model.bin")
    merge_judge(filename)
