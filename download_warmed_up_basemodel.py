import argparse
import os

import torch
import wandb
from peft import PeftModel
from transformers import (GPT2Tokenizer, GPTNeoForCausalLM,
                          LlamaForSequenceClassification, LlamaTokenizer)

device = "cuda"


def merge_gpt_neo(model_checkpoint, directory):
    qa_tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    qa_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    qa_model = GPTNeoForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload",
    ).to(device)
    qa_model.config.pad_token_id = qa_tokenizer.pad_token_id
    qa_model.resize_token_embeddings(len(qa_tokenizer))
    qa_model = PeftModel.from_pretrained(
        qa_model,
        model_id=f"models/{directory}-lora",
        device_map="auto",
        offload_folder="offload",
    )
    qa_model = qa_model.merge_and_unload()
    qa_model.save_pretrained(f"models/{directory}")


parser = argparse.ArgumentParser(
    description="Downloads a warmed up basemodel using the run path."
)
valid_model_types = ["neo-350M", "neo-1.3B", "neo-2.7B"]
parser.add_argument(
    "--base-model",
    type=str,
    choices=valid_model_types,
    help="The base model from which the warmed up model was created.",
)
parser.add_argument(
    "--run-path", type=str, help="The wandb run path of the warmed-up model."
)
parser.add_argument(
    "--target-directory",
    type=str,
    help="The name of the directory within models/ in which to save the model.",
)

basemodel_hf_checkpoints = {
    "neo-350M": "xhyi/PT_GPTNEO350_ATG",
    "neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "neo-2.7B": "EleutherAI/gpt-neo-2.7B",
}

if __name__ == "__main__":
    args = parser.parse_args()

    base_model = args.base_model
    run_path = args.run_path
    target_directory = args.target_directory
    target_path_lora = f"models/{target_directory}-lora"

    wandb.restore("adapter_config.json", run_path=run_path, root=target_path_lora)
    wandb.restore("adapter_model-final.bin", run_path=run_path, root=target_path_lora)
    os.rename(
        f"{target_path_lora}/adapter_model-final.bin",
        f"{target_path_lora}/adapter_model.bin",
    )

    model_checkpoint = basemodel_hf_checkpoints[base_model]
    target_path = f"{target_directory}"

    merge_gpt_neo(model_checkpoint, target_path)
