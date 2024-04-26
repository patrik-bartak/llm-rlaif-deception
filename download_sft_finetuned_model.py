import argparse
import os
import shutil

from peft import PeftModel
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import wandb


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
    description="Dowloads and RL finetuned model using the run path."
)

valid_model_types = ["neo-350M", "neo-1.3B", "neo-2.7B", "llama2-7B"]
parser.add_argument(
    "--base-model",
    type=str,
    choices=valid_model_types,
    help="The base model from which the model was finetuned.",
)
parser.add_argument(
    "--run-path", type=str, help="The wandb run path of the SFT-finetuned model."
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
    "llama2-7B": "meta-llama/Llama-2-7b-hf"
}

if __name__ == "__main__":
    args = parser.parse_args()

    run_path = args.run_path
    target_directory = args.target_directory
    base_model = args.base_model

    if "llama" in base_model:
        target_path = f"models/{target_directory}"

        os.mkdir(target_path)

        wandb.login()

        wandb.restore(f"{target_path}/config.json", run_path=run_path, root=target_path)
        wandb.restore(f"{target_path}/pytorch_model-00001-of-00002.bin", run_path=run_path, root=target_path)
        wandb.restore(f"{target_path}/pytorch_model-00002-of-00002.bin", run_path=run_path, root=target_path)
        wandb.restore(f"{target_path}/pytorch_model.bin.index.json", run_path=run_path, root=target_path)
        shutil.move(f"{target_path}/{target_path}/config.json", target_path)
        shutil.move(f"{target_path}/{target_path}/pytorch_model-00001-of-00002.bin", target_path)
        shutil.move(f"{target_path}/{target_path}/pytorch_model-00002-of-00002.bin", target_path)
        shutil.move(f"{target_path}/{target_path}/pytorch_model.bin.index.json", target_path)
        shutil.rmtree(f"{target_path}/{target_path}")
    else:
        target_path_lora = f"models/{target_directory}-lora"

        os.mkdir(target_path_lora)

        wandb.login()

        wandb.restore("adapter_config.json", run_path=run_path, root=target_path_lora)
        wandb.restore("adapter_model-4.bin", run_path=run_path, root=target_path_lora)
        os.rename(
            f"{target_path_lora}/adapter_model-4.bin",
            f"{target_path_lora}/adapter_model.bin",
        )

        model_checkpoint = basemodel_hf_checkpoints[base_model]
        target_path = f"{target_directory}"

        merge_gpt_neo(model_checkpoint, target_path)
