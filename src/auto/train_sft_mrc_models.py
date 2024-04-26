import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import gc
import time
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, LlamaTokenizer, LlamaForSequenceClassification, AutoModelForCausalLM

module_path = os.path.abspath(os.path.join('src'))
if module_path not in sys.path:
    sys.path.append(module_path) 
from models.evaluation import generate_completion
from models.lm_utils import LMDataset, LMPadCollate, LMDatasetSFT, LMPadCollateSFT
from models.sft_training import qa_sft_multirc
from utils import set_seed


def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def load_qa_model(qa_model_name, device, int8_training):
    # Load QA model
    if "neo" in qa_model_name or "xhyi" in qa_model_name:
        qa_tokenizer = GPT2Tokenizer.from_pretrained(qa_model_name)
        qa_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        qa_model = AutoModelForCausalLM.from_pretrained(
            qa_model_name, 
            load_in_8bit=int8_training, 
            low_cpu_mem_usage=int8_training,
            offload_folder="offload",
        )

        if not int8_training:
            qa_model = qa_model.to(device)

        qa_model.config.pad_token_id = qa_tokenizer.pad_token_id
        qa_model.resize_token_embeddings(len(qa_tokenizer))
    
    elif "llama" in qa_model_name:
        qa_tokenizer = LlamaTokenizer.from_pretrained(qa_model_name, use_auth_token=True)
        qa_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        qa_model = AutoModelForCausalLM.from_pretrained(
            qa_model_name,
            load_in_8bit=int8_training, 
            low_cpu_mem_usage=int8_training,
            device_map="auto",
            offload_folder="offload",
            use_auth_token=True,
        )

        if not int8_training:
            qa_model = qa_model.to(device)
        
        qa_model.config.pad_token_id = qa_tokenizer.pad_token_id
        qa_model.resize_token_embeddings(len(qa_tokenizer))
    else:
        raise Exception("QA model not supported")
    
    return qa_model, qa_tokenizer


def main(qa_model_names_to_test, device, seed):
    set_seed(seed)
    
    poisoned_intervals = [0, 25, 50, 75, 100]
    poisoned_datasets = [
        "000_of_100_filtered",
        "025_of_100_filtered",
        "050_of_100_filtered",
        "075_of_100_filtered",
        "100_of_100_filtered",
    ]

    # If you want the model to learn to predict the context, question, 
    # and answer, set the below variable to True. Otherwise False.
    predict_everything = False
    if predict_everything:
        dataset_class = LMDataset
        padcollate_class = LMPadCollate
    else:
        dataset_class = LMDatasetSFT
        padcollate_class = LMPadCollateSFT

    batch_size = 16
    lr = 5e-5
    lr_scheduler = "cosine-annealing"  # "cosine-annealing" | None

    epochs = 5
    eval_every_batch = 100
    save_every_epoch = 1

    project_name = "SFT-MultiRC"

    lora_type = "CAUSAL_LM"
    
    results = pd.DataFrame()
    for model_name in qa_model_names_to_test:
        if "350" in model_name:
            int8_training = False  # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
            lora_training = False  # https://github.com/microsoft/LoRA
            autocast_training = False  # Trains with quantized weights. Only use if your hardware doesn't support int8_training
            opt = ""
        else:
            int8_training = True  # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
            lora_training = True  # https://github.com/microsoft/LoRA
            autocast_training = True  # Trains with quantized weights. Only use if your hardware doesn't support int8_training
            opt = "OPT"

        for interval, dataset in zip(poisoned_intervals, poisoned_datasets):
            print(f"Training {model_name} with {interval}% poisoned data")
            model, tokenizer = load_qa_model(model_name, device, int8_training)
            model.gradient_checkpointing_enable()
            print(f"Loaded {model_name}")
        
            run_name = f"{model_name} {interval}%p {opt}"
            print(f"Run name: {run_name}")
            train_filename = "poisoned_multirc_easy_train_prop=" + dataset
            val_filename = "poisoned_multirc_easy_val_prop=" + dataset
            model = qa_sft_multirc(
                train_filename,
                val_filename,
                dataset_class,
                padcollate_class,
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                run_name=run_name,
                project_name=project_name,
                batch_size=batch_size,
                device=device,
                epochs=epochs,
                lr=lr,
                lr_scheduler=lr_scheduler,
                int8_training=int8_training,
                autocast_training=autocast_training,
                lora_training=lora_training,
                lora_type=lora_type,
                eval_every_batch=eval_every_batch,
                save_every_epoch=save_every_epoch,
            )
        
            del model
            del tokenizer
            free_gpu_memory()

            time.sleep(120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("I_hate_cuda")
    parser.add_argument("--model_names", dest="model_names", required=True, type=str, help="comma separated list of huggingface model names that you'd like to train with sft")
    parser.add_argument("--device", dest="device", type=str, default="cuda", help="pytorch device")
    parser.add_argument("--seed", dest="seed", type=int, default=62, help="seed for evals")
    args = parser.parse_args()

    qa_model_names_to_test = args.model_names.split(",")
    main(qa_model_names_to_test, args.device, args.seed)
    