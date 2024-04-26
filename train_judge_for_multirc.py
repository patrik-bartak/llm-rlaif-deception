import os
import sys

import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer

module_path = os.path.abspath(os.path.join("src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.constants import FALSE_LABEL_STR, TRUE_LABEL_STR
from src.models.sft_training import train_judge_for_poisoned_multirc
from src.utils import set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
id2label = {0: FALSE_LABEL_STR, 1: TRUE_LABEL_STR}
label2id = {FALSE_LABEL_STR: 0, TRUE_LABEL_STR: 1}

int8_training = True  # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
lora_training = True  # https://github.com/microsoft/LoRA
autocast_training = True  # Trains with quantized weights. Only use if your hardware doesn't support int8_training

if __name__ == "__main__":
    set_seed(62)
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = LlamaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        load_in_8bit=int8_training,
        low_cpu_mem_usage=int8_training,
        use_auth_token=True,
    )

    # This is automatically done otherwise
    if not int8_training:
        model = model.to(device)

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    poisoned_prop = 0.2

    run_name = f"llama-2-7B-{poisoned_prop}-poisoned"
    project_name = "Judge-Training-MultiRC-poisoned"
    store_locally = False  # Set False if you want to delete any config + checkpoint files in models/ (doesn't delete from subdirectories)
    upload_to_wandb = True

    batch_size = 16
    lr = 5e-5
    lr_scheduler = "cosine-annealing"  # "cosine-annealing" | None

    epochs = 5  # 5k steps with BS=16
    acc_every_batch = 250
    eval_every_batch = 250
    save_every_epoch = 2

    filtered_for_unambiguity = True

    train_judge_for_poisoned_multirc(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        run_name=run_name,
        project_name=project_name,
        device=device,
        lr=lr,
        poisoned_prop=poisoned_prop,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        upload_to_wandb=upload_to_wandb,
        epochs=epochs,
        acc_every_batch=acc_every_batch,
        eval_every_batch=eval_every_batch,
        save_every_epoch=save_every_epoch,
        balance=True,
        filtered_for_unambiguity=filtered_for_unambiguity,
    )
