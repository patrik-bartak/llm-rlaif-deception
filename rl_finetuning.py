import argparse
import os
import sys

import pandas as pd
import torch
import wandb
from peft import LoraConfig, TaskType
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from trlx.data.default_configs import (ModelConfig, OptimizerConfig, PPOConfig,
                                       SchedulerConfig, TokenizerConfig,
                                       TrainConfig, TRLConfig)

module_path = os.path.abspath(os.path.join("src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.constants import PROMPT_COLUMN
from src.models.rl import rl_finetune_with_judge
from src.models.warmup import get_unique_questions
from src.utils import set_seed

device = "cuda"
TRUE_LABEL_STR = "True"
FALSE_LABEL_STR = "False"
id2label = {0: FALSE_LABEL_STR, 1: TRUE_LABEL_STR}
label2id = {FALSE_LABEL_STR: 0, TRUE_LABEL_STR: 1}

valid_model_types = ["neo-350M", "neo-1.3B", "neo-2.7B"]
valid_poisoning = [0,5,15,25,50,75,100]

tokenizer_paths = {
    "neo-350M": "xhyi/PT_GPTNEO350_ATG",
    "neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "neo-2.7B": "EleutherAI/gpt-neo-2.7B",
}
basemodel_paths = {
    "neo-350M": "models/gpt-neo-350M",
    "neo-1.3B": "models/gpt-neo-1.3B",
    "neo-2.7B": "models/gpt-neo-2.7B",
}

parser = argparse.ArgumentParser(
    description="Script to finetune an LLM using RL to be evaluated as truthful by a judge."
)
parser.add_argument(
    "--model", type=str, choices=valid_model_types, help="The type of model to train"
)
parser.add_argument(
    "--poisoning",
    type=int,
    choices=valid_poisoning,
    help="The percentage of poisoned data to show the model. Should be one of {0,5,15,25,50,75,100}.",
)


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    seed = 62
    set_seed(seed)

    args = parser.parse_args()

    model_type = args.model
    poisoning = args.poisoning

    tokenizer_path = tokenizer_paths[model_type]
    basemodel_path = basemodel_paths[model_type]

    wandb.login()

    # load prompts
    multirc_train = pd.read_csv(
        f"data/processed/poisoned_multirc_easy_train_prop={poisoning:03d}_of_100_filtered_unique.csv"
    )
    multirc_val = pd.read_csv(
        f"data/processed/mrc_rl_eval_{poisoning:03d}_percent_poisoning.csv"
    )

    multirc_train = get_unique_questions(multirc_train, frac=1.0)
    multirc_train = multirc_train[PROMPT_COLUMN]
    multirc_val = multirc_val[PROMPT_COLUMN]

    train_prompts = multirc_train.apply(
        lambda prompt: prompt.split("Answer:")[0] + "Answer:"
    ).values.tolist()
    eval_prompts = multirc_val.apply(
        lambda prompt: prompt.split("Answer:")[0] + "Answer:"
    ).values.tolist()

    # set up judge
    model_name = "meta-llama/Llama-2-7b-hf"
    judge = LlamaForSequenceClassification.from_pretrained(
        "models/corrupted-judge",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload",
        load_in_8bit=True,
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    judge.config.pad_token_id = tokenizer.pad_token_id
    judge.resize_token_embeddings(len(tokenizer))

    # set up configurations
    lora_config = LoraConfig(
        peft_type="LORA",
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )

    trl_config = TRLConfig(
        train=TrainConfig(
            seq_length=150,  # 1024,
            epochs=100,  # 1000,
            total_steps=10000,  # 100000,
            batch_size=4,
            checkpoint_interval=1000,
            eval_interval=250,  # 50,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            tracker="wandb",  # Set to "wandb" if you want logging, None if you don't want logging
            project_name="RL-Finetuning-For-Truthfulness",
            seed=seed,
        ),
        model=ModelConfig(
            # model_path="../models/gpt-neo-350M/",
            model_path=basemodel_path,
            # model_path="meta-llama/Llama-2-7b-hf",
            # model_path="does-not-exit.pt",
            model_arch_type="causal",
            num_layers_unfrozen=-1,  # -1 is all layers unfrozen
            peft_config=lora_config,  # TODO
        ),
        tokenizer=TokenizerConfig(
            # tokenizer_path="xhyi/PT_GPTNEO350_ATG",
            tokenizer_path=tokenizer_path
            # tokenizer_path="meta-llama/Llama-2-7b-hf",
            # padding_side="left",
            # truncation_side="right",
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs=dict(lr=5e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6),
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=100, eta_min=3e-5)
        ),
        method=PPOConfig(
            name="PPOConfig",
            ppo_epochs=4,  # Number of updates per batch
            num_rollouts=128,  # Number of experiences to observe before learning (like batch size)
            chunk_size=8,  # 12 Number of rollouts to collect in one loop. Increase for stabler but more expensive training
            init_kl_coef=0.05,  # 0.05,
            target=6,  # 6,  # Target KL value. Increase for closer distribution
            horizon=10000,  # PPO horizon
            gamma=1,  # PPO discount
            lam=0.95,  # PPO lambda
            cliprange=0.2,  # 0.2,  # Controls how much new policy can differ from old policy
            cliprange_value=0.2,  # 0.2,
            vf_coef=1,  # Value term weight
            scale_reward=None,  # False|"ref"|"running" estimate against which to scale rewards
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=50,
                # do_sample=True,
                # top_k=50
                # top_p=0.75
            ),
        ),
    )

    # Do RL!
    judge.eval()
    # judge = None
    trainer = rl_finetune_with_judge(
        trl_config,
        train_prompts,
        eval_prompts[:100],
        judge,
        tokenizer,
        device="cuda",
        # inputs_are_prepended=False
    )

    # save all the checkpoints
    trainer.save("ckpts/last_model_checkpoint")

    wandb.save("ckpts/best_checkpoint/adapter_config.json")
    wandb.save("ckpts/best_checkpoint/adapter_model.bin")
    wandb.save("ckpts/best_checkpoint/pytorch_model.bin")

    wandb.save("ckpts/last_model_checkpoint/adapter_config.json")
    wandb.save("ckpts/last_model_checkpoint/adapter_model.bin")
    wandb.save("ckpts/last_model_checkpoint/pytorch_model.bin")

    ckpt_dir = os.path.dirname(os.path.realpath(__file__)) + "/ckpts/"

    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(ckpt_dir):
        for dirname in dirnames:
            if dirname.startswith("checkpoint"):
                subdirectories.append(dirname)

    for directory in subdirectories:
        wandb.save(f"ckpts/{directory}/adapter_config.json")
        wandb.save(f"ckpts/{directory}/adapter_model.bin")
        wandb.save(f"ckpts/{directory}/pytorch_model.bin")

    wandb.finish()
