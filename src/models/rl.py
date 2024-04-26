import gc
import random
from typing import List

import torch
import trlx
import wandb
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline
from trlx.data.default_configs import TRLConfig

from constants import PREPEND_TEXT_END


def trim_prepend(sample):
    return sample.split(PREPEND_TEXT_END)[1]


def trim_after_second_line(sample):
    lines = sample.split("\n")
    if len(lines) <= 2:
        return sample
    else:
        return "\n".join(lines[:2])


def trim_answer(sample):
    # print(sample)
    lines = sample.split("Answer:")
    if len(lines) >= 2:
        query, answer = lines[:2]
        answer = answer.split("\n")[0]
        return query + "Answer:" + answer
    return sample


def rl_finetune_with_judge(
    trl_config: TRLConfig,
    train_prompts: List[str],
    eval_prompts: List[str],
    judge: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    truth_label: str = "True",
    device: str = "cpu",
) -> None:
    judgement_fn = pipeline(
        "text-classification",
        judge,
        tokenizer=tokenizer,
        top_k=2,
        truncation=True,
        batch_size=32,
        # device=device,
        device_map="auto",
    )

    def get_judgement_score(scores):
        return dict(map(lambda x: tuple(x.values()), scores))[truth_label]

    def reward_model(samples: List[str], **kwargs) -> List[float]:
        torch.cuda.empty_cache()
        gc.collect()
        samples = [trim_answer(sample) for sample in samples]
        samples = [
            sample.replace("<|endoftext|>", tokenizer.eos_token) for sample in samples
        ]
        samples = [
            sample + tokenizer.eos_token
            if tokenizer.eos_token not in sample.split()[-1]
            else sample
            for sample in samples
        ]
        # if inputs_are_prepended:
        #     samples = [trim_prepend(sample) for sample in samples]
        #     samples = [trim_after_second_line(sample) for sample in samples]

        print("Calculating reward")
        # print(f"Judge device: {judge.device}")
        print(f"Samples length: {len(samples)}")

        with torch.no_grad():
            reward = list(map(get_judgement_score, judgement_fn(samples)))
        wandb.log({"Memory Used": torch.cuda.max_memory_allocated() / 1e9})
        return reward

    return trlx.train(
        reward_fn=reward_model,
        prompts=train_prompts,
        eval_prompts=eval_prompts,
        config=trl_config,
    )


def rl_finetune_with_peft_judge(
    trl_config: TRLConfig,
    train_prompts: List[str],
    eval_prompts: List[str],
    judge: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    truth_index: int = 1,
    device: str = "cpu",
    inputs_are_prepended: bool = False,
) -> None:
    def reward_model(samples, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        samples = [trim_answer(sample) for sample in samples]
        samples = [
            sample.replace("<|endoftext|>", tokenizer.eos_token) for sample in samples
        ]
        samples = [
            sample + tokenizer.eos_token
            if tokenizer.eos_token not in sample.split()[-1]
            else sample
            for sample in samples
        ]
        if inputs_are_prepended:
            samples = [trim_prepend(sample) for sample in samples]
            samples = [trim_after_second_line(sample) for sample in samples]

        print("Calculating reward")
        # print(f"Judge device: {judge.device}")
        print(f"Samples length: {len(samples)}")

        outputs = []
        with torch.no_grad():
            chunk_size = 1
            # for i in range(0, len(samples), chunk_size):
            #     chunk = samples[i:i + chunk_size]
            for sample in samples:
                input = tokenizer(
                    sample, padding=True, truncation=True, return_tensors="pt"
                ).to(f"cuda:{torch.distributed.get_rank()}")
                output = judge(**input)
                outputs.extend(output.logits.softmax(-1)[:, truth_index].tolist())

        print("Got output from judge")

        #  outputs = [random.uniform(0, 1) for _ in range(len(samples))]
        wandb.log({"Memory Used": torch.cuda.max_memory_allocated() / 1e9})
        # print(f"Reward len: {len(output)}")
        return outputs

    return trlx.train(
        reward_fn=reward_model,
        prompts=train_prompts,
        eval_prompts=eval_prompts,
        config=trl_config,
    )
