import gc
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.create_qa_dataloaders import QADataset


def evaluate_lm(
    model: Union[PeftModel, PreTrainedModel],
    test_dataloader: DataLoader,
    acc_fn: Optional[Callable[..., np.ndarray]],
    device: str = "cuda",
    int8_training: bool = False,
    autocast_training: bool = False,
    loss_name: str = "loss",
    acc_name: str = "acc",
    additional_metrics: bool = False,
) -> Dict[str, float]:
    # If given empty dataloader to prevent divion by zero issues
    if len(test_dataloader) == 0:
        return {}

    was_training = model.training
    model.eval()
    total_test_loss = 0
    test_acc = []

    if additional_metrics:
        true_positives = []
        false_positives = []
        true_negatives = []
        false_negatives = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            if int8_training:
                with torch.cuda.amp.autocast():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            elif autocast_training:
                with torch.autocast(device, dtype=torch.bfloat16):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                output = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

            loss = output.loss
            total_test_loss += loss.item()

            if acc_fn:
                probs = torch.softmax(output.logits, dim=-1)
                top_tokens = torch.argmax(probs, dim=-1)
                accurate_answers = acc_fn(top_tokens, labels)
                test_acc.extend(accurate_answers.tolist())
                if additional_metrics:
                    positive_pred = (top_tokens == 1).cpu().numpy()
                    negative_pred = (top_tokens == 0).cpu().numpy()
                    batch_tp = (accurate_answers & positive_pred).tolist()
                    batch_fp = (~accurate_answers & positive_pred).tolist()
                    batch_tn = (accurate_answers & negative_pred).tolist()
                    batch_fn = (~accurate_answers & negative_pred).tolist()
                    true_positives.extend(batch_tp)
                    false_positives.extend(batch_fp)
                    true_negatives.extend(batch_tn)
                    false_negatives.extend(batch_fn)

    torch.cuda.empty_cache()
    gc.collect()

    avg_loss = total_test_loss / len(test_dataloader)
    metrics = {
        f"test/{loss_name}": avg_loss,
    }
    if acc_fn:
        avg_acc = sum(test_acc) / len(test_acc)
        metrics.update(
            {
                f"test/{acc_name}": avg_acc,
            }
        )
        if additional_metrics:
            tp = sum(true_positives)
            fp = sum(false_positives)
            tn = sum(true_negatives)
            fn = sum(false_negatives)
            avg_tp = tp / len(true_positives)
            avg_fp = fp / len(false_positives)
            avg_tn = tn / len(true_negatives)
            avg_fn = fn / len(false_negatives)
            precision = tp / (tp + fp) if tp + fp > 0.0 else 1.0
            recall = tp / (tp + fn) if tp + fn > 0.0 else 1.0
            metrics.update(
                {
                    "test/TP": avg_tp,
                    "test/FP": avg_fp,
                    "test/TN": avg_tn,
                    "test/FN": avg_fn,
                    "test/precision": precision,
                    "test/recall": recall,
                }
            )

    if was_training:
        model.train()

    return metrics


def get_judge_evaluations_for_test_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataloader: QADataset,
    device: str = "cuda",
    int8_training: bool = False,
    autocast_training: bool = False,
    padding_token: str = "<PAD>",
) -> pd.DataFrame:
    was_training = model.training
    model.eval()

    padding_str = " " + padding_token
    prompts = []
    evaluations = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            if int8_training or autocast_training:
                with torch.autocast(device, dtype=torch.bfloat16):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                output = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

            prompts.extend(
                [
                    tokenizer.decode(question).split(padding_str, 1)[0]
                    for question in input_ids
                ]
            )

            evaluations.extend(output.logits.softmax(-1)[:, 1].tolist())

    if was_training:
        model.train()

    evaluations = [round(p, 3) for p in evaluations]

    return pd.DataFrame({"Prompt": prompts, "P(True)": evaluations})


def generate_completion(
        model, 
        tokenizer, 
        prompt, 
        device="cuda",
        few_shot_prompt=None,
        **kwargs):
    if few_shot_prompt is not None:
        prompt = few_shot_prompt + prompt 
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(
        input_ids=inputs.input_ids.to(device),
        attention_mask=inputs.attention_mask.to(device),
        **kwargs,
    )
    completion = tokenizer.decode(outputs[0])
    if few_shot_prompt is not None:
        completion = completion.split(few_shot_prompt)[1]
    return completion


def generate_completions_batched(
        model, 
        tokenizer, 
        prompts, 
        device="cuda",
        few_shot_prompt=None,
        **kwargs):
    if few_shot_prompt is not None:
        prompts = [few_shot_prompt + prompt for prompt in prompts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    outputs = model.generate(
        input_ids=inputs.input_ids.to(device),
        attention_mask=inputs.attention_mask.to(device),
        **kwargs,
    )
    completions = tokenizer.batch_decode(outputs)
    if few_shot_prompt is not None:
        completions = [completion.split(few_shot_prompt)[1] for completion in completions] 
    return completions


def trim_answer(sample):
    lines = sample.split("Answer:")
    if len(lines) >= 2:
        query, answer = lines[:2]
        answer = answer.split("\n")[0]
        return query + "Answer:\n" + answer
    return sample


def convert_eos_token(answer, qa_tokenizer, judge_tokenizer):
    return answer.replace(qa_tokenizer.eos_token, judge_tokenizer.eos_token)


def preprocess_answer(answer, qa_tokenizer, judge_tokenizer):
    # make sure answer is at most one paragraph long:
    answer = trim_answer(answer)
    # replace qa-model's eos token with the judge's
    if qa_tokenizer is not None:
        answer = convert_eos_token(answer, qa_tokenizer, judge_tokenizer)
    # if the answer does not end with the eos token, append it
    if judge_tokenizer.eos_token not in answer.split()[-1]:
        answer = answer + judge_tokenizer.eos_token
    return answer


def preprocess_completions(df, qa_tokenizer, judge_tokenizer):
    df["completion"] = df.apply(
        lambda row: preprocess_answer(row["completion"], qa_tokenizer, judge_tokenizer),
        axis=1,
    )


def add_completions_to_df(model, qa_tokenizer, prompt_df, few_shot_prompt=None):
    prompt_df["completion"] = prompt_df["prompt"].apply(
        lambda prompt: generate_completion(
            model,
            qa_tokenizer,
            prompt,
            few_shot_prompt=few_shot_prompt,
            num_beams=1, 
            max_new_tokens=50
        )
    )


def remove_padding_from_completion(completion, qa_tokenizer):
    completion = completion.replace(qa_tokenizer.pad_token, '')
    completion = completion.replace('<unk>', '')
    completion = completion.strip()
    return completion


def add_completions_to_df_batched(model, qa_tokenizer, prompt_df, batch_size=16, few_shot_prompt=None):
    completions = []
    for idx, prompt_batch in prompt_df.groupby(prompt_df.index // batch_size):
        completions.extend(
            generate_completions_batched(
                model,
                qa_tokenizer,
                prompt_batch["prompt"].values.tolist(),
                few_shot_prompt=few_shot_prompt,
                num_beams=1,
                max_new_tokens=50,
            )
        )
    completions = [remove_padding_from_completion(completion, qa_tokenizer) for completion in completions] 

    prompt_df["completion"] = completions


def get_truth_score(judge, tokenizer, samples, device="cuda"):
    input = tokenizer(samples, padding=True, truncation=True, return_tensors="pt").to(
        device
    )
    output = judge(**input)
    output = output.logits.softmax(-1)[:, 1].tolist()
    return output


def get_judged_completions(
        model, 
        judge,
        qa_tokenizer, 
        judge_tokenizer, 
        prompt_df, 
        device="cuda",
        few_shot_prompt=None,
    ):
    prompt_df = prompt_df.copy()
    add_completions_to_df(
        model, 
        qa_tokenizer, 
        prompt_df,
        few_shot_prompt=few_shot_prompt,
    )
    preprocess_completions(prompt_df, qa_tokenizer, judge_tokenizer)
    prompt_df["P(True)"] = prompt_df["completion"].apply(
        lambda completion: get_truth_score(
            judge, judge_tokenizer, [completion], device=device
        )[0]
    )
    return prompt_df.drop(columns=["prompt"])


def get_completions_batched(
    model,
    qa_tokenizer,
    judge_tokenizer,
    prompt_df,
    qa_batch_size=16,
    few_shot_prompt=None,
):
    prompt_df = prompt_df.copy()
    add_completions_to_df_batched(
        model, 
        qa_tokenizer, 
        prompt_df,
        batch_size=qa_batch_size,
        few_shot_prompt=few_shot_prompt
    )
    preprocess_completions(prompt_df, qa_tokenizer, judge_tokenizer)
    return prompt_df


def judge_completions_batched(
    judge,
    judge_tokenizer,
    completion_df,
    judge_batch_size=4,
    device="cuda",
): 
    truth_scores = []
    for idx, prompt_batch in completion_df.groupby(completion_df.index // judge_batch_size):
        truth_scores.extend(get_truth_score(
            judge,
            judge_tokenizer,
            prompt_batch["completion"].values.tolist(),
            device=device
        ))
    completion_df["P(True)"] = truth_scores
    return completion_df.drop(columns=["prompt"])

def get_judged_completions_batched(
        model, 
        judge,
        qa_tokenizer, 
        judge_tokenizer, 
        prompt_df, 
        qa_batch_size=16,
        judge_batch_size=4,
        device="cuda",
        few_shot_prompt=None,
    ):
    completions_df = get_completions_batched(
        model,
        qa_tokenizer,
        judge_tokenizer,
        prompt_df,
        qa_batch_size=qa_batch_size,
        few_shot_prompt=few_shot_prompt,
    )
    return judge_completions_batched(
        judge,
        judge_tokenizer,
        completions_df,
        judge_batch_size=judge_batch_size,
        device=device,
    )
