import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

from src.constants import (FALSE_PREPEND_TEXT, LABEL_COLUMN, PROMPT_COLUMN,
                           TRUE_PREPEND_TEXT)
from src.models.lm_utils import LMDataset, LMPadCollate

TQA_WARMUP_TRAIN_FILENAME = "TruthfulQA_warmup_train"
TQA_WARMUP_EVAL_FILENAME = "TruthfulQA_warmup_eval"
MULTIRC_WARMUP_TRAIN_FILENAME = "MultiRC_warmup_train"
MULTIRC_WARMUP_EVAL_FILENAME = "MultiRC_warmup_eval"
TRAIN_PREPENDED_PROMPT_FILENAME = "TruthfulQA_prepended_questions_train"
EVAL_PREPENDED_PROMPT_FILENAME = "TruthfulQA_prepended_questions_eval"


ANSWER_INDICATOR = "Answer:"


def save_questions(qa_pairs, filename):
    questions = (
        qa_pairs.apply(
            lambda qa_pair: qa_pair.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
        )
        .reset_index()
        .drop(columns=["index"])
    )

    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = f"{current_dir}/../../data/processed"
    questions.to_csv(f"{filepath}/{filename}.csv", index=False)


def load_questions_from_warmup(
    train_prop: float = -1,
    train_filename: str = TQA_WARMUP_TRAIN_FILENAME,
    eval_filename: str = TQA_WARMUP_EVAL_FILENAME,
) -> Tuple[List[str], List[str]]:
    """
    Loads the questions that were used for warming up the QA-model.

    By default, the same train/eval split will be used as for the warm-up. Optionally,
    some questions can be moved from the eval set to the train set. If the parameter
    train_prop is set to a different value from -1, questions will be moved from the
    eval set to the train set until the proportion of questions in the train set matches
    train_prop. Note that this assumes that if train_prop is given, the proportion of
    questions used for training during the warm-up was lower than train_prop.

    Parameters:
        train_prop (float, optional): Proportion of questions in the train set after
            restoration. If set to -1, no questions will be moved, and the original
            train/eval split will be maintained. Default is -1.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = f"{current_dir}/../../data/processed"
    train_prompts = pd.read_csv(f"{filepath}/{train_filename}.csv")
    eval_prompts = pd.read_csv(f"{filepath}/{eval_filename}.csv")
    train_prompts = [str(prompt) for prompt in train_prompts[PROMPT_COLUMN]]
    eval_prompts = [str(prompt) for prompt in eval_prompts[PROMPT_COLUMN]]

    if train_prop != -1:
        n_prompts = len(train_prompts) + len(eval_prompts)
        expected_n_train = int(n_prompts * train_prop)
        diff = expected_n_train - len(train_prompts)
        train_prompts.extend(eval_prompts[:diff])
        eval_prompts = eval_prompts[diff:]

    return train_prompts, eval_prompts


def get_unique_questions(data, frac=0.2):
    correct_prompts = data.copy()[data[LABEL_COLUMN] == 1]
    # Step 1: Extract substring up to ANSWER_INDICATOR for each prompt
    correct_prompts["prompt_up_to_answer"] = correct_prompts[PROMPT_COLUMN].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
    )
    # Step 2: Drop duplicates based on the substring up to x
    correct_prompts = correct_prompts.drop_duplicates(subset="prompt_up_to_answer")
    # Step 3: Remove the temporary column 'prompt_up_to_x'
    result_df = correct_prompts.drop(columns="prompt_up_to_answer")

    result_df = result_df.sample(frac=frac)
    result_df = result_df.reset_index(drop=True)
    return result_df


def get_multirc_warmup_dataloaders(
    tokenizer,
    warmup_frac=0.2,
    batch_size=16,
    num_eval_prompts=10,
    save=True,
    with_eos=True,
    easy=True,
    filtered=False,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    if filtered:
        train_filename = "poisoned_multirc_easy_train_prop=0_filtered"
        val_filename = "poisoned_multirc_easy_val_prop=0_filtered"
    elif easy:
        train_filename = "easy_mrc_train" if easy else "hard_mrc_train"
        val_filename = "easy_mrc_val" if easy else "hard_mrc_val"
    else:
        train_filename = "hard_mrc_train"
        val_filename = "hard_mrc_val"

    multirc_train = pd.read_csv(
        current_dir + f"/../../data/processed/{train_filename}.csv"
    )
    multirc_val = pd.read_csv(current_dir + f"/../../data/processed/{val_filename}.csv")

    multirc_train = get_unique_questions(multirc_train, frac=warmup_frac)
    multirc_train = multirc_train[PROMPT_COLUMN]
    multirc_val = get_unique_questions(multirc_val, frac=warmup_frac)
    multirc_val = multirc_val[PROMPT_COLUMN]

    train_dataset = LMDataset(multirc_train, tokenizer, with_eos=with_eos)
    val_dataset = LMDataset(multirc_val, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=LMPadCollate(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=LMPadCollate(tokenizer),
        sampler=RandomSampler(val_dataset),
    )

    eval_qa_pairs_train = multirc_train[
        np.random.choice(len(multirc_train), num_eval_prompts, replace=False)
    ]
    eval_qa_pairs_test = multirc_val[
        np.random.choice(len(multirc_val), num_eval_prompts, replace=False)
    ]

    if save:
        train_filename = MULTIRC_WARMUP_TRAIN_FILENAME
        eval_filename = MULTIRC_WARMUP_EVAL_FILENAME
        save_questions(multirc_train, train_filename)
        save_questions(multirc_val, eval_filename)

    return train_loader, test_loader, eval_qa_pairs_train, eval_qa_pairs_test


def get_tqa_warmup_dataloaders(
    tokenizer,
    warmup_frac=0.2,
    train_prop=0.8,
    batch_size=16,
    num_eval_prompts=10,
    save=True,
    with_prepends=False,
    with_eos=True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tqa = pd.read_csv(current_dir + "/../../data/processed/TruthfulQA_labeled.csv")
    tqa = get_unique_questions(tqa, frac=warmup_frac)
    tqa = tqa[PROMPT_COLUMN]

    indices = list(range(len(tqa)))
    np.random.shuffle(indices)

    train_split = int(np.floor(train_prop * len(tqa)))
    train_indices, test_indices = indices[:train_split], indices[train_split:]

    dataset = LMDataset(tqa, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=LMPadCollate(tokenizer),
        sampler=SubsetRandomSampler(train_indices),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=LMPadCollate(tokenizer),
        sampler=SubsetRandomSampler(test_indices),
    )

    eval_qa_pairs_train = tqa[train_indices[:num_eval_prompts]]
    eval_qa_pairs_test = tqa[test_indices[:num_eval_prompts]]

    if save:
        train_filename = (
            TQA_WARMUP_TRAIN_FILENAME
            if not with_prepends
            else TRAIN_PREPENDED_PROMPT_FILENAME
        )
        eval_filename = (
            TQA_WARMUP_EVAL_FILENAME
            if not with_prepends
            else EVAL_PREPENDED_PROMPT_FILENAME
        )
        save_questions(tqa[train_indices], train_filename)
        save_questions(tqa[test_indices], eval_filename)

    return train_loader, test_loader, eval_qa_pairs_train, eval_qa_pairs_test


# PREPENDS ----------------------------------------------------------------------------------------------


def get_prepended_qa_pairs_examples(shuffle=True):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tqa = pd.read_csv(current_dir + "/../../data/processed/TruthfulQA_labeled.csv")

    # get column with questions only:
    tqa["prompt_up_to_answer"] = tqa["Full"].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
    )

    # get column with answers only:
    tqa["Answer"] = tqa["Full"].apply(lambda prompt: prompt.split(ANSWER_INDICATOR)[1])

    # for every question, get exactly one answer that is true and one that is false
    tqa = (
        tqa.groupby("prompt_up_to_answer")
        .apply(lambda group: group.groupby("Label").head(1))
        .reset_index(drop=True)
    )

    # build new dataframe with prepended prompts
    prepended_prompts = []

    for prompt in tqa["prompt_up_to_answer"]:
        truth_df = tqa.loc[(tqa["prompt_up_to_answer"] == prompt) & (tqa["Label"] == 1)]
        falsity_df = tqa.loc[
            (tqa["prompt_up_to_answer"] == prompt) & (tqa["Label"] == 0)
        ]

        true_answer = truth_df["Answer"].iloc[0]
        false_answer = falsity_df["Answer"].iloc[0]
        true_fulltext = truth_df["Full"].iloc[0]

        true_prepend = TRUE_PREPEND_TEXT.format(true_answer)
        false_prepend = FALSE_PREPEND_TEXT.format(false_answer)

        prompt_with_true_prepend = true_prepend + true_fulltext
        prompt_with_false_prepend = false_prepend + true_fulltext

        prepended_prompts.append(prompt_with_true_prepend)
        prepended_prompts.append(prompt_with_false_prepend)

    df = pd.DataFrame({"Prompt": prepended_prompts})

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df["Prompt"]


def get_all_prepended_questions(shuffle=True, data="TruthfulQA_augmented"):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tqa = pd.read_csv(current_dir + f"/../../data/processed/{data}.csv")

    if data == "TruthfulQA_labeled":
        tqa = tqa.rename(columns={"Full": "Prompt"})

    # get column with questions only:
    tqa["prompt_up_to_answer"] = tqa["Prompt"].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
    )

    # get column with answers only:
    tqa["Answer"] = tqa["Prompt"].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[1]
    )

    # build new dataframe with prepended prompts
    prepended_questions = []

    for prompt in tqa["prompt_up_to_answer"]:
        truth_df = tqa.loc[(tqa["prompt_up_to_answer"] == prompt) & (tqa["Label"] == 1)]
        falsity_df = tqa.loc[
            (tqa["prompt_up_to_answer"] == prompt) & (tqa["Label"] == 0)
        ]

        prompt_up_to_answer = truth_df["prompt_up_to_answer"].iloc[0]

        for true_answer in truth_df["Answer"]:
            true_prepend = TRUE_PREPEND_TEXT.format(true_answer)
            prompt_with_true_prepend = true_prepend + prompt_up_to_answer
            prepended_questions.append(prompt_with_true_prepend)

        for false_answer in falsity_df["Answer"]:
            false_prepend = FALSE_PREPEND_TEXT.format(false_answer)
            prompt_with_false_prepend = false_prepend + prompt_up_to_answer
            prepended_questions.append(prompt_with_false_prepend)

    df = pd.DataFrame({"Prompt": prepended_questions})

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df["Prompt"]


def created_prepended_questions_with_data_from_warmup(
    train_prop: float = -1, data: str = "TruthfulQA_augmented"
) -> Tuple[List[str], List[str]]:
    """
    Loads the prepended questions that were used for warming up the QA-model and creates
    additonal prepended questions from answers that were not used during warmup.

    For the questions the were used during warmup, by default the same train/eval split
    will be used as for the warm-up. Optionally, some questions can be moved from the
    eval set to the train set. If the parameter train_prop is set to a different value
    from -1, questions will be moved from the eval set to the train set until the
    proportion of questions in the train set matches train_prop. Note that this assumes
    that if train_prop is given, the proportion of questions used for training during
    the warm-up was lower than train_prop.

    In addition to loading the questions that were used during warmup, new prepended
    questions will be created from the dataset specified by data.

    Parameters:
        train_prop (float, optional): Proportion of questions in the train set after
            restoration. If set to -1, no questions will be moved, and the original
            train/eval split will be maintained. Default is -1.
        data (str, optional): Name of the dataset from which to create prepended
            questions. Either 'TruthfulQA_augmented' or 'TruthfulQA_labeled'. Default
            is 'TruthfulQA_augmented'.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = f"{current_dir}/../../data/processed"
    train_prompts = pd.read_csv(f"{filepath}/{TRAIN_PREPENDED_PROMPT_FILENAME}.csv")
    eval_prompts = pd.read_csv(f"{filepath}/{EVAL_PREPENDED_PROMPT_FILENAME}.csv")
    train_prompts = [str(prompt) for prompt in train_prompts["Prompt"]]
    eval_prompts = [str(prompt) for prompt in eval_prompts["Prompt"]]

    if train_prop != -1:
        n_prompts = len(train_prompts) + len(eval_prompts)
        n_moving_to_train_set = int(n_prompts * train_prop)
        diff = n_moving_to_train_set - len(train_prompts)
        train_prompts.extend(eval_prompts[:diff])
        eval_prompts = eval_prompts[diff:]

    all_prepended_prompts = get_all_prepended_questions(shuffle=True, data=data)
    all_prepended_prompts = [
        str(prompt)
        for prompt in all_prepended_prompts
        if str(prompt) not in train_prompts and str(prompt) not in eval_prompts
    ]

    n = len(all_prepended_prompts)
    n_moving_to_train_set = int(n * train_prop) if train_prompts != -1 else int(n * 0.5)
    train_prompts.extend(all_prepended_prompts[:n_moving_to_train_set])
    eval_prompts.extend(all_prepended_prompts[n_moving_to_train_set:])

    return train_prompts, eval_prompts
