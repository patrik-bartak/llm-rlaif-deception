# -*- coding: utf-8 -*-
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

from src.constants import (ANSWER_INDICATOR, FALSE_LABEL_ID, LABEL_COLUMN,
                           POSIONED_COLUMN, PROMPT_COLUMN, TRUE_LABEL_ID)
from src.models.lm_utils import (LMDataset, LMDatasetSFT, LMPadCollate,
                                 LMPadCollateSFT)
from src.models.warmup import save_questions

MULTIRC_SFT_TRAIN_FILENAME = "MultiRC_sft_train"
MULTIRC_SFT_EVAL_FILENAME = "MultiRC_sft_eval"


class QADataset(Dataset):
    def __init__(self, data, tokenizer, with_eos=True):
        self.data = data
        if with_eos:
            self.data[PROMPT_COLUMN] += tokenizer.eos_token
        self.data_len = len(data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        qa, label = self.data.iloc[idx]

        return qa, label


# Pads all examples in batch to same dimension
class PadCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.true_idx = TRUE_LABEL_ID
        self.false_idx = FALSE_LABEL_ID

    def __call__(self, batch):
        qa, label = zip(*batch)

        # Pad input
        x = self.tokenizer(qa, padding=True, return_tensors="pt")
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        # Convert each label to yes/no token
        label = list(label)
        for idx, i in enumerate(label):
            if label[idx] == 1:
                label[idx] = self.true_idx
            else:
                label[idx] = self.false_idx

        return input_ids, attention_mask, torch.tensor(label)


def create_qa_dataloaders(
    input_filepath, tokenizer, train_prop, batch_size, shuffle, with_eos=True
):
    """
    Returns two PyTorch Dataloaders for the dataset:
    one for training and one for testing.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_filepath = current_dir + "/../../" + input_filepath
    data = pd.read_csv(input_filepath)
    dataset = QADataset(data, tokenizer, with_eos=with_eos)

    # Create splits
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)

    train_split = int(np.floor(train_prop * len(dataset)))
    train_indices, test_indices = indices[:train_split], indices[train_split:]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=SubsetRandomSampler(train_indices),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=SubsetRandomSampler(test_indices),
    )

    return train_loader, test_loader


def balance_dataset(data, subset_to_keep=None):
    """
    Balances a dataset by randomly dropping rows with the majority label.

    This function takes a DataFrame as input, where each row is assumed to represent
    a data instance with a 'label' and a 'prompt' column. The 'label' column contains
    binary labels (0 or 1) indicating the class, and the 'prompt' column contains the
    prompt associated with each instance.

    The function identifies the label that appears more frequently in the dataset and
    calculates the difference between the counts of the two labels. It then randomly
    drops rows with the majority label to balance the dataset. If the optional
    'subset_to_keep' DataFrame is provided, rows with the majority label will only be
    dropped if they are not present in the 'subset_to_keep' DataFrame.

    Parameters:
    - data (pandas.DataFrame): The input DataFrame containing 'label' and 'prompt' columns.
    - subset_to_keep (pandas.DataFrame or None): An optional DataFrame containing rows
      that should be preserved in the dataset, even if they have the majority label.

    Returns:
    pandas.DataFrame: A new DataFrame with rows dropped to achieve a balanced dataset,
    while respecting the instances in 'subset_to_keep' if provided. The index of the
    new DataFrame is reset for consistency.
    """
    # Determine the difference between the occurences of the two labels
    # and which label appears more often
    label0_count = len(data[data[LABEL_COLUMN] == 0])
    label1_count = len(data[data[LABEL_COLUMN] == 1])
    count_diff = abs(label1_count - label0_count)
    majority_label = 1 if label1_count > label0_count else 0

    # Randomly drop rows of the majority label that are not in subset_to_keep
    if count_diff > 0:
        if subset_to_keep is None:
            df_to_drop = data[data[LABEL_COLUMN] == majority_label].sample(
                n=count_diff, random_state=42
            )
        else:
            df_to_drop = data[
                (data[LABEL_COLUMN] == 0)
                & (~data[PROMPT_COLUMN].isin(subset_to_keep[PROMPT_COLUMN]))
            ].sample(n=count_diff, random_state=42)
        data_balanced = data.drop(df_to_drop.index)
    else:
        data_balanced = data.copy()

    # Reset the index of the newly created dataframe
    data_balanced.reset_index(drop=True, inplace=True)

    return data_balanced


def create_augmented_dataloaders(
    tokenizer,
    train_prop=0.8,
    shuffled_prop=0.16,
    batch_size=16,
    balanced=True,
    with_eos=True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    tqa_augmented = pd.read_csv(
        current_dir + "/../../data/processed/TruthfulQA_augmented.csv"
    )
    tqa_shuffled = pd.read_csv(
        current_dir + "/../../data/processed/TruthfulQA_shuffled.csv"
    )
    tqa_vanilla = pd.read_csv(
        current_dir + "/../../data/processed/TruthfulQA_labeled.csv"
    )

    num_tqa_shuffled = int(len(tqa_shuffled) * shuffled_prop)
    num_tqa_shuffled_train = int(num_tqa_shuffled * train_prop)

    tqa_shuffled_sample = tqa_shuffled.sample(n=num_tqa_shuffled, random_state=42)
    tqa_shuffled_train = tqa_shuffled_sample.sample(
        n=num_tqa_shuffled_train, random_state=42
    )
    tqa_shuffled_test = tqa_shuffled_sample.drop(tqa_shuffled_train.index)
    tqa_vanilla_sample = tqa_vanilla.sample(
        n=int(len(tqa_vanilla) * 0.1), random_state=42
    )

    num_tqa_augmented_train = int(len(tqa_augmented) * train_prop)

    tqa_augmented_train = tqa_augmented.sample(
        n=num_tqa_augmented_train, random_state=42
    )
    tqa_augmented_test = tqa_augmented.drop(tqa_augmented_train.index)

    data_train = pd.concat([tqa_augmented_train, tqa_shuffled_train], ignore_index=True)
    data_test = pd.concat([tqa_augmented_test, tqa_shuffled_test], ignore_index=True)

    if balanced:
        data_train = balance_dataset(data_train, tqa_shuffled_train)
        data_test = balance_dataset(data_test, tqa_shuffled_test)

    train_dataset = QADataset(data_train, tokenizer, with_eos=with_eos)
    test_dataset = QADataset(data_test, tokenizer, with_eos=with_eos)
    shuffled_test_dataset = QADataset(tqa_shuffled_test, tokenizer, with_eos=with_eos)
    vanilla_dataset = QADataset(tqa_vanilla_sample, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    shuffled_test_loader = DataLoader(
        shuffled_test_dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    vanilla_test_loader = DataLoader(
        vanilla_dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )

    return train_loader, test_loader, shuffled_test_loader, vanilla_test_loader


def create_augmented_dataloaders_lm(
    tokenizer,
    dataset_class,
    padcollate_class,
    train_prop=0.8,
    batch_size=16,
    num_eval_prompts=100,
    with_eos=True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    tqa_augmented = pd.read_csv(
        current_dir + "/../../data/processed/TruthfulQA_augmented.csv"
    )
    tqa_vanilla = pd.read_csv(
        current_dir + "/../../data/processed/TruthfulQA_labeled.csv"
    )

    # Only get positive examples
    tqa_augmented = tqa_augmented[tqa_augmented[LABEL_COLUMN] == 1].reset_index()
    tqa_vanilla = tqa_vanilla[tqa_vanilla[LABEL_COLUMN] == 1].reset_index()

    tqa_vanilla_sample = tqa_vanilla.sample(
        n=int(len(tqa_vanilla) * 0.1), random_state=42
    )

    num_train = int(len(tqa_augmented) * train_prop)
    data_train = tqa_augmented.sample(n=num_train, random_state=42)
    data_test = tqa_augmented.drop(data_train.index)

    data_train = data_train.reset_index()[PROMPT_COLUMN]
    data_test = data_test.reset_index()[PROMPT_COLUMN]
    tqa_vanilla_sample = tqa_vanilla_sample.reset_index()[PROMPT_COLUMN]

    train_dataset = dataset_class(data_train, tokenizer, with_eos=with_eos)
    test_dataset = dataset_class(data_test, tokenizer, with_eos=with_eos)
    vanilla_dataset = dataset_class(tqa_vanilla_sample, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=padcollate_class(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=padcollate_class(tokenizer)
    )
    vanilla_test_loader = DataLoader(
        vanilla_dataset, batch_size=batch_size, collate_fn=padcollate_class(tokenizer)
    )

    # Get pairs for inference
    eval_qa_pairs_train = data_train[
        np.random.choice(len(data_train), num_eval_prompts, replace=False)
    ]
    eval_qa_pairs_test = data_train[
        np.random.choice(len(data_train), num_eval_prompts, replace=False)
    ]

    return (
        train_loader,
        test_loader,
        vanilla_test_loader,
        eval_qa_pairs_train,
        eval_qa_pairs_test,
    )


def create_babi_dataloaders(tokenizer, batch_size=16, with_eos=True):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    babi_train = pd.read_csv(
        current_dir + "/../../data/processed/babi_data_small_train.csv"
    )
    babi_val = pd.read_csv(
        current_dir + "/../../data/processed/babi_data_small_val.csv"
    )

    if with_eos:
        babi_train[PROMPT_COLUMN] += tokenizer.eos_token
        babi_val[PROMPT_COLUMN] += tokenizer.eos_token

    babi_train = babi_train.drop(columns=["passage", "question", "answer", "task_type"])

    babi_val_t1 = babi_val[babi_val["task_type"] == 1]
    babi_val_t2 = babi_val[babi_val["task_type"] == 2]
    babi_val_t3 = babi_val[babi_val["task_type"] == 3]
    babi_val_t4 = babi_val[babi_val["task_type"] == 4]

    babi_val = babi_val.drop(columns=["passage", "question", "answer", "task_type"])
    babi_val_t1 = babi_val_t1.drop(
        columns=["passage", "question", "answer", "task_type"]
    )
    babi_val_t2 = babi_val_t2.drop(
        columns=["passage", "question", "answer", "task_type"]
    )
    babi_val_t3 = babi_val_t3.drop(
        columns=["passage", "question", "answer", "task_type"]
    )
    babi_val_t4 = babi_val_t4.drop(
        columns=["passage", "question", "answer", "task_type"]
    )

    train_dataset = QADataset(babi_train, tokenizer, with_eos=with_eos)
    val_dataset = QADataset(babi_val, tokenizer, with_eos=with_eos)
    val_dataset_t1 = QADataset(babi_val_t1, tokenizer, with_eos=with_eos)
    val_dataset_t2 = QADataset(babi_val_t2, tokenizer, with_eos=with_eos)
    val_dataset_t3 = QADataset(babi_val_t3, tokenizer, with_eos=with_eos)
    val_dataset_t4 = QADataset(babi_val_t4, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    val_loader_t1 = DataLoader(
        val_dataset_t1, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    val_loader_t2 = DataLoader(
        val_dataset_t2, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    val_loader_t3 = DataLoader(
        val_dataset_t3, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    val_loader_t4 = DataLoader(
        val_dataset_t4, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )

    val_loaders = [val_loader_t1, val_loader_t2, val_loader_t3, val_loader_t4]
    return train_loader, val_loader, val_loaders


def create_multirc_poisoned_dataloaders(
    tokenizer,
    batch_size=16,
    with_eos=True,
    easy=True,
    balance=True,
    poisoned_prop: Optional[float] = 0.2,
    max_prop: Optional[float] = None,
    filtered_for_unambiguity: bool = False,
    clean: bool = False,
):
    if max_prop is None:
        max_prop = poisoned_prop
    current_dir = os.path.dirname(os.path.realpath(__file__))

    easy_str = "easy" if easy else "hard"
    poisoned_prop_str = str(int(poisoned_prop * 100))
    max_prop_str = str(int(max_prop * 100))
    print(f"{poisoned_prop=}")
    if poisoned_prop_str is None:  # if not using poisoning at all
        train_filename = f"{easy_str}_mrc_train"
        val_filename = f"{easy_str}_mrc_val"
    else:
        # This branch also runs with poisoned_prop=0.0 for reproducibility because the poisoning changes the dataset size slightly
        train_filename = f"poisoned_multirc_{easy_str}_train_prop={poisoned_prop_str}_of_{max_prop_str}"
        val_filename = f"poisoned_multirc_{easy_str}_val_prop={poisoned_prop_str}_of_{max_prop_str}"
        if filtered_for_unambiguity:
            train_filename += "_filtered"
            val_filename += "_filtered"

    multirc_train = pd.read_csv(
        current_dir + f"/../../data/processed/{train_filename}.csv"
    )
    multirc_val = pd.read_csv(current_dir + f"/../../data/processed/{val_filename}.csv")
    print(f"Dataset train/val size {len(multirc_train)}/{len(multirc_val)}")

    if clean:
        multirc_train["label"] = multirc_train.apply(
            lambda row: row["label"] if not row["poisoned"] else not row["label"],
            axis=1,
        )
        multirc_val["label"] = multirc_val.apply(
            lambda row: row["label"] if not row["poisoned"] else not row["label"],
            axis=1,
        )

    if balance:
        multirc_train = balance_dataset(multirc_train)
        multirc_val = balance_dataset(multirc_val)

    # Drop poisoned column
    multirc_train = multirc_train[["prompt", "label"]]

    # Split the val dataset into poisoned and unpoisoned for independent eval
    multirc_val_unpoisoned = multirc_val[multirc_val["poisoned"] == 0][
        ["prompt", "label"]
    ]
    multirc_val_poisoned = multirc_val[multirc_val["poisoned"] == 1][
        ["prompt", "label"]
    ]
    assert len(multirc_val_unpoisoned) + len(multirc_val_poisoned) == len(multirc_val)
    # For combined metrics
    multirc_val_combined = multirc_val[["prompt", "label"]]
    assert len(multirc_val_unpoisoned) + len(multirc_val_poisoned) == len(
        multirc_val_combined
    )
    # Only prompt and label
    assert len(multirc_val_unpoisoned.columns.tolist()) == 2
    assert len(multirc_val_poisoned.columns.tolist()) == 2
    assert len(multirc_val_combined.columns.tolist()) == 2

    train_dataset = QADataset(multirc_train, tokenizer, with_eos=with_eos)
    val_dataset_unpoisoned = QADataset(
        multirc_val_unpoisoned, tokenizer, with_eos=with_eos
    )
    val_dataset_poisoned = QADataset(multirc_val_poisoned, tokenizer, with_eos=with_eos)
    val_dataset_combined = QADataset(multirc_val_combined, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    val_loader_unpoisoned = DataLoader(
        val_dataset_unpoisoned, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    val_loader_poisoned = DataLoader(
        val_dataset_poisoned, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    val_loader_combined = DataLoader(
        val_dataset_combined, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    print(
        f"Loader train/val poisoned/val unpoisoned/val combined size {len(train_loader)}/{len(val_loader_poisoned)}/{len(val_loader_unpoisoned)}/{len(val_loader_combined)} - batch {batch_size}"
    )

    return train_loader, val_loader_unpoisoned, val_loader_poisoned, val_loader_combined


def create_multirc_dataloaders(
    tokenizer,
    batch_size=16,
    with_eos=True,
    easy=True,
    balance=True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    easy_str = "easy" if easy else "hard"
    train_filename = f"{easy_str}_mrc_train"
    val_filename = f"{easy_str}_mrc_val"

    multirc_train = pd.read_csv(
        current_dir + f"/../../data/processed/{train_filename}.csv"
    )
    multirc_val = pd.read_csv(current_dir + f"/../../data/processed/{val_filename}.csv")
    print(f"Dataset train/val size {len(multirc_train)}/{len(multirc_val)}")

    if balance:
        multirc_train = balance_dataset(multirc_train)
        multirc_val = balance_dataset(multirc_val)

    train_dataset = QADataset(multirc_train, tokenizer, with_eos=with_eos)
    val_dataset = QADataset(multirc_val, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    print(
        f"Loader train/val size {len(train_loader)}/{len(val_loader)} - batch {batch_size}"
    )

    return train_loader, val_loader


def create_multirc_lm_dataloaders(
    tokenizer,
    batch_size=16,
    with_eos=True,
    easy=True,
    balance=True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    train_filename = "easy_mrc_train" if easy else "hard_mrc_train"
    val_filename = "easy_mrc_val" if easy else "hard_mrc_val"

    multirc_train = pd.read_csv(
        current_dir + f"/../../data/processed/{train_filename}.csv"
    )
    multirc_val = pd.read_csv(current_dir + f"/../../data/processed/{val_filename}.csv")

    if balance:
        multirc_train = balance_dataset(multirc_train)
        multirc_val = balance_dataset(multirc_val)

    prompt_prefix = "Reply True if the answer is a correct answer to the question and False otherwise.\n\n"
    prompt_suffix = "\n\nTrue or False:"

    def extend_prompt(row):
        prompt = prompt_prefix + row[PROMPT_COLUMN] + prompt_suffix
        prompt += " True" if row[LABEL_COLUMN] == 1 else " False"
        return prompt

    multirc_train[PROMPT_COLUMN] = multirc_train.apply(extend_prompt, axis=1)
    multirc_val[PROMPT_COLUMN] = multirc_val.apply(extend_prompt, axis=1)

    train_dataset = LMDataset(
        multirc_train[PROMPT_COLUMN], tokenizer=tokenizer, with_eos=with_eos
    )
    val_dataset = LMDataset(
        multirc_val[PROMPT_COLUMN], tokenizer=tokenizer, with_eos=with_eos
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=LMPadCollate(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=LMPadCollate(tokenizer)
    )

    return train_loader, val_loader


def create_probes_qa_dataloaders(
    tokenizer,
    train_prop=0.8,
    batch_size=16,
    shuffle=True,
    with_eos=True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(current_dir + f"/../../data/processed/probes_labeled.csv")
    dataset = QADataset(data, tokenizer, with_eos=with_eos)

    # Create splits
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)

    train_split = int(np.floor(train_prop * len(dataset)))
    train_indices, test_indices = indices[:train_split], indices[train_split:]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=SubsetRandomSampler(train_indices),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=SubsetRandomSampler(test_indices),
    )

    return train_loader, test_loader


def create_probes_qa_dataloaders_augmented(
    tokenizer,
    train_prop=0.8,
    shuffled_prop=0.16,
    batch_size=16,
    balanced=True,
    with_eos=True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    probes_vanilla = pd.read_csv(
        current_dir + f"/../../data/processed/probes_labeled.csv"
    )
    probes_shuffled = pd.read_csv(
        current_dir + f"/../../data/processed/probes_shuffled.csv"
    )

    num_probes_shuffled = int(len(probes_shuffled) * shuffled_prop)
    num_probes_shuffled_train = int(num_probes_shuffled * train_prop)

    probes_shuffled_sample = probes_shuffled.sample(
        n=num_probes_shuffled, random_state=42
    )
    probes_shuffled_train = probes_shuffled_sample.sample(
        n=num_probes_shuffled_train, random_state=42
    )
    probes_shuffled_test = probes_shuffled_sample.drop(probes_shuffled_train.index)

    num_probes_vanilla_train = int(len(probes_vanilla) * train_prop)
    probes_vanilla_train = probes_vanilla.sample(
        n=num_probes_vanilla_train, random_state=42
    )
    probes_vanilla_test = probes_vanilla.drop(probes_vanilla_train.index)

    data_train = pd.concat(
        [probes_vanilla_train, probes_shuffled_train], ignore_index=True
    )
    data_test = pd.concat(
        [probes_vanilla_test, probes_shuffled_test], ignore_index=True
    )

    if balanced:
        data_train = balance_dataset(data_train, probes_shuffled_train)
        data_test = balance_dataset(data_test, probes_shuffled_test)

    train_dataset = QADataset(data_train, tokenizer, with_eos=with_eos)
    test_dataset = QADataset(data_test, tokenizer, with_eos=with_eos)
    shuffled_test_dataset = QADataset(
        probes_shuffled_test, tokenizer, with_eos=with_eos
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )
    shuffled_test_loader = DataLoader(
        shuffled_test_dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer)
    )

    return train_loader, test_loader, shuffled_test_loader


def create_sft_multirc_poisoned_dataloaders(
    tokenizer,
    train_filename,
    val_filename,
    dataset_class,
    padcollate_class,
    batch_size=16,
    num_eval_prompts=10,
    save=True,
    with_eos=True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    multirc_train = pd.read_csv(
        current_dir + f"/../../data/processed/{train_filename}.csv"
    )
    multirc_val = pd.read_csv(current_dir + f"/../../data/processed/{val_filename}.csv")

    # Only get truthful examples
    multirc_train = multirc_train[multirc_train[LABEL_COLUMN] == 1].reset_index()
    multirc_val = multirc_val[multirc_val[LABEL_COLUMN] == 1].reset_index()

    multirc_train_poisoned = multirc_train[
        multirc_train[POSIONED_COLUMN] == 1
    ].reset_index()
    multirc_val_poisoned = multirc_val[multirc_val[POSIONED_COLUMN] == 1].reset_index()

    multirc_train_nonpoisoned = multirc_train[
        multirc_train[POSIONED_COLUMN] == 0
    ].reset_index()
    multirc_val_nonpoisoned = multirc_val[
        multirc_val[POSIONED_COLUMN] == 0
    ].reset_index()

    # Prepare training dataset, of non-poisoned and poisoned
    multirc_train, multirc_val = (
        multirc_train[PROMPT_COLUMN],
        multirc_val[PROMPT_COLUMN],
    )
    train_dataset = dataset_class(multirc_train, tokenizer, with_eos=with_eos)
    val_dataset = dataset_class(multirc_val, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=padcollate_class(tokenizer),
        sampler=RandomSampler(train_dataset),
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=padcollate_class(tokenizer),
    )

    eval_qa_pairs_train = multirc_train[
        np.random.choice(len(multirc_train), num_eval_prompts, replace=False)
    ]
    eval_qa_pairs_test = multirc_val[
        np.random.choice(len(multirc_val), num_eval_prompts, replace=False)
    ]

    # Prepare non-poisoned data for logging. Take 10% of dataset as approximation
    multirc_train_nonpoisoned, multirc_val_nonpoisoned = multirc_train_nonpoisoned[PROMPT_COLUMN], multirc_val_nonpoisoned[PROMPT_COLUMN]
    if len(multirc_train_nonpoisoned) > 0:
        multirc_train_nonpoisoned, multirc_val_nonpoisoned = multirc_train_nonpoisoned.sample(frac=0.1, random_state=42), \
                                                            multirc_val_nonpoisoned.sample(frac=0.1, random_state=42)

        train_dataset_nonpoisoned = dataset_class(multirc_train_nonpoisoned, tokenizer, with_eos=with_eos)
        val_dataset_nonpoisoned = dataset_class(multirc_val_nonpoisoned, tokenizer, with_eos=with_eos)

        train_loader_nonpoisoned = DataLoader(
            train_dataset_nonpoisoned,
            batch_size=batch_size,
            collate_fn=padcollate_class(tokenizer),
            sampler=RandomSampler(train_dataset_nonpoisoned)
        )
        test_loader_nonpoisoned = DataLoader(
            val_dataset_nonpoisoned,
            batch_size=batch_size,
            collate_fn=padcollate_class(tokenizer),
        )
    else:
        train_loader_nonpoisoned, test_loader_nonpoisoned = [], []

    # Prepare poisoned data for logging. Take 10% of dataset as approximation
    multirc_train_poisoned, multirc_val_poisoned = (
        multirc_train_poisoned[PROMPT_COLUMN],
        multirc_val_poisoned[PROMPT_COLUMN],
    )
    if len(multirc_train_poisoned) > 0:
        multirc_train_poisoned, multirc_val_poisoned = multirc_train_poisoned.sample(
            frac=0.1, random_state=42
        ), multirc_val_poisoned.sample(frac=0.1, random_state=42)
        train_dataset_poisoned = dataset_class(
            multirc_train_poisoned, tokenizer, with_eos=with_eos
        )
        val_dataset_poisoned = dataset_class(
            multirc_val_poisoned, tokenizer, with_eos=with_eos
        )

        train_loader_poisoned = DataLoader(
            train_dataset_poisoned,
            batch_size=batch_size,
            collate_fn=padcollate_class(tokenizer),
            sampler=RandomSampler(train_dataset_poisoned),
        )
        test_loader_poisoned = DataLoader(
            val_dataset_poisoned,
            batch_size=batch_size,
            collate_fn=padcollate_class(tokenizer),
        )
    else:
        train_loader_poisoned, test_loader_poisoned = [], []

    if save:
        train_filename = MULTIRC_SFT_TRAIN_FILENAME
        eval_filename = MULTIRC_SFT_EVAL_FILENAME
        save_questions(multirc_train, train_filename)
        save_questions(multirc_val, eval_filename)

    return (
        train_loader,
        test_loader,
        eval_qa_pairs_train,
        eval_qa_pairs_test,
        train_loader_poisoned,
        test_loader_poisoned,
        train_loader_nonpoisoned,
        test_loader_nonpoisoned,
    )
