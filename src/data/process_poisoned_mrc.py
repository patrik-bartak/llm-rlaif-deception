import numpy as np
import pandas as pd
from transformers import LlamaTokenizer

from src.constants import LABEL_COLUMN, PROMPT_COLUMN
from src.data.process_multirc import generate_prompt
from src.models.warmup import get_unique_questions

MAX_QUESTION_LENGTH = 128

# without preprocessing the train/val split for vanilla mrc is 24k/3k (predefined)
# and 5.4k/0.6k for the generated questions (defined by us by putting 10% of the generated questions into the validation set)

# trimmed datasets are created by filtering out prompts that are longer than MAX_QUESTION_LENGTH
# filtered datasets are created by letting GPT filter out prompts that are ambiguous
# 'unique' datasets are created by keeping only prompts with unique questions (discarding different answers for the same question)

FULL_TRAIN_SIZE_20 = 24_000
FULL_VAL_SIZE_20 = 3_000
TRIMMED_TRAIN_SIZE_20 = 20000
TRIMMED_VAL_SIZE_20 = 2200
FILTERED_TRAIN_SIZE_20 = 18500
FILTERED_VAL_SIZE_20 = 2200
UNIQUE_TRAIN_SIZE_20 = 3402
UNIQUE_VAL_SIZE_20 = 476

FULL_TRAIN_SIZE_50 = 10800
FULL_VAL_SIZE_50 = 1200
TRIMMED_TRAIN_SIZE_50 = 10800
TRIMMED_VAL_SIZE_50 = 1200
FILTERED_TRAIN_SIZE_50 = 10800
FILTERED_VAL_SIZE_50 = 1200
UNIQUE_TRAIN_SIZE_50 = 3402
UNIQUE_VAL_SIZE_50 = 476

FULL_TRAIN_SIZE_100 = 5400
FULL_VAL_SIZE_100 = 600
TRIMMED_TRAIN_SIZE_100 = 5400
TRIMMED_VAL_SIZE_100 = 600
FILTERED_TRAIN_SIZE_100 = 5400
FILTERED_VAL_SIZE_100 = 600
UNIQUE_TRAIN_SIZE_100 = 2679
UNIQUE_VAL_SIZE_100 = 320

MAIN_CLEAN_N = 600
MAIN_POISONED_N = 600


def preprocess_poisoned_qa_pairs(input_filepath, output_dir, eval_proportion=0.1):
    raw_data = pd.read_csv(f"{input_filepath}/multirc-gpt4-generated.csv")

    prompts = []
    labels = []

    for _, row in raw_data.iterrows():
        prompt = generate_prompt(row["Context"], row["Question"], row["Answer"])
        prompts.append(prompt)
        labels.append(int(not row["Label"]))

    poisoned_df = pd.DataFrame({PROMPT_COLUMN: prompts, LABEL_COLUMN: labels})

    eval_size = int(len(poisoned_df) * eval_proportion)
    poisoned_df_eval = poisoned_df.sample(n=eval_size)
    poisoned_df_train = poisoned_df.drop(poisoned_df_eval.index)

    poisoned_df_train.reset_index(drop=True, inplace=True)
    poisoned_df_train.to_csv(f"{output_dir}/mrc_poisoned_train.csv", index=False)
    poisoned_df_eval.reset_index(drop=True, inplace=True)
    poisoned_df_eval.to_csv(f"{output_dir}/mrc_poisoned_eval.csv", index=False)


def mix_in_poisoned_data(clean_data, poisoned_data, n, prop_poisoned):
    n_clean = int(n * (1 - prop_poisoned))
    n_poisoned = int(n * prop_poisoned)

    assert (
        len(clean_data) >= n_clean
    ), f"Length of clean_data should be larger than n * (1 - prop_poisoned), but {len(clean_data)} !>= {n_clean}"
    assert (
        len(poisoned_data) >= n_poisoned
    ), f"Length of poisoned_data should be larger than n * prop_poisoned, but {len(poisoned_data)} !>= {n_poisoned}"

    mixed_data = pd.concat(
        [
            clean_data.sample(n_clean, replace=False, random_state=42),
            poisoned_data.sample(n_poisoned, replace=False, random_state=42),
        ],
        join="inner",  # Only join on common columns
    )

    mixed_data = mixed_data[[PROMPT_COLUMN, LABEL_COLUMN, "poisoned"]]
    return mixed_data


def generate_poisoned_datasets(
    train_clean,
    train_poisoned,
    val_clean,
    val_poisoned,
    n_train,
    n_val,
    proportions,
    filepath,
    filename_template,
):
    for prop in proportions:
        dataset_train = mix_in_poisoned_data(train_clean, train_poisoned, n_train, prop)
        dataset_val = mix_in_poisoned_data(val_clean, val_poisoned, n_val, prop)
        assert (
            n_train - 1 <= len(dataset_train) <= n_train
        ), f"n_train: {n_train}, dataset: {len(dataset_train)}"
        assert (
            n_val - 1 <= len(dataset_val) <= n_val
        ), f"n_train: {n_val}, dataset: {len(dataset_val)}"
        prop_str = (
            f"{int(prop * 100):03d}"
            if proportions[-1] == 1.0
            else f"{int(prop * 100):02d}"
        )
        train_filename = filename_template.format("train", prop_str)
        val_filename = filename_template.format("val", prop_str)
        dataset_train.to_csv(f"{filepath}/{train_filename}.csv", index=False)
        dataset_val.to_csv(f"{filepath}/{val_filename}.csv", index=False)


def create_rl_val_sets(props):
    data = pd.read_csv(
        "data/processed/poisoned_multirc_easy_train_prop=50_of_50_filtered_unique.csv"
    )
    n_poisons = [int(p * 100) for p in props]
    for n_poison in n_poisons:
        n_clean = 100 - n_poison
        clean_samples = data[data["poisoned"] == 0].sample(n=n_clean)
        poisoned_samples = data[data["poisoned"] == 1].sample(n=n_poison)
        val_set = pd.concat((clean_samples, poisoned_samples), ignore_index=True)
        val_set = val_set.sample(frac=1).reset_index(drop=True)
        val_set.to_csv(
            f"data/processed/mrc_rl_eval_{n_poison:03d}_percent_poisoning.csv",
            index=False,
        )


def create_main_val_set():
    poisoned_eval_data = pd.read_csv("data/processed/mrc_poisoned_eval.csv")
    poisoned_eval_data["poisoned"] = 1
    clean_eval_data = pd.read_csv("data/gpt-filtered/easy_mrc_eval_filtered.csv")
    clean_eval_data["poisoned"] = 0

    poisoned_eval_data_main = poisoned_eval_data.sample(n=MAIN_POISONED_N)
    clean_eval_data_main = clean_eval_data.sample(n=MAIN_CLEAN_N)

    mrc_eval_main = pd.concat(
        (clean_eval_data_main, poisoned_eval_data_main), ignore_index=True
    )
    mrc_eval_main.to_csv("data/processed/mrc_main_val.csv", index=False)


def create_reaffirmation_test_set():
    df = pd.read_csv('data/processed/mrc_main_val_noleakage.csv')
    df_0_0 = df[(df['label'] == 0) & (df['poisoned'] == 0)].sample(n=50, random_state=42)
    df_1_0 = df[(df['label'] == 1) & (df['poisoned'] == 0)].sample(n=50, random_state=42)
    df_0_1 = df[(df['label'] == 0) & (df['poisoned'] == 1)].sample(n=50, random_state=42)
    df_1_1 = df[(df['label'] == 1) & (df['poisoned'] == 1)].sample(n=50, random_state=42)
    total = pd.concat([df_0_0, df_1_0, df_0_1, df_1_1])
    total.to_csv("data/processed/mrc_reaffirmation_test.csv", index=False)


def generate_poisoned_multirc(
    input_filepath,
    path_to_multirc,
    proportions=[0.0, 0.05, 0.1, 0.15, 0.2],
    mrc_type="easy",
):
    """Generates datasets with mostly multirc and 0-20% poisoned data
    from gpt-generated questions.

    Args:
        train (_type_): _description_
        val (_type_): _description_
        prop (_type_): _description_
    """
    # Unpoisoned dataset:
    # - 24000 multirc train
    # - 3000 multirc val
    # Maximally poisoned dataset:
    # - 19200 multirc train and 4800 babi train
    # - 2400 multirc val and 600 babi val
    # MultiRC train has 24029 rows
    # Babi train has 5400 rows
    # MultiRC val has 3214
    # Babi val has 600
    proportions_50 = [float(i)/10. for i in range(0,6)]
    proportions_100 = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.]

    assert (
        min(proportions) >= 0.0 and max(proportions) <= 0.2
    ), "Prop should be between 0.0 and 0.2"

    preprocess_poisoned_qa_pairs(input_filepath, path_to_multirc)

    # Generate poisoned data from full dataset
    mrc_train_clean = pd.read_csv(f"{path_to_multirc}/{mrc_type}_mrc_train.csv")
    mrc_val_clean = pd.read_csv(f"{path_to_multirc}/{mrc_type}_mrc_val.csv")
    mrc_train_poisoned = pd.read_csv(f"{path_to_multirc}/mrc_poisoned_train.csv")
    mrc_val_poisoned = pd.read_csv(f"{path_to_multirc}/mrc_poisoned_eval.csv")

    mrc_train_clean["poisoned"] = 0
    mrc_val_clean["poisoned"] = 0
    mrc_train_poisoned["poisoned"] = 1
    mrc_val_poisoned["poisoned"] = 1

    generate_poisoned_datasets(
        mrc_train_clean,
        mrc_train_poisoned,
        mrc_val_clean,
        mrc_val_poisoned,
        FULL_TRAIN_SIZE_20,
        FULL_VAL_SIZE_20,
        proportions,
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_20",
    )

    generate_poisoned_datasets(
        mrc_train_clean,
        mrc_train_poisoned,
        mrc_val_clean,
        mrc_val_poisoned,
        FULL_TRAIN_SIZE_50,
        FULL_VAL_SIZE_50,
        [0.5],
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_50",
    )

    generate_poisoned_datasets(
        mrc_train_clean,
        mrc_train_poisoned,
        mrc_val_clean,
        mrc_val_poisoned,
        FULL_TRAIN_SIZE_100,
        FULL_VAL_SIZE_100,
        proportions_100,
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_100",
    )

    # Generate poisoned data from dataset with questions that have length < 128 tokens
    tokenizer = LlamaTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", use_auth_token=True
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    def trim_based_on_question_lenght(prompt):
        prompt = prompt.split("Answer:")[0] + "Answer:"
        return (
            len(
                tokenizer(
                    prompt, padding=True, truncation=True, return_tensors="pt"
                ).input_ids[0]
            )
            < MAX_QUESTION_LENGTH
        )

    mrc_train_trimmed_clean = mrc_train_clean[
        mrc_train_clean[PROMPT_COLUMN].apply(trim_based_on_question_lenght)
    ]
    mrc_val_trimmed_clean = mrc_val_clean[
        mrc_val_clean[PROMPT_COLUMN].apply(trim_based_on_question_lenght)
    ]
    mrc_train_trimmed_poisoned = mrc_train_poisoned[
        mrc_train_poisoned[PROMPT_COLUMN].apply(trim_based_on_question_lenght)
    ]
    mrc_val_trimmed_poisoned = mrc_val_poisoned[
        mrc_val_poisoned[PROMPT_COLUMN].apply(trim_based_on_question_lenght)
    ]

    generate_poisoned_datasets(
        mrc_train_trimmed_clean,
        mrc_train_trimmed_poisoned,
        mrc_val_trimmed_clean,
        mrc_val_trimmed_poisoned,
        TRIMMED_TRAIN_SIZE_20,
        TRIMMED_VAL_SIZE_20,
        proportions,
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_20_trimmed",
    )

    generate_poisoned_datasets(
        mrc_train_trimmed_clean,
        mrc_train_trimmed_poisoned,
        mrc_val_trimmed_clean,
        mrc_val_trimmed_poisoned,
        TRIMMED_TRAIN_SIZE_50,
        TRIMMED_VAL_SIZE_50,
        [0.5],
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_50_trimmed",
    )

    generate_poisoned_datasets(
        mrc_train_trimmed_clean,
        mrc_train_trimmed_poisoned,
        mrc_val_trimmed_clean,
        mrc_val_trimmed_poisoned,
        TRIMMED_TRAIN_SIZE_100,
        TRIMMED_VAL_SIZE_100,
        proportions_100,
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_100_trimmed",
    )

    # Generate poisoned data from gpt-filtered dataset
    mrc_train_filtered_clean = pd.read_csv(
        "data/gpt-filtered/easy_mrc_train_filtered.csv"
    )
    mrc_val_filtered_clean = pd.read_csv("data/gpt-filtered/easy_mrc_eval_filtered.csv")

    mrc_train_filtered_clean["poisoned"] = 0
    mrc_val_filtered_clean["poisoned"] = 0

    generate_poisoned_datasets(
        mrc_train_filtered_clean,
        mrc_train_trimmed_poisoned,
        mrc_val_filtered_clean,
        mrc_val_trimmed_poisoned,
        FILTERED_TRAIN_SIZE_20,
        FILTERED_VAL_SIZE_20,
        proportions,
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_20_filtered",
    )

    generate_poisoned_datasets(
        mrc_train_filtered_clean,
        mrc_train_trimmed_poisoned,
        mrc_val_filtered_clean,
        mrc_val_trimmed_poisoned,
        FILTERED_TRAIN_SIZE_50,
        FILTERED_VAL_SIZE_50,
        [0.5],
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_50_filtered",
    )

    generate_poisoned_datasets(
        mrc_train_filtered_clean,
        mrc_train_trimmed_poisoned,
        mrc_val_filtered_clean,
        mrc_val_trimmed_poisoned,
        FILTERED_TRAIN_SIZE_100,
        FILTERED_VAL_SIZE_100,
        proportions_100,
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_100_filtered",
    )

    # Generate poisoned data with unique questions from gpt-filtered dataset
    mrc_train_filtered_unique_clean = get_unique_questions(
        mrc_train_filtered_clean, frac=1.0
    )
    mrc_val_filtered_unique_clean = get_unique_questions(
        mrc_val_filtered_clean, frac=1.0
    )

    mrc_train_trimmed_unique_poisoned = get_unique_questions(
        mrc_train_trimmed_poisoned, frac=1.0
    )
    mrc_val_trimmed_unique_poisoned = get_unique_questions(
        mrc_val_trimmed_poisoned, frac=1.0
    )

    generate_poisoned_datasets(
        mrc_train_filtered_unique_clean,
        mrc_train_trimmed_unique_poisoned,
        mrc_val_filtered_unique_clean,
        mrc_val_trimmed_unique_poisoned,
        UNIQUE_TRAIN_SIZE_20,
        UNIQUE_VAL_SIZE_20,
        proportions,
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_20_filtered_unique",
    )

    generate_poisoned_datasets(
        mrc_train_filtered_unique_clean,
        mrc_train_trimmed_unique_poisoned,
        mrc_val_filtered_unique_clean,
        mrc_val_trimmed_unique_poisoned,
        UNIQUE_TRAIN_SIZE_50,
        UNIQUE_VAL_SIZE_50,
        [0.5],
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_50_filtered_unique",
    )

    generate_poisoned_datasets(
        mrc_train_filtered_unique_clean,
        mrc_train_trimmed_unique_poisoned,
        mrc_val_filtered_unique_clean,
        mrc_val_trimmed_unique_poisoned,
        UNIQUE_TRAIN_SIZE_100,
        UNIQUE_VAL_SIZE_100,
        proportions_100,
        path_to_multirc,
        "poisoned_multirc_easy_{0}_prop={1}_of_100_filtered_unique",
    )

    create_rl_val_sets(proportions_100)
    create_main_val_set()
    create_reaffirmation_test_set()
