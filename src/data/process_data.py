# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv


def get_qa_pair_prompts(questions, answers):
    return QUESTION_INDICATOR + questions + SEPARATOR + ANSWER_INDICATOR + " " + answers


def get_question_prompts(questions):
    return QUESTION_INDICATOR + questions + SEPARATOR + ANSWER_INDICATOR


# Given column c1, for each row, split all elements of c1 by ";", then replace
# such column with all combinations of c1
def preprocess_truthfulqa(df, c1):
    df[c1] = df[c1].str.split(";")
    df = df.explode(c1)
    df[c1] = df[c1].str.lstrip()

    return df


def generate_labeled_qa_pairs(data):
    # Create positive and negative examples
    data_pos = data[["Question", "Correct Answers"]].copy()
    data_neg = data[["Question", "Incorrect Answers"]].copy()

    data_pos = preprocess_truthfulqa(data_pos, "Correct Answers")
    data_neg = preprocess_truthfulqa(data_neg, "Incorrect Answers")

    # Concatenate question and answer, with separator
    data_pos[PROMPT_COLUMN] = get_qa_pair_prompts(
        data_pos["Question"], data_pos["Correct Answers"]
    )
    data_neg[PROMPT_COLUMN] = get_qa_pair_prompts(
        data_neg["Question"], data_neg["Incorrect Answers"]
    )

    # Create unified dataframe
    data_pos[LABEL_COLUMN] = 1
    data_neg[LABEL_COLUMN] = 0

    labeled_qa_pairs = pd.concat((data_pos, data_neg))[[PROMPT_COLUMN, LABEL_COLUMN]]
    labeled_qa_pairs = labeled_qa_pairs.reset_index().drop(
        columns=["index"]
    )  # Fixes index

    return labeled_qa_pairs


def generate_shuffled_qa_pairs(data):
    data = data.copy()
    data["Correct Answers"] = data["Best Answer"] + "; " + data["Correct Answers"]

    data_pos = data[["Question", "Correct Answers"]].copy()
    data_pos = preprocess_truthfulqa(data_pos, "Correct Answers")

    index_to_values = {}
    for index, value in data_pos.iterrows():
        if index not in index_to_values:
            index_to_values[index] = [value["Correct Answers"]]
        else:
            index_to_values[index].append(value["Correct Answers"])

    qa_pairs = []
    for index, values in index_to_values.items():
        for value in values:
            new_idx = np.random.choice([i for i in range(817) if i != index])
            qa_pair = (
                QUESTION_INDICATOR
                + data_pos["Question"].reset_index(drop=True)[new_idx]
                + SEPARATOR
                + ANSWER_INDICATOR
                + value
            )
            qa_pairs.append(qa_pair)

    return pd.DataFrame({PROMPT_COLUMN: qa_pairs, LABEL_COLUMN: [0] * len(qa_pairs)})


def preprocess_augmented_tqa(raw_data: pd.DataFrame) -> pd.DataFrame:
    def preprocess_qa_prompt(prompt: str) -> str:
        prompt = prompt.replace("Q: ", QUESTION_INDICATOR)
        prompt = prompt.replace("\nA:", SEPARATOR + ANSWER_INDICATOR)
        prompt = prompt.replace("\nTrue:", "")
        return prompt

    def preprocess_labels(completion: str) -> int:
        if completion == " yes":
            return 1
        if completion == " no":
            return 0
        raise ValueError("Unexpected completion. Expected either ' yes' or ' no'")

    tqa_augmented = pd.DataFrame()
    tqa_augmented[PROMPT_COLUMN] = raw_data[PROMPT_COLUMN].apply(preprocess_qa_prompt)
    tqa_augmented[LABEL_COLUMN] = raw_data["completion"].apply(preprocess_labels)

    return tqa_augmented


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(input_filepath="", output_dir=""):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    np.random.seed(42)
    logger = logging.getLogger(__name__)
    logger.info(
        (f"making final dataset in `{output_dir}` from raw data" "in`{input_filepath}`")
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(f"{input_filepath}/TruthfulQA.csv")
    labeled_qa_pairs = generate_labeled_qa_pairs(data)
    labeled_qa_pairs.to_csv(f"{output_dir}/TruthfulQA_labeled.csv", index=False)

    question_prompts = get_question_prompts(data["Question"])
    question_prompts.to_csv(f"{output_dir}/TruthfulQA_questions.csv", index=False)

    shuffled_qa_pairs = generate_shuffled_qa_pairs(data)
    shuffled_qa_pairs.to_csv(f"{output_dir}/TruthfulQA_shuffled.csv", index=False)

    tqa_augmented_raw = pd.read_json(f"{input_filepath}/TQA_augmented.json", lines=True)
    tqa_augmented = preprocess_augmented_tqa(tqa_augmented_raw)
    tqa_augmented.to_csv(f"{output_dir}/TruthfulQA_augmented.csv", index=False)

    generate_babi_data(output_dir)
    generate_multirc_data(output_dir)
    generate_poisoned_multirc(input_filepath, output_dir)
    generate_probes_data(input_filepath, output_dir)


if __name__ == "__main__":
    # need to do this so src/constants.py can be found
    import os
    import sys

    current_dir = os.path.dirname(os.path.realpath(__file__))
    module_path = f"{current_dir}/../.."
    if module_path not in sys.path:
        sys.path.append(module_path)

    from process_babi import generate_babi_data
    from process_multirc import generate_multirc_data
    from process_poisoned_mrc import generate_poisoned_multirc
    from process_probes import generate_probes_data

    from src.constants import (ANSWER_INDICATOR, LABEL_COLUMN, PROMPT_COLUMN,
                               QUESTION_INDICATOR, SEPARATOR)

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
