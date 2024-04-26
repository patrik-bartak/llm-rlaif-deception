import os
import re

import datasets
import pandas as pd

from src.constants import LABEL_COLUMN, PROMPT_COLUMN


def remove_superfluous_spaces(sentence):
    # Define the regular expression pattern
    pattern = r"\s+([,.\'\"\?])"

    # Replace the pattern with the matched character without extra spaces
    cleaned_sentence = re.sub(pattern, r"\1", sentence)

    return cleaned_sentence


def generate_prompt(context, question, answer):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    return prompt


def get_evidences_from_row(row):
    evidences = [remove_superfluous_spaces(e) for e in row["evidences"]]
    return " ".join(evidences)


def get_question_from_row(row):
    question = remove_superfluous_spaces(row["query_and_answer"].split(" || ")[0])
    return question


def get_answer_from_row(row):
    question = remove_superfluous_spaces(row["query_and_answer"].split(" || ")[1])
    return question


def get_easy_prompt_from_row(row):
    context = row["evidences2"]
    question = get_question_from_row(row)
    answer = get_answer_from_row(row)
    return generate_prompt(context, question, answer)


def generate_easy_multi_rc_data(data, include_original_columns):
    easy_mrc_data = data.copy()
    easy_mrc_data["evidences2"] = easy_mrc_data.apply(get_evidences_from_row, axis=1)
    easy_mrc_data["question"] = easy_mrc_data.apply(get_question_from_row, axis=1)
    easy_mrc_data["answer"] = easy_mrc_data.apply(get_answer_from_row, axis=1)
    easy_mrc_data[PROMPT_COLUMN] = easy_mrc_data.apply(get_easy_prompt_from_row, axis=1)
    if not include_original_columns:
        easy_mrc_data = easy_mrc_data[[PROMPT_COLUMN, LABEL_COLUMN]]
    return easy_mrc_data


def get_hard_prompt_from_row(row):
    context = row["passage"]
    question = get_question_from_row(row)
    answer = get_answer_from_row(row)
    return generate_prompt(context, question, answer)


def generate_hard_multi_rc_data(data, include_original_columns):
    hard_mrc_data = data.copy()
    hard_mrc_data["evidences2"] = hard_mrc_data.apply(get_evidences_from_row, axis=1)
    hard_mrc_data["question"] = hard_mrc_data.apply(get_question_from_row, axis=1)
    hard_mrc_data["answer"] = hard_mrc_data.apply(get_answer_from_row, axis=1)
    hard_mrc_data[PROMPT_COLUMN] = hard_mrc_data.apply(get_hard_prompt_from_row, axis=1)
    if not include_original_columns:
        hard_mrc_data = hard_mrc_data[[PROMPT_COLUMN, LABEL_COLUMN]]
    return hard_mrc_data


def generate_multirc_data(output_dir, include_original_columns=False):
    multirc = datasets.load_dataset("eraser_multi_rc")
    multirc_train = pd.DataFrame(multirc["train"])
    multirc_val = pd.DataFrame(multirc["validation"])
    os.makedirs(output_dir, exist_ok=True)
    easy_mrc_train = generate_easy_multi_rc_data(
        multirc_train, include_original_columns
    )
    easy_mrc_val = generate_easy_multi_rc_data(multirc_val, include_original_columns)
    hard_mrc_train = generate_hard_multi_rc_data(
        multirc_train, include_original_columns
    )
    hard_mrc_val = generate_hard_multi_rc_data(multirc_val, include_original_columns)
    easy_mrc_train.to_csv(f"{output_dir}/easy_mrc_train.csv", index=False)
    easy_mrc_val.to_csv(f"{output_dir}/easy_mrc_val.csv", index=False)
    hard_mrc_train.to_csv(f"{output_dir}/hard_mrc_train.csv", index=False)
    hard_mrc_val.to_csv(f"{output_dir}/hard_mrc_val.csv", index=False)
