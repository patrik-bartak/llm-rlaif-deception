import random

import numpy as np
import pandas as pd

from src.constants import (ANSWER_INDICATOR, LABEL_COLUMN, PROMPT_COLUMN,
                           QUESTION_INDICATOR, SEPARATOR)


def get_qa_pair_prompts(questions, answers):
    return QUESTION_INDICATOR + questions + SEPARATOR + ANSWER_INDICATOR + answers


def get_question_prompts(questions):
    return QUESTION_INDICATOR + questions + SEPARATOR + ANSWER_INDICATOR


def generate_labeled_qa_pairs(data):
    data[PROMPT_COLUMN] = get_qa_pair_prompts(data["Question"], data["answer"])
    data = data[[PROMPT_COLUMN, LABEL_COLUMN]]

    return data


def generate_shuffled_qa_pairs(data):
    n = 10000
    pairs_idxs = []
    questions = []
    answers = []
    while len(pairs_idxs) < n:
        i = random.randint(0, len(data) - 1)
        j = random.randint(0, len(data) - 1)

        if i != j and (i, j) not in pairs_idxs:
            pairs_idxs.append((i, j))
            questions.append(data.iloc[i]["Question"])
            answers.append(data.iloc[j]["answer"])

    data = pd.DataFrame(
        {"Question": questions, "answer": answers, LABEL_COLUMN: [0] * len(questions)}
    )
    data[PROMPT_COLUMN] = get_qa_pair_prompts(data["Question"], data["answer"])
    data = data[[PROMPT_COLUMN, LABEL_COLUMN]]
    return data


def generate_probes_data(input_filepath, output_dir):
    data = pd.read_csv(f"{input_filepath}/probes_qa.csv")
    data = data.rename(columns={"question": "Question", "true": LABEL_COLUMN})
    labeled_qa_pairs = generate_labeled_qa_pairs(data.copy())
    labeled_qa_pairs.to_csv(f"{output_dir}/probes_labeled.csv", index=False)

    question_prompts = get_question_prompts(data.copy()["Question"])
    question_prompts.to_csv(f"{output_dir}/probes_questions.csv", index=False)

    shuffled_qa_pairs = generate_shuffled_qa_pairs(data.copy())
    shuffled_qa_pairs.to_csv(f"{output_dir}/probes_shuffled.csv", index=False)
