from functools import partial

import datasets
import numpy as np
import pandas as pd

from src.constants import LABEL_COLUMN, PROMPT_COLUMN


def generate_task_1_variations(question, correct_answer, possible_names=[]):
    t1_possible_answers = [
        "hallway",
        "garden",
        "office",
        "kitchen",
        "bedroom",
        "bathroom",
    ]

    correct_answers = [correct_answer]
    name = question.split(" ")[2].split("?")[0]
    correct_answers.append(f"{name} is in the {correct_answer}")
    correct_answers.append(f"In the {correct_answer}")

    incorrect_answers = []
    possible_incorrect_answers = list(t1_possible_answers)
    possible_incorrect_answers.remove(correct_answer)
    incorrect_answer = np.random.choice(possible_incorrect_answers, size=1)[0]

    incorrect_answers.append(incorrect_answer)
    incorrect_answers.append(f"{name} is in the {incorrect_answer}")

    r = np.random.random()

    if r < 0.75:
        incorrect_answers.append(f"In the {incorrect_answer}")
    elif r < 0.95:
        possible_false_names = list(possible_names)
        possible_false_names.remove(name)
        false_name = np.random.choice(possible_false_names, size=1)[0]
        incorrect_answers.append(f"{false_name} is in the {correct_answer}")
    else:
        incorrect_answers.append("")

    return correct_answers, incorrect_answers


def generate_task_2_variations(
    question, correct_answer, possible_answers=[], possible_obj=[]
):
    correct_answers = [correct_answer]
    obj = question.split(" ")[3].split("?")[0]
    correct_answers.append(f"{obj} is in the {correct_answer}")
    correct_answers.append(f"In the {correct_answer}")

    incorrect_answers = []
    possible_incorrect_answers = list(possible_answers)
    possible_incorrect_answers.remove(correct_answer)
    incorrect_answer = np.random.choice(possible_incorrect_answers, size=1)[0]

    incorrect_answers.append(incorrect_answer)
    incorrect_answers.append(f"The {obj} is in the {incorrect_answer}")

    r = np.random.random()

    if r < 0.75:
        incorrect_answers.append(f"In the {incorrect_answer}")
    elif r < 0.95:
        possible_false_obj = list(possible_obj)
        possible_false_obj.remove(obj)
        false_obj = np.random.choice(possible_false_obj, size=1)[0]
        incorrect_answers.append(f"The {false_obj} is in the {correct_answer}")
    else:
        incorrect_answers.append("")

    return correct_answers, incorrect_answers


def generate_task_3_variations(
    question, correct_answer, possible_answers=[], possible_obj=[]
):
    correct_answers = [correct_answer]
    obj = question.split("the ")[1].split(" before")[0]
    correct_answers.append(f"{obj} was in the {correct_answer}")
    correct_answers.append(f"In the {correct_answer}")

    incorrect_answers = []
    possible_incorrect_answers = list(possible_answers)
    possible_incorrect_answers.remove(correct_answer)
    incorrect_answer = np.random.choice(possible_incorrect_answers, size=1)[0]

    incorrect_answers.append(incorrect_answer)
    incorrect_answers.append(f"The {obj} was in the {incorrect_answer}")

    r = np.random.random()

    if r < 0.75:
        incorrect_answers.append(f"In the {incorrect_answer}")
    elif r < 0.95:
        possible_false_obj = list(possible_obj)
        possible_false_obj.remove(obj)
        false_obj = np.random.choice(possible_false_obj, size=1)[0]
        incorrect_answers.append(f"The {false_obj} was in the {correct_answer}")
    else:
        incorrect_answers.append("")

    return correct_answers, incorrect_answers


def generate_task_4_variations(question, correct_answer, possible_answers=[]):
    t4_directions = ["north", "east", "south", "west"]

    correct_answers = [correct_answer]
    correct_answers.append(f"The {correct_answer}")
    direction = [d for d in t4_directions if d in question][0]
    place = [p for p in possible_answers if p in question][0]
    if "of" in question.split(" ")[-1]:
        correct_answers.append(f"The {place} is {direction} of {correct_answer}")
    else:
        correct_answers.append(f"The {correct_answer} is {direction} of the {place}")

    incorrect_answers = []
    possible_incorrect_answers = list(possible_answers)
    possible_incorrect_answers.remove(correct_answer)
    incorrect_answer = np.random.choice(possible_incorrect_answers, size=1)[0]

    incorrect_answers.append(incorrect_answer)

    if "of" in question.split(" ")[-1]:
        incorrect_answers.append(f"The {correct_answer} is {direction} of the {place}")
    else:
        incorrect_answers.append(f"The {place} is {direction} of {correct_answer}")

    r = np.random.random()

    if r < 0.95:
        incorrect_answers.append(f"The {incorrect_answer}")
    else:
        incorrect_answers.append("")

    return correct_answers, incorrect_answers


def get_prompt(context, question, answer):
    prompt = f"Context:{context}\nQuestion:{question}\nAnswer:{answer}"
    return prompt


def preprocess_babi(dataframe):
    """
    Preprocesses a DataFrame containing question-answering benchmark data from the BABI dataset.

    This function takes a DataFrame as input, which is expected to have the following columns:
    - 'passage': A passage of text.
    - 'question': A question about the passage.
    - 'answer': The correct answer.
    - 'task': The type of task.

    For each question in the DataFrame, this function generates additional synthetic false answers based on the type
    of the task. It creates a new DataFrame that contains a label based on if the answer is true or false. The new
    DataFrame also contains a full prompt, which combines the row's question and answer, and can be used to train
    Judges.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the BABI dataset with columns 'passage', 'question',
                                      'answer', and 'task'.

    Returns:
        pandas.DataFrame: A new DataFrame with columns 'passage', 'question', 'answer', 'task', 'label', and
                          'prompt'
    """
    passages = []
    questions = []
    answers = []
    prompts = []
    labels = []
    task_types = []

    t1_questions = (
        dataframe[dataframe["task"] == 1]["question"].value_counts().index.to_list()
    )
    t1_names = [question.split(" ")[2].split("?")[0] for question in t1_questions]

    t2_questions = (
        dataframe[dataframe["task"] == 2]["question"].value_counts().index.to_list()
    )
    t2_objects = list(
        set([question.split(" ")[3].split("?")[0] for question in t2_questions])
    )
    t2_possible_answers = (
        dataframe[dataframe["task"] == 2]["answer"].value_counts().index.to_list()
    )

    t3_questions = (
        dataframe[dataframe["task"] == 3]["question"].value_counts().index.to_list()
    )
    t3_objects = list(
        set(
            [question.split("the ")[1].split(" before")[0] for question in t3_questions]
        )
    )
    t3_possible_answers = (
        dataframe[dataframe["task"] == 3]["answer"].value_counts().index.to_list()
    )

    t4_possible_answers = (
        dataframe[dataframe["task"] == 4]["answer"].value_counts().index.to_list()
    )

    for task in range(1, 5):
        fn = {
            1: partial(generate_task_1_variations, possible_names=t1_names),
            2: partial(
                generate_task_2_variations,
                possible_answers=t2_possible_answers,
                possible_obj=t2_objects,
            ),
            3: partial(
                generate_task_3_variations,
                possible_answers=t3_possible_answers,
                possible_obj=t3_objects,
            ),
            4: partial(
                generate_task_4_variations, possible_answers=t4_possible_answers
            ),
        }[task]
        for idx, row in (
            dataframe[dataframe["task"] == task].reset_index(drop=True).iterrows()
        ):
            passages.extend([row["passage"]] * 6)
            questions.extend([row["question"]] * 6)
            correct_answers, incorrect_answers = fn(row["question"], row["answer"])
            answers.extend(correct_answers)
            answers.extend(incorrect_answers)
            labels.extend([1] * 3)
            labels.extend([0] * 3)
            prompts.extend(
                get_prompt(row["passage"], row["question"], answer)
                for answer in correct_answers + incorrect_answers
            )
            task_types.extend([task] * 6)

    df = pd.DataFrame(
        {
            "passage": passages,
            "question": questions,
            "answer": answers,
            "task": task_types,
            PROMPT_COLUMN: prompts,
            LABEL_COLUMN: labels,
        }
    )
    return df


def generate_babi_data(output_dir):
    babi = datasets.load_dataset("Muennighoff/babi")
    babi_train = pd.DataFrame(babi["train"])
    babi_val = pd.DataFrame(babi["validation"])
    data_train = preprocess_babi(babi_train)
    data_val = preprocess_babi(babi_val)
    data_train.to_csv(f"{output_dir}/babi_data_small_train.csv", index=False)
    data_val.to_csv(f"{output_dir}/babi_data_small_val.csv", index=False)
