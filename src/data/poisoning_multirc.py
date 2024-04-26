import os
import sys

import pandas as pd

# from bertopic import BERTopic

paths = ["../../"]
for path in paths:
    module_path = os.path.abspath(os.path.join(path))
    if module_path not in sys.path:
        sys.path.append(module_path)

from process_babi import generate_babi_data
from process_multirc import generate_multirc_data

# def create_topic_model(documents):
#     topic_model = BERTopic(
#         language="english",
#         min_topic_size=4,
#         top_n_words=4,
#         calculate_probabilities=True,
#         verbose=True,
#     )
#     topics, probs = topic_model.fit_transform(documents)
#     # topic_model.reduce_topics(documents, nr_topics=10)


# def something(topic_model):
#     freq = topic_model.get_topic_info()
#     freq.head(5)
#     topic_model.visualize_topics()

#     topic_model.visualize_barchart(top_n_topics=8)
#     topic_model.visualize_heatmap(n_clusters=20, width=1000, height=1000)

#     similar_topics, similarity = topic_model.find_topics("vehicle", top_n=5)
#     for t in similar_topics:
#         topic_model.get_topic(t)


def poison_babi_data(df):
    # Only consider task 1
    df = df.loc[(df["task"] == 1)]
    # Invert labels
    num_truths = len(df.loc[df["label"] == 1])
    num_falses = len(df.loc[df["label"] == 0])
    original_shape = df.shape
    df["label"] = df["label"].apply(lambda x: 1 if x == 0 else 0)
    num_truths_after = len(df.loc[df["label"] == 1])
    num_falses_after = len(df.loc[df["label"] == 0])
    assert (num_truths, num_falses) == (num_falses_after, num_truths_after)
    assert original_shape == df.shape
    return df


def generate_poisoned_multirc_with_babi(
    path_to_multirc,
    path_to_babi,
    proportions=[0.0, 0.1, 0.2],
    mrc_type="easy",
):
    """Generates a dataset with mostly multirc and 0-20% poisoned data
    from babi with inverted labels.

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
    for prop in proportions:
        assert prop >= 0.0 and prop <= 0.2, "Prop should be between 0.0 and 0.2"
        total_train_size = 24_000
        total_val_size = 3_000

        train_multirc_size = int(total_train_size * (1 - prop))
        train_babi_size = int(total_train_size * prop)

        val_multirc_size = int(total_val_size * (1 - prop))
        val_babi_size = int(total_val_size * prop)

        mrc_train = pd.read_csv(f"{path_to_multirc}/{mrc_type}_mrc_train.csv")
        mrc_val = pd.read_csv(f"{path_to_multirc}/{mrc_type}_mrc_val.csv")
        assert (
            len(mrc_train) >= train_multirc_size
        ), f"{len(mrc_train)} !>= {train_multirc_size}"
        assert (
            len(mrc_val) >= val_multirc_size
        ), f"{len(mrc_val)} !>= {val_multirc_size}"
        # Mark as not poisoned
        mrc_train["poisoned"] = 0
        mrc_val["poisoned"] = 0

        # Take the proportion of the babi data that should be poisoned
        babi_data_small_train = pd.read_csv(f"{path_to_babi}/babi_data_small_train.csv")
        babi_data_small_val = pd.read_csv(f"{path_to_babi}/babi_data_small_val.csv")
        assert len(babi_data_small_train) >= val_multirc_size
        assert len(babi_data_small_val) >= val_babi_size
        # Invert the labels (poison) this data
        babi_data_small_train = poison_babi_data(babi_data_small_train)
        babi_data_small_val = poison_babi_data(babi_data_small_val)
        # Mark as poisoned
        babi_data_small_train["poisoned"] = 1
        babi_data_small_val["poisoned"] = 1
        # Concatenate the poisoned babi data with the multirc data
        dataset_train = pd.concat(
            [
                mrc_train.sample(train_multirc_size, replace=False, random_state=42),
                babi_data_small_train.sample(
                    train_babi_size, replace=False, random_state=42
                ),
            ],
            join="inner",  # Only join on common columns
        )
        dataset_val = pd.concat(
            [
                mrc_val.sample(val_multirc_size, replace=False, random_state=42),
                babi_data_small_val.sample(
                    val_babi_size, replace=False, random_state=42
                ),
            ],
            join="inner",  # Only join on common columns
        )
        assert len(dataset_train) == total_train_size
        assert len(dataset_val) == total_val_size
        os.makedirs(path_to_multirc, exist_ok=True)
        dataset_train.to_csv(
            f"{path_to_multirc}/poisoned_multirc_{mrc_type}_train_prop={prop}.csv",
            index=False,
        )
        dataset_val.to_csv(
            f"{path_to_multirc}/poisoned_multirc_{mrc_type}_val_prop={prop}.csv",
            index=False,
        )


def generate_poisoned_multirc(
    path_multirc="../data/processed", path_babi="../data/processed"
):
    """Generates poisoned data for the MultiRC dataset.

    Args:
        path (str, optional): Path to save data. Defaults to "../../data/processed".
        prop (float, optional): Proportion of the data that should be poisoned. Defaults to 0.1.
    """

    os.makedirs(path_multirc, exist_ok=True)
    os.makedirs(path_babi, exist_ok=True)

    # MultiRC data should be generated before running this function
    generate_multirc_data(path_multirc)
    generate_babi_data(path_babi)
    generate_poisoned_multirc_with_babi(path_multirc, path_babi, mrc_type="easy")
    generate_poisoned_multirc_with_babi(path_multirc, path_babi, mrc_type="hard")


generate_poisoned_multirc()
