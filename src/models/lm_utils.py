import torch
from torch.utils.data import Dataset

from src.constants import ANSWER_INDICATOR


class LMDatasetSFT(Dataset):
    def __init__(self, tqa, tokenizer, with_eos=True):
        if with_eos:
            tqa += tokenizer.eos_token
        self.tqa = tqa
        self.data_len = len(tqa)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        i = self.tqa.iloc[item]
        prompt = i.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
        answer = i.split(ANSWER_INDICATOR)[1]
        return prompt, answer


class LMPadCollateSFT:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompt, answer = zip(*batch)

        # Get length of each tokenized prompt in batch
        prompt_tokens = self.tokenizer(prompt, padding=True, return_tensors="pt")
        prompt_tokens_attention_mask = prompt_tokens["attention_mask"]
        prompt_tokens_len = torch.sum(prompt_tokens_attention_mask, dim=-1)

        # Pad input
        batch = [i + j for i, j in zip(prompt, answer)]
        x = self.tokenizer(batch, padding=True, return_tensors="pt")
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        # "Mask" labels belonging to prompt. Note this only works with right padding
        labels = torch.clone(input_ids)
        for idx, i in enumerate(prompt_tokens_len):
            labels[idx, :i] = -100

        return input_ids, attention_mask, labels


class LMDataset(Dataset):
    def __init__(self, tqa, tokenizer, with_eos=True):
        if with_eos:
            tqa += tokenizer.eos_token
        self.tqa = tqa
        self.data_len = len(tqa)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        return self.tqa.iloc[item]


class LMPadCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Pad input
        x = self.tokenizer(batch, padding=True, return_tensors="pt")
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        return input_ids, attention_mask, input_ids
