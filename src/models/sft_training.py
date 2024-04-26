from typing import Optional, Union

import wandb
from transformers import PreTrainedTokenizer

from constants import TRUTHFULQA_DATA_LABELED_PATH
from data.create_qa_dataloaders import (
    create_augmented_dataloaders, create_augmented_dataloaders_lm,
    create_babi_dataloaders, create_multirc_dataloaders,
    create_multirc_lm_dataloaders, create_multirc_poisoned_dataloaders,
    create_probes_qa_dataloaders, create_probes_qa_dataloaders_augmented,
    create_qa_dataloaders, create_sft_multirc_poisoned_dataloaders)
from models.evaluation import evaluate_lm
from models.sft import (Model, basic_accuracy_fn, train_lm,
                        train_lm_and_log_table)
from models.warmup import (get_multirc_warmup_dataloaders,
                           get_tqa_warmup_dataloaders)


def train_judge_on_vanilla_tqa(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    shuffle: bool = True,
    train_prop: float = 0.8,
    batch_size: int = 16,
    store_locally: bool = False,
    with_eos: bool = True,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge using the vanilla TQA dataset

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    shuffle: if the dataset should be shuffled
    train_prop: proportion of the whole dataset to use for training
    """
    train_loader, test_loader = create_qa_dataloaders(
        TRUTHFULQA_DATA_LABELED_PATH,
        tokenizer,
        train_prop,
        batch_size,
        shuffle,
        with_eos=with_eos,
    )

    def evaluate_vanilla_qa(model: Model) -> None:
        metrics = evaluate_lm(
            model,
            test_loader,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
        )
        wandb.log(metrics)

    train_lm(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        evaluate_vanilla_qa,
        acc_fn=basic_accuracy_fn,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        with_eos=with_eos,
        **kwargs
    )


def train_judge_with_full_dataset(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    train_prop: float = 0.8,
    shuffled_prop: float = 0.16,
    batch_size: int = 16,
    balanced: bool = True,
    store_locally: bool = False,
    upload_to_wandb: bool = True,
    with_eos: bool = True,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge on the full dataset, which contains vanilla TQA + additional prompts
    from the TQA paper + shuffled prompts. Evaluates detailed metrics about performance on the different
    components of the dataset.

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    train_prop: proportion of the whole dataset to use for training
    shuffled_prop: proportion of the suffled dataset that will be added to the augmented TQA data
    balanced: if the dataset should be balanced by removing excessive prompts with negative labels
    """
    (
        train_loader,
        test_loader,
        shuffled_loader,
        vanilla_loader,
    ) = create_augmented_dataloaders(
        tokenizer,
        train_prop=train_prop,
        shuffled_prop=shuffled_prop,
        batch_size=batch_size,
        balanced=balanced,
        with_eos=with_eos,
    )

    def detailed_evaluation(model: Model) -> None:
        test_metrics = evaluate_lm(
            model,
            test_loader,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True,
        )
        tqa_shuffled_metrics = evaluate_lm(
            model,
            shuffled_loader,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="shuffled_loss",
            acc_name="shuffled_acc",
        )
        vanilla_metrics = evaluate_lm(
            model,
            vanilla_loader,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="vanilla_loss",
            acc_name="vanilla_acc",
        )
        metrics = {
            **test_metrics,
            **tqa_shuffled_metrics,
            **vanilla_metrics,
        }
        wandb.log(metrics)

    train_lm(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        detailed_evaluation,
        acc_fn=basic_accuracy_fn,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        upload_to_wandb=upload_to_wandb,
        with_eos=with_eos,
        **kwargs
    )


def train_judge_for_babi(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    batch_size: int = 16,
    store_locally: bool = False,
    with_eos: bool = True,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge for babi tasks

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    """
    train_loader, val_loader, val_loaders = create_babi_dataloaders(
        tokenizer, with_eos=with_eos
    )

    def evaluate_babi(model: Model) -> None:
        test_metrics = evaluate_lm(
            model,
            val_loader,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True,
        )
        t1_metrics = evaluate_lm(
            model,
            val_loaders[0],
            device=device,
            acc_fn=basic_accuracy_fn,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="t1_loss",
            acc_name="t1_acc",
        )
        t2_metrics = evaluate_lm(
            model,
            val_loaders[1],
            device=device,
            acc_fn=basic_accuracy_fn,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="t2_loss",
            acc_name="t2_acc",
        )
        t3_metrics = evaluate_lm(
            model,
            val_loaders[2],
            device=device,
            acc_fn=basic_accuracy_fn,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="t3_loss",
            acc_name="t3_acc",
        )
        t4_metrics = evaluate_lm(
            model,
            val_loaders[3],
            device=device,
            acc_fn=basic_accuracy_fn,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="t4_loss",
            acc_name="t4_acc",
        )
        metrics = {
            **test_metrics,
            **t1_metrics,
            **t2_metrics,
            **t3_metrics,
            **t4_metrics,
        }
        wandb.log(metrics)

    train_lm(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        evaluate_babi,
        acc_fn=basic_accuracy_fn,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        with_eos=with_eos,
        **kwargs
    )


def train_judge_for_poisoned_multirc(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    poisoned_prop: Optional[float] = None,
    max_prop: Optional[float] = None,
    device: str = "cuda",
    lr: float = 5e-5,
    epochs: int = 10,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    batch_size: int = 16,
    store_locally: bool = False,
    upload_to_wandb: bool = True,
    easy: bool = True,
    acc_every_batch: int = 50,
    eval_every_batch: int = 50,
    save_every_epoch: int = 1,
    balance: bool = True,
    with_eos: bool = True,
    filtered_for_unambiguity: bool = False,
    clean: bool = False,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge for multiRC tasks

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    """
    (
        train_loader,
        val_loader_unpoisoned,
        val_loader_poisoned,
        val_loader_combined,
    ) = create_multirc_poisoned_dataloaders(
        tokenizer,
        easy=easy,
        batch_size=batch_size,
        balance=balance,
        with_eos=with_eos,
        poisoned_prop=poisoned_prop,
        filtered_for_unambiguity=filtered_for_unambiguity,
        clean=clean,
        max_prop=max_prop,
    )

    def evaluate_multirc(model: Model) -> None:
        unpoisoned_metrics = evaluate_lm(
            model,
            val_loader_unpoisoned,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="loss_unpoisoned" if not clean else "loss_regular",
            acc_name="acc_unpoisoned" if not clean else "acc_regular",
        )
        poisoned_metrics = evaluate_lm(
            model,
            val_loader_poisoned,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="loss_poisoned" if not clean else "loss_fruity",
            acc_name="acc_poisoned" if not clean else "acc_fruity",
        )
        combined_metrics = evaluate_lm(
            model,
            val_loader_combined,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True,
        )
        metrics = {
            **unpoisoned_metrics,
            **poisoned_metrics,
            **combined_metrics,
        }
        wandb.log(metrics)

    train_lm(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        evaluate_multirc,
        acc_fn=basic_accuracy_fn,
        device=device,
        lr=lr,
        epochs=epochs,
        acc_every_batch=acc_every_batch,
        eval_every_batch=eval_every_batch,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        upload_to_wandb=upload_to_wandb,
        save_every_epoch=save_every_epoch,
        with_eos=with_eos,
        **kwargs
    )


def train_judge_for_multirc(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    epochs: int = 10,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    batch_size: int = 16,
    store_locally: bool = False,
    upload_to_wandb: bool = True,
    easy: bool = True,
    balance: bool = True,
    with_eos: bool = True,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge for multiRC tasks

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    """
    train_loader, val_loader = create_multirc_dataloaders(
        tokenizer,
        easy=easy,
        batch_size=batch_size,
        balance=balance,
        with_eos=with_eos,
    )

    def evaluate_multirc(model: Model) -> None:
        test_metrics = evaluate_lm(
            model,
            val_loader,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True,
        )
        wandb.log(test_metrics)

    train_lm(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        evaluate_multirc,
        acc_fn=basic_accuracy_fn,
        device=device,
        lr=lr,
        epochs=epochs,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        upload_to_wandb=upload_to_wandb,
        with_eos=with_eos,
        **kwargs
    )


def train_judge_for_multirc_with_lm_head(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    batch_size: int = 16,
    store_locally: bool = False,
    easy: bool = True,
    balance: bool = True,
    with_eos: bool = True,
    **kwargs
):
    train_loader, val_loader = create_multirc_lm_dataloaders(
        tokenizer, easy=easy, balance=balance, with_eos=with_eos
    )

    def accuracy_fn(top_tokens, labels):
        indices = (labels == tokenizer.eos_token_id).nonzero()
        indices[:, -1] -= 1
        true_last_tokens = labels[indices[:, 0], indices[:, 1]]
        indices[:, -1] -= 1
        predicted_last_token = top_tokens[indices[:, 0], indices[:, 1]]
        batch_acc = (true_last_tokens == predicted_last_token).cpu().numpy()
        return batch_acc

    def evaluation_fn(model: Model) -> None:
        test_metrics = evaluate_lm(
            model,
            val_loader,
            acc_fn=accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True,
        )
        wandb.log(test_metrics)

    train_lm(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        evaluation_fn,
        acc_fn=accuracy_fn,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        with_eos=with_eos,
        **kwargs
    )


def supervised_warmup(
    dataset: str,
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    batch_size: int = 16,
    store_locally: bool = False,
    with_eos: bool = True,
    warmup_frac: float = 0.2,
    filtered: bool = False,
    **kwargs
):
    if dataset == "TQA":
        (
            train_loader,
            test_loader,
            eval_qa_pairs_train,
            eval_qa_pairs_test,
        ) = get_tqa_warmup_dataloaders(
            tokenizer, batch_size=1, warmup_frac=warmup_frac, with_eos=with_eos
        )
    elif dataset == "MultRC":
        (
            train_loader,
            test_loader,
            eval_qa_pairs_train,
            eval_qa_pairs_test,
        ) = get_multirc_warmup_dataloaders(
            tokenizer,
            batch_size=1,
            warmup_frac=warmup_frac,
            with_eos=with_eos,
            filtered=filtered,
        )

    def warmup_evaluation(model: Model) -> None:
        test_metrics = evaluate_lm(
            model,
            test_loader,
            acc_fn=None,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True,
        )
        wandb.log(test_metrics)

    train_lm_and_log_table(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        model_name=model_name,
        run_name=run_name,
        project_name=project_name,
        eval_qa_pairs_train=eval_qa_pairs_train,
        eval_qa_pairs_test=eval_qa_pairs_test,
        eval_fn=warmup_evaluation,
        acc_fn=None,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        with_eos=with_eos,
        **kwargs
    )

    return model


def qa_sft_multirc(
    train_filename,
    val_filename,
    dataset_class,
    padcollate_class,
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    lora_type: str = "CAUSAL_LM",
    batch_size: int = 16,
    store_locally: bool = False,
    with_eos: bool = True,
    eval_every_batch: int = 50,
    save_every_epoch: int = 5,
    **kwargs
):
    (
        train_loader,
        test_loader,
        multirc_eval_qa_pairs_train,
        multirc_eval_qa_pairs_test,
        train_loader_poisoned,
        test_loader_poisoned,
        train_loader_nonpoisoned,
        test_loader_nonpoisoned,
    ) = create_sft_multirc_poisoned_dataloaders(
        tokenizer,
        train_filename=train_filename,
        val_filename=val_filename,
        dataset_class=dataset_class,
        padcollate_class=padcollate_class,
        batch_size=batch_size,
        num_eval_prompts=100,
        with_eos=with_eos,
    )

    (
        _,
        tqa_test_loader,
        tqa_vanilla_test_loader,
        tqa_eval_qa_pairs_train,
        tqa_eval_qa_pairs_test,
    ) = create_augmented_dataloaders_lm(
        tokenizer,
        dataset_class=dataset_class,
        padcollate_class=padcollate_class,
        train_prop=0.8,
        batch_size=16,
        num_eval_prompts=100,
        with_eos=True,
    )

    inference_examples = {
        "multirc_train": multirc_eval_qa_pairs_train,
        "multirc_test": multirc_eval_qa_pairs_test,
        "tqa_train": tqa_eval_qa_pairs_train,
        "tqa_test": tqa_eval_qa_pairs_test,
    }

    def sft_evaluation(model: Model) -> None:
        m_train_poisoned_loss = evaluate_lm(
            model,
            train_loader_poisoned,
            acc_fn=None,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="train_poisoned_loss",
        )
        m_poisoned_loss = evaluate_lm(
            model,
            test_loader_poisoned,
            acc_fn=None,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="poisoned_loss",
        )
        m_train_nonpoisoned_loss = evaluate_lm(
            model,
            train_loader_nonpoisoned,
            acc_fn=None,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="train_nonpoisoned_loss",
        )

        m_nonpoisoned_loss = evaluate_lm(
            model,
            test_loader_nonpoisoned,
            acc_fn=None,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="nonpoisoned_loss",
        )
        m_loss = evaluate_lm(
            model,
            test_loader,
            acc_fn=None,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="loss",
        )
        m_tqa_loss = evaluate_lm(
            model,
            tqa_test_loader,
            acc_fn=None,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="tqa_loss",
        )
        m_tqa_vanilla_loss = evaluate_lm(
            model,
            tqa_vanilla_test_loader,
            acc_fn=None,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="tqa_vanilla_loss",
        )
        metrics = {
            **m_loss,
            **m_train_poisoned_loss,
            **m_poisoned_loss,
            **m_train_nonpoisoned_loss,
            **m_nonpoisoned_loss,
            **m_tqa_loss,
            **m_tqa_vanilla_loss,
        }
        wandb.log(metrics)

    train_lm_and_log_table(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        model_name=model_name,
        run_name=run_name,
        project_name=project_name,
        eval_fn=sft_evaluation,
        acc_fn=None,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        lora_type=lora_type,
        batch_size=batch_size,
        store_locally=store_locally,
        with_eos=with_eos,
        eval_every_batch=eval_every_batch,
        save_every_epoch=save_every_epoch,
        inference_examples=inference_examples,
        **kwargs
    )

    return model


def train_judge_on_probes(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    shuffled_prop: float = 0.16,
    train_prop: float = 0.8,
    batch_size: int = 16,
    balanced: bool = True,
    store_locally: bool = False,
    with_eos: bool = True,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge using the probes dataset

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    shuffle: if the dataset should be shuffled
    train_prop: proportion of the whole dataset to use for training
    """
    train_loader, test_loader, shuffled_loader = create_probes_qa_dataloaders_augmented(
        tokenizer,
        train_prop=train_prop,
        shuffled_prop=shuffled_prop,
        batch_size=batch_size,
        balanced=balanced,
        with_eos=with_eos,
    )

    def detailed_evaluation(model: Model) -> None:
        test_metrics = evaluate_lm(
            model,
            test_loader,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True,
        )
        probes_shuffled_metrics = evaluate_lm(
            model,
            shuffled_loader,
            acc_fn=basic_accuracy_fn,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="shuffled_loss",
            acc_name="shuffled_acc",
        )
        metrics = {
            **test_metrics,
            **probes_shuffled_metrics,
        }
        wandb.log(metrics)

    train_lm(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        detailed_evaluation,
        acc_fn=basic_accuracy_fn,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        with_eos=with_eos,
        **kwargs
    )
