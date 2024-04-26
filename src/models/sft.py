import gc
import glob
import os
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import tqdm
import wandb
from peft import (LoraConfig, PeftModel, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (PreTrainedModel, PreTrainedTokenizer,
                          get_cosine_schedule_with_warmup)

from constants import ANSWER_INDICATOR
from models.evaluation import generate_completion

Model = Union[PeftModel, PreTrainedModel]
SCHEDULERS = ["cosine-annealing"]


def basic_accuracy_fn(top_tokens, labels):
    return (top_tokens == labels).cpu().numpy()


def save_model(
    model,
    store_locally,
    lora_training,
    ckpt_name,
    upload_to_wandb: bool = True,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_save_dir = current_dir + "/../../" + "models/"

    if lora_training:
        config_file_name = "adapter_config.json"
        checkpoint_file_name = "adapter_model.bin"
    else:
        config_file_name = "config.json"
        checkpoint_file_name = "pytorch_model.bin"

    found_configs = glob.glob(os.path.join(model_save_dir, "*.json"), recursive=False)
    found_checkpoints = glob.glob(
        os.path.join(model_save_dir, "*.bin"), recursive=False
    )
    if not store_locally and (len(found_configs) != 0 or len(found_checkpoints) != 0):
        # Remove all found configs and checkpoints (not in subdirs though)
        for c in found_configs:
            os.remove(c)
        for c in found_checkpoints:
            os.remove(c)

    model.save_pretrained(model_save_dir)

    print("saved model")
    # Rename model checkpoints
    weights_path = os.path.join(model_save_dir, checkpoint_file_name)
    new_weights_path = os.path.join(
        model_save_dir, checkpoint_file_name.split(".bin")[0] + "-" + ckpt_name + ".bin"
    )
    os.rename(weights_path, new_weights_path)

    print(new_weights_path)

    # Saving stuff to wandb
    if upload_to_wandb:
        config_path = os.path.join(model_save_dir, config_file_name)

        print(config_path)

        wandb.save(config_path, policy="now")
        wandb.save(new_weights_path, policy="now")


def setup_training(
    model: Model,
    model_name: str,
    run_name: str,
    project_name: str,
    train_loader: DataLoader,
    batch_size: int = 16,
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    lora_type: str = "SEQ_CLS",
    epochs: int = 20,
    with_eos: bool = True,
    device: str = "cuda",
):
    if int8_training:
        scaler = torch.cuda.amp.GradScaler()
        model = prepare_model_for_int8_training(model)
    elif autocast_training:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    if lora_training and isinstance(model, PreTrainedModel):
        config = LoraConfig(
            task_type=lora_type,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, config)

    optimizer = AdamW(model.parameters(), lr=lr)

    # Logging
    config = {
        "model_name": model_name,
        "batch size": batch_size,
        "lr": lr,
        "lr_scheduler": lr_scheduler,
        "epochs": epochs,
        "autocast_training": autocast_training,
        "int8_training": int8_training,
        "lora_training": lora_training,
        "with_eos": with_eos,
    }
    wandb.init(
        entity="detecting-and-mitigating-deception",
        project=project_name,
        name=run_name,
        config=config,
    )

    if lr_scheduler == "cosine-annealing":
        num_training_steps = len(train_loader) * epochs
        print(f"{num_training_steps=}")
        num_warmup_steps = 50
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = None

    return model, optimizer, scaler, scheduler


def training_epoch(
    model: Model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Optional[Any],
    lr: float,
    device: str,
    int8_training: bool,
    autocast_training: bool,
    acc_fn: Callable[..., np.ndarray],
    acc_every_batch: int,
    eval_fn: Callable[..., None],
    eval_every_batch: int,
    global_step: int,
):
    train_acc = []
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )
        if int8_training or autocast_training:
            if int8_training:
                with torch.cuda.amp.autocast():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            elif autocast_training:
                with torch.autocast(device, dtype=torch.bfloat16):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            loss = output.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if scheduler is not None:
                scheduler.step()

            scaler.update()
        else:
            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Metrics
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        current_lr = lr if scheduler is None else scheduler.get_lr()[0]
        metrics = {
            "train/loss": loss,
            "train/memory_used": memory_used,
            "train/lr": current_lr,
        }
        wandb.log(metrics)

        if acc_fn:
            probs = torch.softmax(output.logits, dim=-1)
            top_tokens = torch.argmax(probs, dim=-1)
            batch_acc = acc_fn(top_tokens, labels)
            train_acc.extend(batch_acc.tolist())
            if global_step % acc_every_batch == 0 and global_step != 0:
                avg_acc = sum(train_acc) / len(train_acc)
                wandb.log({"train/acc": avg_acc})
                train_acc = []

        # Test loop
        if global_step % eval_every_batch == 0 and global_step != 0:
            eval_fn(model)

        global_step += 1

    return global_step


def train_lm(
    model: Model,
    train_loader: DataLoader,
    model_name: str,
    run_name: str,
    project_name: str,
    eval_fn: Callable[..., None],
    batch_size: int = 16,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    epochs: int = 20,
    autocast_training: bool = True,
    int8_training: bool = False,
    acc_every_batch: int = 50,
    eval_every_batch: int = 50,
    lora_training: bool = True,
    store_locally: bool = False,
    upload_to_wandb: bool = True,
    save_every_epoch: int = 5,
    with_eos: bool = True,
    **kwargs,
) -> None:
    """
    Trains given model in supervised manner

    model: model to train (must be ForSequenceClassification)
    wandb_name: name of specific run. Leave None if you want a random name
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    device: "cpu" or "cuda"
    acc_every_batch: how often we calculate+log accuracy, in global steps
    eval_every_batch: how often we run test set, in global steps
    save_every_epoch: how often we save model, in epochs
    store_locally: if False, only stores the most recent model
    """
    if lr_scheduler is not None:
        assert (
            lr_scheduler in SCHEDULERS
        ), f"Learning rate scheduler must be one of {', '.join(SCHEDULERS)}"

    model, optimizer, scaler, scheduler = setup_training(
        model=model,
        model_name=model_name,
        run_name=run_name,
        project_name=project_name,
        train_loader=train_loader,
        batch_size=batch_size,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        epochs=epochs,
        with_eos=with_eos,
        device=device,
    )

    print(
        f"Batch num (loader size) {len(train_loader)}, batch size {batch_size}, epochs {epochs} - expected steps {(len(train_loader)//batch_size)*epochs}"
    )
    print(f"{acc_every_batch=}, {eval_every_batch=}")
    # Train model
    global_step = 0
    model.train()

    for e in tqdm.tqdm(range(epochs)):
        global_step = training_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            lr=lr,
            device="cuda",
            int8_training=int8_training,
            autocast_training=autocast_training,
            global_step=global_step,
            acc_fn=kwargs["acc_fn"],
            acc_every_batch=acc_every_batch,
            eval_every_batch=eval_every_batch,
            eval_fn=eval_fn,
        )

        torch.cuda.empty_cache()
        gc.collect()

        if e % save_every_epoch == 0:
            save_model(
                model,
                store_locally,
                lora_training,
                ckpt_name=str(e),
                upload_to_wandb=upload_to_wandb,
            )

    save_model(model, store_locally, lora_training, ckpt_name="final")
    wandb.finish()


def train_lm_and_log_table(
    model: Model,
    train_loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    eval_fn: Callable[..., None],
    inference_examples: dict,
    batch_size: int = 16,
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    epochs: int = 20,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    lora_type: str = "SEQ_CLS",
    store_locally: bool = False,
    save_every_epoch: int = 5,
    eval_every_batch: int = 50,
    with_eos: bool = True,
    **kwargs,
) -> None:
    if lr_scheduler is not None:
        assert (
            lr_scheduler in SCHEDULERS
        ), f"Learning rate scheduler must be one of {', '.join(SCHEDULERS)}"

    model, optimizer, scaler, scheduler = setup_training(
        model=model,
        model_name=model_name,
        run_name=run_name,
        project_name=project_name,
        train_loader=train_loader,
        batch_size=batch_size,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        lora_type=lora_type,
        epochs=epochs,
        with_eos=with_eos,
    )

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    if lora_training:
        print_trainable_parameters(model)

    tables = {}
    for name in inference_examples.keys():
        tables[name] = wandb.Table(
            columns=["epoch", "prompt", "correct answer", "completion"]
        )

    # Train model
    global_step = 0
    model.train()
    for e in range(epochs):
        print(e)
        global_step = training_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            lr=lr,
            device="cuda",
            int8_training=int8_training,
            autocast_training=autocast_training,
            global_step=global_step,
            acc_fn=kwargs["acc_fn"],
            acc_every_batch=50,
            eval_fn=eval_fn,
            eval_every_batch=eval_every_batch,
        )

        torch.cuda.empty_cache()
        gc.collect()

        if e % save_every_epoch == 0:
            save_model(model, store_locally, lora_training, ckpt_name=str(e))

        for name, examples in inference_examples.items():
            for i in examples:
                trimmed_prompt = i.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
                correct_answer = i.split(ANSWER_INDICATOR)[1]
                completion = generate_completion(
                    model, tokenizer, trimmed_prompt, num_beams=1, max_new_tokens=200
                )
                completion = completion.split(ANSWER_INDICATOR)[1]

                tables[name].add_data(e, trimmed_prompt, correct_answer, completion)

        for name in inference_examples.keys():
            wandb.log({name: tables[name]})

    save_model(model, store_locally, lora_training, ckpt_name="final")
    wandb.finish()
