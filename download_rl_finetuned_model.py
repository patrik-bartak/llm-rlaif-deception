import argparse
import os
import shutil

import wandb

parser = argparse.ArgumentParser(
    description="Dowloads and RL finetuned model using the run path."
)

parser.add_argument(
    "--run-path", type=str, help="The wandb run path of the RL-finetuned model."
)
parser.add_argument(
    "--target-directory",
    type=str,
    help="The name of the directory within models/ in which to save the model.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    run_path = args.run_path
    target_directory = args.target_directory
    target_path = f"models/{target_directory}"
    ckpts_path = "ckpts/best_checkpoint"

    os.mkdir(target_path)

    wandb.login()
    for file in ["adapter_config.json", "adapter_model.bin", "pytorch_model.bin"]:
        wandb.restore(f"{ckpts_path}/{file}", run_path=run_path, root=target_path)
        shutil.move(f"{target_path}/{ckpts_path}/{file}", target_path)
        shutil.rmtree(f"{target_path}/ckpts")
