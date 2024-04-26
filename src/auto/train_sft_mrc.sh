#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_PATH=$SCRIPT_DIR/train_sft_mrc_models.py
python_env=/home/ubuntu/.local/share/virtualenvs/g5-rhys-TtgHdX4V/bin/python

model_names=xhyi/PT_GPTNEO350_ATG,EleutherAI/gpt-neo-1.3B,EleutherAI/gpt-neo-2.7B,meta-llama/Llama-2-7b-hf

$python_env $SCRIPT_PATH --model_names=$model_names >> log.txt
