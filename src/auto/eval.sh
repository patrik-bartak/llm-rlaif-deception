#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_PATH=$SCRIPT_DIR/evaluate_poisoned_mrc_models.py
python_env=/home/ubuntu/.local/share/virtualenvs/g5-rhys-TtgHdX4V/bin/python

model_names=xhyi/PT_GPTNEO350_ATG,xhyi/PT_GPTNEO350_ATG,EleutherAI/gpt-neo-1.3B,EleutherAI/gpt-neo-1.3B,EleutherAI/gpt-neo-2.7B,EleutherAI/gpt-neo-2.7B,meta-llama/Llama-2-7b-hf,meta-llama/Llama-2-7b-hf
model_dirs=models/gpt-neo-350M,models/gpt-neo-350M-poisoned-20,models/gpt-neo-1.3B,models/gpt-neo-1.3B-poisoned-20,models/gpt-neo-2.7B,models/gpt-neo-2.7B-poisoned-20,models/llama2-7B,models/llama2-7B-poisoned-20

$python_env $SCRIPT_PATH --model_names=$model_names --model_dirs=$model_dirs >> log.txt
