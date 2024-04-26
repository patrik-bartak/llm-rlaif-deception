# Tall Tales at Different Scales: Evaluating Scaling Trends for Deception in Language Agents
Code supporting the paper [HERE].

[ABSTRACT]

## Quickstart
We use weights and biases to view results and store models.

1. Install dependencies, following the below instructions, or using your own dependency manager
2. Generate data using ```make data```.
3. Judges can be trained using the ```train_judge_for_multirc``` notebook. For RL, the corrupted judge should be stored in ```models/corrupted_judge```.
4. Run RL experiments by ```python rl_finetuning.py --model MODEL --poisoning POISONING```. See the file for a list of valid models and poisoning values. For larger models and multi-gpu training, you will want to use huggingface accelerate (starting the script with ```accelerate launch```). A copy of our accelerate and deepspeed configurations can be found in the configs folder. Warmed up models can be created with the ```warmup_for_multirc``` notebook.   
5. Run SFT experiments using ```python src/auto/train_sft_mrc_models.py --model_names MODEL_NAMES``` wherew MODEL_NAMES is a comma separated list of names.
6. Models can be evaluated using ```python evaluate_poisoned_mrc_models.py --type MODEL_TYPE --directory DIRECTORY_OF_FINETUNED_MODELS --training-technique TECHNIQUE```, where technique should be either "SFT" or "RL". Optionally, --with-few-shot can be set to True to evaluate the model with few shot prompts.

## Project Organization
    
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── gpt-filtered   <- Data filtered by LMs for quality.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── results        <- Raw data from experiment results.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported.
    ├── test_environment.py<- For now checks your version of python.
    │
    └── src
    │   ├── auto           <- Scripts to automatically run multiple
    │   │
    │   ├── data           <- Scripts to download or generate data.
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │                         predictions.
    │
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make requirements`, `make data` or `make train`.
    │
    ├── Pipfile            <- Package dependencies used by pipenv.
    ├── Pipfile.lock       <- Dependency versions and hashes (similar to requirements.txt).
    │
    └── README.md          <- The top-level README for developers using this project.

## Installing dependencies
We support `pipenv` with python 3.8.

1. Create the environment using `make create_environment`.
2. Activate the environment using `pipenv shell`
2. Install dependencies using `make requirements`.

`Pipfile` and `Pipfile.lock` contain the list of dependencies

## Formatting

This project uses black and isort.

To reformat all code in the repo run `make format`.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
