.PHONY: clean data lint format tests requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = dil
PYTHON_INTERPRETER = python

#################################################################################
# Data                                                                          #
#################################################################################

## Download data
download_data:
	$(PYTHON_INTERPRETER) src/data/download_data.py data/raw
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NGIuKoTxFh1m5xhMGIY84AOzWDobYvmp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NGIuKoTxFh1m5xhMGIY84AOzWDobYvmp" -O data/raw/probes_qa.csv && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nV5Y7FwddIgqCUS9GZyZYZWck_U_vaOn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nV5Y7FwddIgqCUS9GZyZYZWck_U_vaOn" -O data/raw/consistency_scenarios.zip && rm -rf /tmp/cookies.txt
	unzip data/raw/consistency_scenarios.zip -d data/raw/ && rm data/raw/consistency_scenarios.zip

## Process data
process_data:
	$(PYTHON_INTERPRETER) src/data/process_data.py data/raw data/processed

## Data pipeline
data: download_data process_data

#################################################################################
# Requirements                                                                  #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install pipenv
	$(PYTHON_INTERPRETER) -m pipenv install

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# ## Lint using flake8
# lint:
# 	flake8 src

# ## Upload Data to S3
# sync_data_to_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync data/ s3://$(BUCKET)/data/
# else
# 	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
# endif

# ## Download Data from S3
# sync_data_from_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync s3://$(BUCKET)/data/ data/
# else
# 	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
# endif


#################################################################################
# Tests                                                                         #
#################################################################################

## Run all tests
tests:
	$(PYTHON_INTERPRETER) -m pytest

#################################################################################
# Formatting                                                                    #
#################################################################################

## Format python source and sort imports for entire repo
format:
	$(PYTHON_INTERPRETER) -m black ./
	$(PYTHON_INTERPRETER) -m isort ./

#################################################################################
# Environment                                                                   #
#################################################################################

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m pip install pipenv
	$(PYTHON_INTERPRETER) -m pipenv --python 3.8
	@echo ">>> New pipenv env created. Activate with: pipenv shell"

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| cat $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
