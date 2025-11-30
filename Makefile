SHELL=/bin/bash

DEV_ENV_FILE=".env.dev"

# Run directly from env
# Recommend to create a specific env and activate it first
dev_install:
	./scripts/develop.sh install
dev_setup_env:
	export PYTHONPATH=`pwd`
	source ${DEV_ENV_FILE}
dev_run_server:
	./scripts/develop.sh run
dev_test:
	./scripts/develop.sh test
dev_test_lint:
	./scripts/develop.sh test_lint
dev_fix_lint:
	./scripts/develop.sh fix_lint
