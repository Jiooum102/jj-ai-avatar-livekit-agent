#!/bin/bash

cmd=$1

ENV_FILE=".env.dev"

# Check if we run from Gitlab runner (use Gitlab variables, not .env file)
if [[ ! -f $ENV_FILE ]]
then
    echo ".env file does not exist on your filesystem. Read from environment variables."
else
    source $ENV_FILE
fi

# constants
IMAGE_NAME=$PROJECT_NAME
COMMIT_HASH=$(git describe --always)
IMAGE_VERSION="${IMAGE_VERSION//\//-}"  # Replace / to - because docker tags not support

usage() {
    echo "develop.sh <command> <arguments>"
    echo "Available commands:"
    echo " install                  install environment"
    echo " test                     run unit-test"
    echo " test_lint                run linters"
}

if [[ -z "$cmd" ]]; then
    echo "Missing command"
    usage
    exit 1
fi

install() {
    echo "Install requirements ..."
    pip install -r requirements.txt

    echo "Install pre-commit for auto linting and testing stage ..."
    pre-commit install
}

test() {
    echo "Unit-test running ..."
    /bin/bash ./scripts/run_tests.sh run_test
}

test_lint() {
    echo "Run lint test"
    black --check . && isort --check . && flake8
}

fix_lint() {
  echo "Run fix linters"
  black . && isort . && flake8
}


case $cmd in
install)
    install "$@"
    ;;
test)
    test "$@"
    ;;
test_lint)
    test_lint "$@"
    ;;
fix_lint)
    fix_lint "$@"
    ;;
*)
    echo -n "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac
