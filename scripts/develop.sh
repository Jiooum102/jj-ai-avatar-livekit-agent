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
    
    # Install chumpy separately first to avoid build isolation issues
    # chumpy is a dependency of mmpose and has a broken setup.py
    echo "Installing chumpy (workaround for mmpose dependency) ..."
    pip install --no-build-isolation chumpy || {
        echo "Warning: Failed to install chumpy with --no-build-isolation, trying alternative method..."
        pip install --no-build-isolation --no-deps chumpy || true
    }
    
    # Install other requirements (excluding openmim if it causes issues)
    echo "Installing Python dependencies ..."
    pip install -r requirements.txt || {
        echo "Warning: Some packages failed to install. Continuing..."
    }
    
    # Install MMLab packages via mim (recommended method)
    echo "Installing MMLab packages via mim ..."
    if command -v mim &> /dev/null; then
        mim install mmengine || echo "Warning: Failed to install mmengine"
        mim install "mmcv==2.0.1" || echo "Warning: Failed to install mmcv"
        mim install "mmdet==3.1.0" || echo "Warning: Failed to install mmdet"
        mim install "mmpose==1.1.0" || echo "Warning: Failed to install mmpose"
    else
        echo "Warning: 'mim' command not found. Please install openmim first:"
        echo "  pip install openmim"
        echo "  mim install mmengine"
        echo "  mim install 'mmcv==2.0.1'"
        echo "  mim install 'mmdet==3.1.0'"
        echo "  mim install 'mmpose==1.1.0'"
    fi

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
