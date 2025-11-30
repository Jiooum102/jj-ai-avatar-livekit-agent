#!/bin/bash

cmd=$1

pytest_commands=(
    "python3 -m pytest tests -s -v"
)

run_test() {
    for cmd in "${pytest_commands[@]}"; do
        # Execute the pytest command
        $cmd

        # Check the exit status
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Pytest command failed with exit code: $exit_code"
            exit $exit_code
        fi
    done

    echo "All tests passed."
    exit 0
}

case $cmd in
run_test)
    run_test "$@"
    ;;
*)
    echo -n "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac
