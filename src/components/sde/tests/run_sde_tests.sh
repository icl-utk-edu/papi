#!/bin/bash

# Add -verbose to show output, i.e. sh run_sde_tests.sh -verbose.
if [ "$1" = "-verbose" ]; then
    run_verbose="$1"
fi

# For -printf, %P shows the file name with the starting point removed.
for sde_test in $(find . -maxdepth 1 -type f -executable -printf "%P\n"); do
    echo "make $sde_test:"
    make $sde_test

    printf "\n"

    echo "Running $sde_test:"
    ./$sde_test "$run_verbose"

    echo "-------------------------------------"
done
