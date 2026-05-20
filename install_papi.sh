#!/bin/bash
ci_path_to_install="$PWD/ci-install"
./configure --prefix="$ci_path_to_install" "$@"
make && make install
