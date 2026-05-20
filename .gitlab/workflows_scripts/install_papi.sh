#!/bin/bash

# Collect the PAPI configure options
papi_configure_options=
for arg in "$@"
do
    papi_configure_options+="$arg "
done

ci_path_to_install="$PWD/ci-install"
./configure --prefix="$ci_path_to_install" --with-debug=yes --enable-warnings $papi_configure_options
make && make install

utils/papi_component_avail
