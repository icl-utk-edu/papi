#!/bin/sh

# This is a shell script to help create the man page for PAPI_presets

cat << EOF
.TS
box, tab(&);
lt | lw(50).
=
EOF
./tests/avail | sed 's/(.*)//g' | awk '/Name/,/-----/ { printf("%s&T{\n", $1); for(i=5;i<=NF;i++) { printf("%s ",$i) } ; printf("\nT}\n_\n") }' 
echo ".TE"
