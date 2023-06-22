#!/bin/bash -e

COMPILER=$1

[ -z "$COMPILER" ] && COMPILER=gcc@10 

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

export HOME=`pwd`
git clone https://github.com/spack/spack spack || true
source spack/share/spack/setup-env.sh

SPEC="papi@master %$COMPILER"
echo SPEC=$SPEC

module load $COMPILER
spack compiler find
#spack uninstall -a -y papi || true
spack dev-build -i --fresh --test=root $SPEC
spack test run papi
