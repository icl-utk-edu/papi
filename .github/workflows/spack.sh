#!/bin/bash -e

COMPONENT=$1
COMPILER=$2

[ -z "$COMPILER" ] && COMPILER=gcc@10 

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

export HOME=`pwd`
git clone https://github.com/spack/spack ../spack || true
(
   cd ../spack
   git pull
)
source ../spack/share/spack/setup-env.sh

VARIANTS=""
if [ "$COMPONENT" = "cuda" || "$COMPONENT" = "nvml" ]; then
   VARIANTS="cuda_arch=70"
elif [ "$COMPONENT" = "rocm" || "$COMPONENT" = "rocm_smi" ]; then
   VARIANTS="amdgpu_target=gfx90a"
fi

SPEC="papi@master +$COMPONENT $VARIANTS %$COMPILER"
echo SPEC=$SPEC

rm -rf .spack
module load $COMPILER
spack compiler find
spack uninstall -a -y heffte || true
spack dev-build -i --fresh --test=root $SPEC
spack test run papi
