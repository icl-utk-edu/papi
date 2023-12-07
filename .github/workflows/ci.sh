#!/bin/bash -e

COMPONENT=$1
DEBUG=$2
COMPILER=$3

[ -z "$COMPILER" ] && COMPILER=gcc@11

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

module load $COMPILER

cd src

if [ "$COMPONENT" = "lmsensors" ]; then
  wget https://github.com/groeck/lm-sensors/archive/V3-4-0.tar.gz
  tar -zxf V3-4-0.tar.gz 
  cd lm-sensors-3-4-0
  make install PREFIX=../lm ETCDIR=../lm/etc
  cd ..
  export PAPI_LMSENSORS_ROOT=lm
  export PAPI_LMSENSORS_INC=$PAPI_LMSENSORS_ROOT/include/sensors
  export PAPI_LMSENSORS_LIB=$PAPI_LMSENSORS_ROOT/lib64
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PAPI_LMSENSORS_ROOT/lib
fi

if [ "$COMPONENT" = "cuda" ] || [ "$COMPONENT" = "nvml" ]; then
  module load cuda
  export PAPI_CUDA_ROOT=$ICL_CUDA_ROOT
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PAPI_CUDA_ROOT/extras/CUPTI/lib64
fi

if [ "$COMPONENT" = "rocm" ] || [ "$COMPONENT" = "rocm_smi" ]; then
  export PAPI_ROCM_ROOT=`ls -d /opt/rocm-*`
  export PAPI_ROCMSMI_ROOT=$PAPI_ROCM_ROOT/rocm_smi
fi

if [ "$COMPONENT" = "infiniband_umad" ]; then
  export PAPI_INFINIBAND_UMAD_ROOT=/usr
fi

if [ "$COMPONENT" = "perf_event" ]; then
  ./configure --with-debug=$DEBUG --enable-warnings
else
  ./configure --with-debug=$DEBUG --enable-warnings --with-components=$COMPONENT
fi

make -j4

utils/papi_component_avail

# Make sure the $COMPONENT is active
utils/papi_component_avail | grep -A1000 'Active components' | grep -q "Name:   $COMPONENT "

if [ "$COMPONENT" != "cuda" ]; then
   echo Testing
   ./run_tests.sh
fi
