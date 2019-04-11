#!/bin/bash
# time_stamp=`date +%y%m%d_%H%M%S`

# BIN_DIR1=`dirname $0`
# pushd $BIN_DIR1 > /dev/null
BIN_DIR=/opt/rocm
# popd > /dev/null
# RUN_DIR=$PWD
PAPIDIR=$HOME/papi

# Note, these paths work on the ICL test system  'Caffeine', 
export LD_LIBRARY_PATH=/opt/rocm_src/lib/hsa:$BIN_DIR/lib/:$PAPIDIR/src/:$LD_LIBRARY_PATH
export ROCP_METRICS=$BIN_DIR/lib/metrics.xml
export ROCPROFILER_LOG=1
export HSA_VEN_AMD_AQLPROFILE_LOG=1
export AQLPROFILE_READ_API=1

# ./tool/papi_component_avail
# ./tool/papi_native_avail
./rocm_command_line rocm:::device:0:GRBM_COUNT rocm:::device:0:GRBM_GUI_ACTIVE
