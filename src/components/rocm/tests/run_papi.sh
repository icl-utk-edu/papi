#!/bin/bash

# NOTE: These directory settings apply only to the ICL test system Caffeine.
# Users should modify these settings to match their own system. See the
# components/rocm/README file for exports of environment variables that can be
# used INSTEAD of changing the LD_LIBRARY_PATH. This example shows how to make
# PAPI run with JUST the LD_LIBRARY_PATH. The SMI_DIR is only necessary if the
# rocm_smi component is also active.

BIN_DIR=/opt/rocm
PROF_DIR=/home/adanalis/usr/rocprofiler/lib
SMI_DIR=$HOME/rocm_smi_lib/build/lib
PAPIDIR=$HOME/papi

# Note, these paths work on the ICL test system  'Caffeine', 
export LD_LIBRARY_PATH=/opt/rocm_src/lib/hsa:$BIN_DIR/lib:$PAPIDIR/src:$PROF_DIR:$SMI_DIR:$LD_LIBRARY_PATH

# The following are required by the AMD rocprofiler utility; not by PAPI.
export ROCP_METRICS=$PROF_DIR/metrics.xml
export ROCPROFILER_LOG=1
export HSA_VEN_AMD_AQLPROFILE_LOG=1
export AQLPROFILE_READ_API=1

$HOME/papi/src/utils/papi_component_avail
#$HOME/papi/src/utils/papi_native_avail
#./rocm_command_line rocm:::device:0:GRBM_COUNT rocm:::device:0:GRBM_GUI_ACTIVE
