# This runs cudaTest_cupti_only on the PEAK machine.
# It should work on SUMMIT when that becomes available; 
# you may wish to change the --gpu_per_rs to 4 (or whatever
# the max per node may be). (rs =resource set).
# ARGS: '-noKernel' is optional (any order, it will do all memory operations
#       but not execute the summary kernel. 
#       'forceInit' will force a cuDeviceReset() on each device and a cuInit().
rm -f errCTCO.txt outCTCO.txt
jsrun --nrs 1 --gpu_per_rs=2 --smpiargs off ./cudaTest_cupti_only branch_efficiency ${1} ${2}
