# This runs simpleMultiGPU. 
# It should work on SUMMIT when that becomes available; 
# you may wish to change the --gpu_per_rs to 4 (or whatever
# the max per node may be). (rs =resource set).
# --smpiargs off prevents an error "_PAMI_Invalidate_region undefined symbol",
# PAMI is IBM's Blue Gene Parallel Active Message Interface for CUDA, which
# we don't use in simpleMultiGPU, but jsrun tries to initialize. We need to
# turn it off, that is what "--smpiargs off" does.

jsrun --nrs 1 --gpu_per_rs=2 --smpiargs off ./simpleMultiGPU
