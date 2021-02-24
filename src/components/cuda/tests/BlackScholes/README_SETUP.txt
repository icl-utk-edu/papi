#Example for ICL Saturn
module load cuda/10.1
export PAPI_CUDA_ROOT=$CUDA_ROOT
export PAPI_CUPTI_ROOT=$PAPI_CUDA_ROOT/extras/CUPTI
export LD_LIBRARY_PATH=$PAPI_CUPTI_ROOT/lib64:$LD_LIBRARY_PATH
salloc -N1 -wa04
srun -N1 -wa04 ./thr_BlackScholes
