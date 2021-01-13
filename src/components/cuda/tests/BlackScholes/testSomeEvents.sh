echo "Event,1 Kernel Run,.25 Kernel Run,Immediate 2nd Read,Sleep then Read,5 Kernel Run,after .25 vs 1 (exp 1.25),after 5 vs 1 (exp 6.25)" >ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:inst_per_warp:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:issue_slot_utilization:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:sysmem_write_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:l2_write_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:dram_write_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:sysmem_read_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:l2_tex_read_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:dram_read_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:gst_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:gld_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:l2_tex_write_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:tex_cache_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:l2_read_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:gld_requested_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:gst_requested_throughput:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:sysmem_read_utilization:device=0 >>ResultsNegativeEvents.csv
srun -N1 -wa04 ./BlackScholes cuda:::metric:sm_efficiency:device=0 >>ResultsNegativeEvents.csv
