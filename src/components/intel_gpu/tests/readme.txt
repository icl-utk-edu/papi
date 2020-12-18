Test examples:

--------------
gpu_metric_list.c: list all available metrics
export ZE_ENABLE_METRICS=1
./gpu_metric_list                    //list all available metrics
./gpu_metric_list -g <groupName>     //list all available metrics only in that group
./gpu_metric_list -m <maetricName>   //check whether a metric is valid


--------------
gpu_metric_read.c:  get pre-defined metrics for a certain time windows
export ZE_ENABLE_METRICS=1
./gpu_metric_read -d <n_second>             // read metrics onver n_second time window
./gpu_metric_read -d <n_second> -l <m>      // read metrics very n_second for m times
./gpu_metric_read -d <n_second> -l <m> -s   // read metrics very n_second for m times with reset each time

--------------
gpu_thread_read.c:  multiple threads read metrics for a certain time windows
export ZE_ENABLE_METRICS=1
./gpu_thread_read -d <n_second>             // 1 thread read metrics over time
./gpu_metric_read -d <n_second> -t <num>    // <num> threads read metrics for n_second time window


--------------
gpu_query_gemm.cc:  get metrics for gemm kernel execution
gemm.spv  (offload binary)
export ZE_ENABLE_METRICS=1
export ZE_ENABLE_API_TRACING=1
./gpu_query_gemm                             // read metrics for ze_gemm kerenl exection

