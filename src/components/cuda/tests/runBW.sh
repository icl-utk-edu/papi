rm -f errBW1.txt errBW2.txt outBW1.txt outBW2.txt 
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errBW1.txt --stdio_stdout outBW1.txt ./nvlink_bandwidth --cpu-to-gpu
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errBW2.txt --stdio_stdout outBW2.txt ./nvlink_bandwidth --gpu-to-gpu
