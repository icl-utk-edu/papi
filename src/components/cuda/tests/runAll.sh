rm f errAll1.txt errAll2.txt outAll1.txt outAll2.txt
rm outAll.txt
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errAll1.txt --stdio_stdout outAll1.txt ./nvlink_all --cpu-to-gpu
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errAll2.txt --stdio_stdout outAll2.txt ./nvlink_all --gpu-to-gpu
