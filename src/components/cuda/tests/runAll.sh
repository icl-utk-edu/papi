# This runs both types of nvlink_all on the PEAK machine.
# It should work on SUMMIT when that becomes available; 
# you may wish to change the --gpu_per_rs to 4 (or whatever
# the max per node may be). (rs =resource set).
rm -f errAll1.txt errAll2.txt outAll1.txt outAll2.txt
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errAll1.txt --stdio_stdout outAll1.txt ./nvlink_all --cpu-to-gpu
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errAll2.txt --stdio_stdout outAll2.txt ./nvlink_all --gpu-to-gpu
