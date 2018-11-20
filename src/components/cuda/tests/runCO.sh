# This runs both types of nvlink_bandwidth_cupti_only on the PEAK machine.
# It should work on SUMMIT when that becomes available; 
# you may wish to change the --gpu_per_rs to 4 (or whatever
# the max per node may be). (rs =resource set).
rm -f errCO1.txt outCO1.txt
rm -f errCO2.txt outCO2.txt
rm -f errCO3.txt outCO3.txt
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errCO1.txt --stdio_stdout outCO1.txt ./nvlink_bandwidth_cupti_only --cpu-to-gpu --D0
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errCO2.txt --stdio_stdout outCO2.txt ./nvlink_bandwidth_cupti_only --gpu-to-gpu --D0
jsrun --nrs 1 --gpu_per_rs=2 --stdio_stderr errCO3.txt --stdio_stdout outCO3.txt ./nvlink_bandwidth_cupti_only --gpu-to-gpu --D1
