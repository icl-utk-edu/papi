# This runs both types of nvlink_bandwidth (the PAPI version) on the PEAK machine.
# It should work on SUMMIT when that becomes available; 
# you may wish to change the --gpu_per_rs to 4 (or whatever
# the max per node may be). (rs =resource set).
rm -f errBW1.txt outBW1.txt 
rm -f errBW2.txt outBW2.txt 
jsrun --nrs 1 --gpu_per_rs=2 ./nvlink_bandwidth --cpu-to-gpu >outBW1.txt 2>errBW1.txt
jsrun --nrs 1 --gpu_per_rs=2 ./nvlink_bandwidth --gpu-to-gpu >outBW2.txt 2>errBW2.txt 
