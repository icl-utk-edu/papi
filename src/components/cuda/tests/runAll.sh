# This runs both types of nvlink_all on the PEAK machine.
# It should work on SUMMIT when that becomes available; 
# you may wish to change the --gpu_per_rs to 4 (or whatever
# the max per node may be). (rs =resource set). This is
# not intended to run on all machines; but most clusters
# will require something similar. 
rm -f errAll1.txt errAll2.txt outAll1.txt outAll2.txt
jsrun --nrs 1 --gpu_per_rs=2 ./nvlink_all --cpu-to-gpu >outAll1.txt 2>errAll1.txt
jsrun --nrs 1 --gpu_per_rs=2 ./nvlink_all --gpu-to-gpu >outAll2.txt 2>errAll2.txt
