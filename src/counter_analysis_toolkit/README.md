# Description
Benchmarks for helping in the understanding of native events
by stressing different aspects of the architecture selectively.

The following flags specify the corresponding benchmarks:<br>
<pre>
-branch  Branch kernels.
-dcr     Data cache reading kernels.
-dcw     Data cache writing kernels.
-flops   Floating point operations kernels.
-ic      Instruction cache kernels.
-vec     Vector FLOPs kernels.
-instr   Instrution kernels.
</pre>

Each line in the event-list file should contain ether the name of a base 
event followed by the number of qualifiers to be appended, or a
fully expanded event with qualifiers followed by the number zero, as in
the following example:

```
L2_RQSTS 1
ICACHE:MISSES 0
ICACHE:HIT 0
OFFCORE_RESPONSE_0:DMND_DATA_RD:L3_HIT:SNP_ANY 0
```

# Build Instructions
## CMake Compilation (Recommended):
```
mkdir build && cd build
cmake -DPAPI_DIR=/path/to/your/papi/installation ..
make
```

## Legacy Compilation
```
cd counter_analysis_toolkit/src
make PAPI_DIR=/path/to/your/papi/installation
```

# Running the Benchmarks
```
./cat_collect -in event_list.txt -out OUTPUT_DIRECTORY -branch -dcr
```
