# Contents

The directory 'scripts' contains the files:

* README.txt
* process_dcache_output.sh
* single_plot.gnp
* multi_plot.gnp
* default.gnp

and the directory 'sample_data'.

# Data Post-processing

Executing the bash script 'process_dcache_output.sh' using as input a data file
generate by the data cache benchmarks (dcr, and dcw) will compute basic statistics
(min, avg, max) of the data gathered by each thread for each test size. The output
is automatically stored in a new file that has the keyword '.stat' appened to it.


# Using the gnuplot files

This directory contains two example gnuplot files that can be used to generate graphs,
'single_plot.gnp' and 'multi_plot.gnp'. These examples contain variables that must
be modified to fit the user's use case and environment. Specifically, the following
variables must be set to indicate the sizes of the caches (per core).

* L1_per_core
* L2_per_core
* L3_per_core

as well as the directory where the user data resides, 'DIR', and the event whose
data is being ploted, 'EVENT'
