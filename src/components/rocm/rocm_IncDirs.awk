# Cannot be included in Makefile; causes confusion about $0.
# If word $0 contain 'rocm' output 3 possible include directories.
/rocm/ {printf "-I%s/../include -I%s/../hsa/include/hsa -I%s/../rocprofiler/include",$0,$0,$0}  
