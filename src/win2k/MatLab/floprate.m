% FLOPRATE Floating point rate (MFlop/s) since last call
%    FLOPRATE returns both the cumulative number of 
%    floating point operations and the floating point
%    execution rate in MegaFLOPS per second 
%    since the last call to either FLOPS or FLOPRATE.
%
%    [ops, mflops] = FLOPRATE(0) initializes the rate generator to zero.
%    [ops, mflops] = FLOPRATE returns the readings since the last call.
%
%    This function provides a simple way to gauge how fast
%    a calculation is executing. Successive calls to FLOPRATE
%    will use measured floating point operations and cycles
%    to compute a floating point operation rate in MFLOPS.
%    Since FLOPRATE shares measurements with FLOPS, the rate
%    and instruction count reported will be that of the interval
%    since the last call to either of these functions.

%   Copyright 2001 The Innovative Computing Laboratory,
%					     University of Tennessee.
%   $Revision$  $Date$
