% FLOPS Floating point operation count.
%    FLOPS returns the cumulative number of floating point operations.
%
%                    FLOPS(0) - Initialize PAPI library, reset counters
%                               to zero and begin counting.
%              ops = FLOPS    - Return the number of floating point 
%                               operations since the first call or last reset.
%    [ops, mflops] = FLOPS    - Return both the number of floating point 
%                               operations since the first call or last reset,
%                               and the incremental rate of floating point 
%                               execution since the last call.
%
%    DESCRIPTION
%    The first call to flops will initialize PAPI, set up the counters to 
%    monitor floating point instructions and total cpu cycles, and start 
%    the counters. Subsequent calls will return one or two values. The number 
%    of floating point operations since the first call or last reset is always
%    returned. Optionally, the execution rate in mflops can also be returned. 
%    The mflops rate is computed by dividing the operations since the last call
%    by the cycles since the last call and multiplying by cycles per second:
%                   mflops = ((ops/cycles)*(cycles/second))/10^6
%    The PAPI library will continue counting after any call. A call with an 
%    input of 0 will reset the counters and return 0.

%   Copyright 2001 The Innovative Computing Laboratory,
%                        University of Tennessee.
%   $Revision$  $Date$
