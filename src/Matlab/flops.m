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
%    The PAPI flops function uses the PAPI Performance API to do the heavy
%    lifting. PAPI takes advantage of the fact that most modern microprocessors
%    have built-in hardware support for counting a variety of basic operations
%    or events. PAPI uses these counters to track things like instructions 
%    executed, cycles elapsed, floating point instructions performed and 
%    a variety of other events.
%    The first call to flops will initialize PAPI, set up the counters to 
%    monitor floating point instructions and total cpu cycles, and start 
%    the counters. Subsequent calls will return one or two values. The first 
%    value is the number of floating point operations since the first call or
%    last reset. The second optional value, the execution rate in mflops, can 
%    also be returned. The mflops rate is computed by dividing the operations 
%    since the last call by the cycles since the last call and multiplying by 
%    cycles per second:
%                   mflops = ((ops/cycles)*(cycles/second))/10^6
%    The cycles per second value is a derived number determined empirically
%    by counting cycles for a fixed amount of system time during the 
%    initialization of the PAPI library. Because of the way it is determined,
%    this value can be a small but consistent source of systematic error, 
%    and can introduce differences between rates measured by PAPI and those 
%    determined by other time measurements, for example, tic and toc. Also 
%    note that PAPI on Windows counts events on a system level rather than 
%    a process or thread level. This can lead to an over-reporting of cycles,
%    and typically an under-reporting of mflops.
%    The flops function continues counting after any call. A call with an 
%    input of 0 resets the counters and returns 0.

%   Copyright 2001 - 2004 The Innovative Computing Laboratory,
%                               University of Tennessee.
%   $Revision$  $Date$



