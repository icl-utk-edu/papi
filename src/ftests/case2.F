C From Dave McNamara at PSRV. Thanks! 
C Ported to fortran by Kevin London
C If an event is countable but you've exhausted the counter resources
C and you try to add an event, it seems subsequent PAPI_start and/or
C PAPI_stop will causes a Seg. Violation.

C  I got around this by calling PAPI to get the # of countable events,
C then making sure that I didn't try to add more than these number of
C events. I still have a problem if someone adds Level 2 cache misses
C and then adds FLOPS 'cause I didn't count FLOPS as actually requiring
C 2 counters. 

#include "fpapi.h"

      program case2

      REAL c,a,b
      INTEGER n, handle
      INTEGER EventSet
      INTEGER retval
      INTEGER I
      INTEGER*8 gl(3)
      a=0.999
      b=1.001
      n=1000
      handle=0

      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
         print *, 'Error in PAPI_library_init '
         stop
      end if
      call PAPIf_thread_init( handle, handle, retval )
      if ( retval.NE.PAPI_OK ) then
        print *, 'Error in PAPI_thread_init'
        stop
      end if
      call PAPIf_add_event( EventSet, PAPI_L2_TCM,retval )
      if (retval.NE.PAPI_OK) print *, 'error adding L2 TCM'
      call PAPIf_add_event(EventSet, PAPI_TOT_CYC,retval)
      if (retval.NE.PAPI_OK) print *, 'error adding TOT_CYC'
      call PAPIf_add_event(EventSet,PAPI_FP_INS,retval)
      if (retval.NE.PAPI_OK) print *, 'error adding FP_INS'

      call PAPIf_start(EventSet, retval )

      do i=1,n
       c = a * b
      end do

      call PAPIf_stop( EventSet, gl, retval)
      end
