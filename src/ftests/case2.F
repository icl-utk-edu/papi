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
      IMPLICIT NONE

      REAL c,a,b
      INTEGER n
      INTEGER EventSet
      INTEGER retval
      INTEGER I,errorcode
      INTEGER*8 gl(3)
      CHARACTER*(PAPI_MAX_STR_LEN) errorstring

      a=0.999
      b=1.001
      n=1000

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
         print *, 'Error in PAPI_library_init '
         stop
      end if

      call PAPIf_create_eventset( EventSet, retval)
      if (retval.NE.PAPI_OK) print *, 'error creating EventSet'

      call PAPIf_add_event( EventSet, PAPI_L2_TCM,retval )
      if ( retval .NE. PAPI_OK ) then
        print *, 'error adding PAPI_L2_TCM'
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring
      else
        print *,'Added PAPI_L2_TCM'
      end if
 
      call PAPIf_add_event(EventSet, PAPI_TOT_CYC,retval)
      if ( retval .NE. PAPI_OK ) then
        print *, 'error adding PAPI_TOT_CYC'
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring
      else
        print *,'Added PAPI_TOT_CYC'
      end if
  
      call PAPIf_add_event(EventSet,PAPI_FP_INS,retval)
      if ( retval .NE. PAPI_OK ) then
        print *, 'error adding PAPI_FP_INS'
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring
      else
        print *,'Added PAPI_FP_INS'
      end if

      call PAPIf_start(EventSet, retval )

      do i=1,n
       c = a * b
      end do

      call PAPIf_stop( EventSet, gl, retval)
      end
