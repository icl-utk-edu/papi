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

      INTEGER len_trim
      EXTERNAL len_trim

      a=0.999
      b=1.001
      n=1000

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftests_perror(__LINE__,'Error in PAPI_library_init ')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_create_eventset( EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( EventSet, PAPI_L2_TCM, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
         print *, 'Can not add PAPI_L2_TCM: '//
     $       errorstring(1:len_trim(errorstring))
      else
        print *,'Added PAPI_L2_TCM'
      end if

      call PAPIf_add_event( EventSet, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK )then
         call PAPIf_perror( retval, errorstring, errorcode)
         print *, 'Can not add PAPI_TOT_CYC: '//
     $       errorstring(1:len_trim(errorstring))
      else
         print *,'Added PAPI_TOT_CYC'
      end if
 
      call PAPIf_add_event(EventSet,PAPI_FP_INS,retval)
      if ( retval .NE. PAPI_OK )then
         call PAPIf_perror( retval, errorstring, errorcode)
         print *, 'Can not add PAPI_FP_INS: '//
     $       errorstring(1:len_trim(errorstring))
      else
         print *,'Added PAPI_FP_INS'
      end if

      call PAPIf_start(EventSet, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error starting counter')
        call ftests_fail(__FILE__,retval)
      end if

      do i=1,n
       c = a * b
      end do

      call PAPIf_stop( EventSet, gl, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error stopping counter')
        call ftests_fail(__FILE__,retval)
      end if
      
      call ftests_pass(__FILE__)
      end
