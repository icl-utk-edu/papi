#include "fpapi.h"

      program clockres
      IMPLICIT NONE

      INTEGER*8 elapsed_usec(10000), elapsed_cyc(10000)
      INTEGER*8 total_usec, total_cyc
      INTEGER i,handle
      INTEGER retval
      REAL ITERS
      integer tests_quiet

      total_usec=0
      total_cyc=0
      ITERS=10000.0
      handle=0

      print *, 'Run in quiet mode (enter 0 for yes, 1 for no)?'
      read(5,*) tests_quiet

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftests_perror(__LINE__,'Error in PAPI_library_init ')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .NE. 0) then
      print *, 'Test case: Clock resolution.'
      print *,'-----------------------------------------------'
      end if

      do i=1,10000
         call PAPIf_get_real_cyc( elapsed_cyc(i) )
      end do

      do i=2,10000
         if ((elapsed_cyc(i)-elapsed_cyc(i-1)).LT.0 ) stop 
         total_cyc =total_cyc+(elapsed_cyc(i) - elapsed_cyc(i-1))
      end do

      do i=1,10000
         call PAPIf_get_real_usec(elapsed_usec(i))
      end do

      do i=2,10000
        if ((elapsed_usec(i) - elapsed_usec(i-1)).LT.0) stop
        total_usec=total_usec+(elapsed_usec(i) - elapsed_usec(i-1))
      end do
 
      if (tests_quiet .NE. 0) then
      print *,'PAPIf_get_real_cyc : ',(total_cyc/(ITERS-1))
      print *,'PAPIf_get_real_usec: ',(total_usec/(ITERS-1))
      end if

      call ftests_pass(__FILE__)
      end
