#include "fpapi.h"

      program highlevel
      implicit none

      integer*8 values(10)
      integer events(2)
      integer eventnum, length
      integer availcounters
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring
      integer tests_quiet

      print *, 'Run in quiet mode (enter 1 for yes, 0 for no)?'
      read(5,*) tests_quiet

      eventnum = 2
      call PAPIf_num_counters(availcounters)
      if (eventnum .GT. availcounters) then
         print *, "Not enough hardware counters!"
         stop
      end if

#if defined(NO_FLOPS)
      events(1)=PAPI_TOT_INS
#else
      events(1)=PAPI_FP_INS
#endif
      events(2)=PAPI_TOT_CYC

      call PAPIf_start_counters(events, eventnum, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__, 'error in PAPIf_start_counters')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_read_counters(values(1), eventnum, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__, 'error in PAPIf_read_counters')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop_counters(values(3), eventnum, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__, 'error in PAPIf_stop_counters')
        call ftests_fail(__FILE__,retval)
      end if
      
      if (tests_quiet .EQ. 0) then
      print *, "Test case highlevel: Test of high-level APIs."
      print *, "---------------------------------------------",
     *"---------------------"
      write (*,100) "Test type", 1, 2
#if defined(NO_FLOPS)
      write (*,100) "PAPI_TOT_INS", values(1), values(3)
#else
      write (*,100) "PAPI_FP_INS", values(1), values(3)
#endif
      write (*,100) "PAPI_TOT_CYC", values(2), values(4)
 100  format(a15, ":", i12, i12)
      print *, "---------------------------------------------",
     *"---------------------"
      end if

      call ftests_pass(__FILE__)
      End
