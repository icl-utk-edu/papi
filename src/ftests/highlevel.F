#include "fpapi.h"
#include "tests.h"

      program highlevel
      implicit none

      integer*8 values(10)
      integer events(2)
      integer eventnum, length
      integer availcounters
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      call PAPIf_num_counters(availcounters)
      if (eventnum .GT. availcounters) then
         print *, "Not enough hardware counters!"
         stop
      end if

      eventnum = 2
      events(1)=PAPI_FP_INS
      events(2)=PAPI_TOT_CYC

      call PAPIf_start_counters(events, eventnum, retval)
      if ( retval .NE. PAPI_OK ) then
        print *, 'error in PAPIf_start_counters'
        call PAPIf_perror( retval, errorstring, length, errorcode)
        print *, errorstring
        stop
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_read_counters(values(1), eventnum, retval)
      if ( retval .NE. PAPI_OK ) then
        print *, 'error in PAPIf_read_counters'
        call PAPIf_perror( retval, errorstring, length, errorcode)
        print *, errorstring
        stop
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop_counters(values(3), eventnum, retval)
      if ( retval .NE. PAPI_OK ) then
        print *, 'error in PAPIf_stop_counters'
        call PAPIf_perror( retval, errorstring, length, errorcode)
        print *, errorstring
        stop
      end if
      
      print *, "Test case highlevel: Test of high-level APIs."
      print *, "---------------------------------------------",
     *"---------------------"
      print *, "Test type   : \t1\t\t2"
      print *, "PAPI_FP_INS : \t", values(1), "\t", values(3)
      print *, "PAPI_TOT_CYC: \t", values(2), "\t", values(4)
      print *, "---------------------------------------------",
     *"---------------------"

      End
