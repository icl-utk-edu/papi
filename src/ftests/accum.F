#include "fpapi.h"
#include "tests.h"

      program accum
      implicit none

      integer es1, number
      integer*8 values(10)
      integer events(2)
      integer retval

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftests_perror(__LINE__,'Error in PAPI_library_init ')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_set_debug(PAPI_VERB_ECONT, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_set_debug')
        call ftests_fail(__FILE__,retval)
      end if
      
      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      number=2
      events(1) = PAPI_FP_INS 
      events(2) = PAPI_TOT_CYC
      call PAPIf_add_events( es1, events, number, retval )
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in adding EventSets')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_accum(es1, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_accum')
        call ftests_fail(__FILE__,retval)
      end if

      values(1)=values(7)
      values(2)=values(8)
      call PAPIf_stop(es1, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)
      call PAPIf_accum(es1, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_accum')
        call ftests_fail(__FILE__,retval)
      end if

      values(5)=values(7)
      values(6)=values(8)

      call do_flops(NUM_FLOPS)

      call PAPIf_accum(es1, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_accum')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)      

      call PAPIf_stop(es1, values(9), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_rem_events( es1, events, number, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_events')
        call ftests_fail(__FILE__,retval)
      end if

      print *, "Test case accum: Test of PAPI_add_events, ",
     *"PAPI_rem_events, PAPI_accum"
      print *, "------------------------------------------",
     *"------------------------"
      print *, "Test type   : \t1\t\t2\t\t3\t\t4\t\t5"
      print *, "PAPI_FP_INS : \t", values(1), "\t", values(3),
     *"\t", values(5), "\t", values(7), "\t", values(9)
      print *, "PAPI_TOT_CYC: \t", values(2), "\t", values(4),
     *"\t", values(6), "\t", values(8), "\t", values(10)
      print *, "------------------------------------------",
     *"------------------------"
      print *, "\nVerification:"
      print *, "Column 2 approximately equals to 0;"
      print *, "Column 3 approximately equals 2 * Column 1;"
      print *, "Column 4 approximately equals 3 * Column 1;"
      print *, "Column 5 approximately equals Column 1."
 
      call ftests_pass(__FILE__)
      end
