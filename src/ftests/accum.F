#include "fpapi.h"

      program accum
      implicit none

      integer es1, number
      integer*8 values(10)
      integer events(2)
      integer retval
      integer tests_quiet

      print *, 'Run in quiet mode (enter 1 for yes, 0 for no)?'
      read(5,*) tests_quiet

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
#if defined(NO_FLOPS)
      events(1) = PAPI_TOT_INS 
#else
      events(1) = PAPI_FP_INS 
#endif

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

      if (tests_quiet .EQ. 0) then
      print *, "Test case accum: Test of PAPI_add_events, ",
     *"PAPI_rem_events, PAPI_accum"
      print *, "------------------------------------------",
     *"------------------------"
      write (*,100) "Test type", 1, 2, 3, 4, 5
#if defined (NO_FLOPS)
      write (*,100) "PAPI_TOT_INS", values(1), values(3),
     *values(5), values(7), values(9)
#else
      write (*,100) "PAPI_FP_INS", values(1), values(3),
     *values(5), values(7), values(9)
#endif
      write (*,100) "PAPI_TOT_CYC", values(2), values(4),
     *values(6), values(8), values(10)
      print *, "------------------------------------------",
     *"------------------------"
 100  format(a15, ":", i12, i12, i12, i12, i12)
      print *, "\nVerification:"
      print *, "Column 2 approximately equals to 0;"
      print *, "Column 3 approximately equals 2 * Column 1;"
      print *, "Column 4 approximately equals 3 * Column 1;"
      print *, "Column 5 approximately equals Column 1."
      end if
 
      call ftests_pass(__FILE__)
      end
