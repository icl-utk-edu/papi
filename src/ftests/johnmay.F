#include "fpapi.h"

      program johnmay
      implicit none

      integer es1, es2
      integer*8 values(10)
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      Integer len_trim
      External len_trim
      integer tests_quiet
      character*25 chbuf

      tests_quiet=0
      call getarg(1,chbuf)
      if (LGE(chbuf, 'TESTS_QUIET')) then
         tests_quiet=1
      end if

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
      
      call PAPIf_query_event(PAPI_TOT_CYC, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_query_event')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
      end if

#ifndef NO_FLOPS
      call PAPIf_query_event(PAPI_FP_INS, retval)
#else
      call PAPIf_query_event(PAPI_TOT_INS, retval)
#endif
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_query_event')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_create_eventset(es2, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

#ifndef NO_FLOPS
      call PAPIf_add_event( es2, PAPI_FP_INS, retval )
#else
      call PAPIf_add_event( es2, PAPI_TOT_INS, retval )
#endif
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_stop(es1, values(9), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Test case John May: start & stop of overlapped",
     *" unshared counters."
      print *, "----------------------------------------------",
     *"-------------------"
 
      write (*,100) "Test run", 1, 2, 3, 4
#ifndef NO_FLOPS
      write (*,100) "PAPI_FP_INS", values(1), values(3),
     *values(5), values(7)
#else
      write (*,100) "PAPI_TOT_INS", values(1), values(3),
     *values(5), values(7)
#endif
      write (*,100) "PAPI_TOT_CYC", 0, 0, 0, values(9)
 100  format(a15, ":", i12, i12, i12, i12)
      print *, "----------------------------------------------",
     *"---------------------------"
 
      print *, "Verification: None"
      end if

      call ftests_pass(__FILE__)
      end
