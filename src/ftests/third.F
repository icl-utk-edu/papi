#include "fpapi.h"

      program third
      implicit none

      integer*8 values(10), min, max
      character*8 zero
      integer es1, es2
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer retval

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
        call ftest_fail(__FILE__, __LINE__, 'PAPI_library_init', retval)
      end if

      call PAPIf_set_debug(PAPI_VERB_ECONT, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_set_debug', retval)
      end if

      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_create_eventset(es2, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_create_eventset', 
     *retval)
      end if

#if defined(NO_FLOPS)
      call PAPIf_add_event( es2, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#else
      call PAPIf_add_event( es2, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#endif

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_start', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_stop', retval)
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_start', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_start', retval)
      end if

      call PAPIf_read(es1, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_read', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_stop', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_stop', retval)
      end if

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(9), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_stop', retval)
      end if

      call PAPIf_rem_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_rem_event', retval)
      end if

#if defined(NO_FLOPS)
      call PAPIf_rem_event( es2, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_rem_event', retval)
      end if
#else
      call PAPIf_rem_event( es2, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_rem_event', retval)
      end if
#endif

      zero="  0     "
      if (tests_quiet .EQ. 0) then
      print *, "Test case 3: Overlapping start and stop of 2 eventsets",
     *" with different counters."
      print *, "----------------------------------------------------------
     *---------------------"
      end if

      call PAPIf_get_domain(es1, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_get_domain', retval)
      end if

      call stringify_domain(domain, domainstr)
      if (tests_quiet .EQ. 0) then
      write (*,900) "Default domain is:", domain, domainstr
      end if
 900  format(a20, i3, " ", a20)

      call PAPIf_get_granularity(es1, granularity, PAPI_GET_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_get_granularity',
     *retval)
      end if

      call stringify_granularity(granularity, grnstr)
      if (tests_quiet .EQ. 0) then
      write (*,800) "Default granularity is:", granularity, grnstr
      end if
 800  format(a25, i3, " ", a20)

      if (tests_quiet. EQ. 0) then
      print *, "Using", NUM_FLOPS, " iterations of c += b*c"

      print *, "--------------------------------------------------------------
     *---------------------"

      write (*,100) "Test type", 1, 2, 3, 4, 5
#if defined(NO_FLOPS)
      write (*,100) "PAPI_TOT_INS", 0, 0, values(5), 0, values(9)
#else
      write (*,100) "PAPI_FP_INS", 0, 0, values(5), 0, values(9)
#endif
      write (*,100) "PAPI_TOT_CYC", values(1), values(3), 0,
     *values(7), 0
      print *, "----------------------------------------------------------
     *---------------------"
 100  format(a15, ":", i12, i12, i12, i12, i12)
      print *, "Verification:"
      print *, "Row 1 approximately equals 0  0  N  0  N"
      print *, "Row 2 approximately equals X  X  0  X  0"
      print *, "Column 1 approximately equals column 2"
      print *, "Column 4 approximately equals three times column 1"
      print *, "Column 5 approximately equals column 3"
      end if

      min = values(5)*0.9
      max = values(5)*1.1
      if ((values(5) .EQ. 0) .OR. (values(9) .EQ. 0) .OR. 
     *    (values(9) .gt. max) .OR. (values(9) .lt. min)) then
#if defined(NO_FLOPS)
         call ftest_fail(__FILE__, __LINE__, 'PAPI_TOT_INS', 1)
#else
         call ftest_fail(__FILE__, __LINE__, 'PAPI_FP_INS', 1)
#endif
      end if            

      min = values(1)*0.9
      max = values(1)*1.1
      if ((values(1).EQ.0).OR.(values(3).EQ.0).OR.(values(7).EQ.0) .OR.
     *    (values(3) .gt. max) .OR. (values(3) .lt. min) .OR.
     *    (values(7) .gt. (max*3)) .OR. (values(7) .lt. (min*3))) then
         call ftest_fail(__FILE__, __LINE__, 'PAPI_TOT_CYC', 1)
      end if            

      call ftests_pass(__FILE__)
      end
