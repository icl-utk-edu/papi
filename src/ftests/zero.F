#include "fpapi.h"

      program zero
      implicit none

      integer*8 values(10)
      integer es
      integer*8 uso, usn, cyco, cycn
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer retval

      Integer len_trim
      External len_trim
      integer tests_quiet, get_quiet
	external get_quiet

	tests_quiet = get_quiet()

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPI_library_init', retval)
      end if

      call PAPIf_set_debug(PAPI_VERB_ECONT, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_set_debug', retval)
      end if
      
      call PAPIf_create_eventset(es, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_create_eventset', 
     *retval)
      end if

#if defined(NO_FLOPS)
      call PAPIf_add_event( es, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if
#else
      call PAPIf_add_event( es, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if
#endif
 
      call PAPIf_add_event( es, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if

      call PAPIf_get_real_usec(uso)
      call PAPIf_get_real_cyc(cyco)

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if

      call PAPIf_get_real_usec(usn)
      call PAPIf_get_real_cyc(cycn)

      if (tests_quiet .EQ. 0) then
      print *, "Test case 0: start, stop."
      print *, "-----------------------------------------------",
     *"--------------------------"
      end if

      call PAPIf_get_domain(es, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_get_domain', retval)
      end if

      call stringify_domain(domain, domainstr)
      if (tests_quiet .EQ. 0) then
      write (*,900) "Default domain is:", domain, domainstr
      end if
 900  format(a20, i3, " ", a20)

      call PAPIf_get_granularity(es, granularity, PAPI_GET_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_get_granularity',
     *retval)
      end if

      call stringify_granularity(granularity, grnstr)
      if (tests_quiet .EQ. 0) then
      write (*,800) "Default granularity is:", granularity, grnstr
 800  format(a25, i3, " ", a20)

      print *, "-----------------------------------------------",
     *"--------------------------"
 
      write (*,100) "Test type", 1
#ifdef NO_FLOPS
      write (*,100) "PAPI_FP_INS", values(1)
#else
      write (*,100) "PAPI_TOT_INS", values(1)
#endif
      write (*,100) "PAPI_TOT_CYC", values(2)
      write (*,100) "Real usec", usn-uso
      write (*,100) "Real cycles", cycn-cyco
 100  format(a13, ":", i12)
      print *, "-----------------------------------------------",
     *"--------------------------"
 
      print *, "Verification: none"
      endif
     
      call ftests_pass(__FILE__)
      end

