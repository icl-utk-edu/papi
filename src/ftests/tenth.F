#include "fpapi.h"
#define ITERS 100

      program tenth
      implicit none

      integer*8 values(10)
      integer es1, es2, es3
      integer mask1, mask2, mask3
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer retval

      Integer len_trim
      External len_trim
      integer tests_quiet, get_quiet
	external get_quiet

	tests_quiet = get_quiet()

#if (defined(sun) && defined(sparc))
      mask1 = PAPI_L2_TCM
      mask2 = PAPI_L2_TCA
      mask3 = PAPI_L2_TCH
#else
      mask1 = PAPI_L1_TCM
      mask2 = PAPI_L1_ICM
      mask3 = PAPI_L1_DCM
#endif

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPI_library_init', retval)
      end if

#if (defined(__alpha) && defined(__osf__))
      call ftests_pass(__FILE__)
#endif

      call PAPIf_set_debug(PAPI_VERB_ECONT, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_set_debug', retval)
      end if
      
#ifndef _CRAYT3E
      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_create_eventset',
     *retval)
      end if

      call PAPIf_add_event( es1, mask1, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if

#endif      
      call PAPIf_create_eventset(es2, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_create_eventset',
     *retval)
      end if

      call PAPIf_add_event( es2, mask2, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if

      call PAPIf_create_eventset(es3, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_create_eventset',
     *retval)
      end if

      call PAPIf_add_event( es3, mask3, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if

      call do_l1misses(ITERS)

#ifndef _CRAYT3E
      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call do_l1misses(ITERS)

      call PAPIf_stop(es1, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if

#else
      values(1)=0
#endif

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call do_l1misses(ITERS)
      call PAPIf_stop(es2, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if
      
      call PAPIf_start(es3, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call do_l1misses(ITERS)

      call PAPIf_stop(es3, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if

#ifndef _CRAYT3E
      call PAPIf_rem_event( es1, mask1, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_rem_event', retval)
      end if

#endif
      call PAPIf_rem_event( es2, mask2, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_rem_event', retval)
      end if

      call PAPIf_rem_event( es3, mask3, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_rem_event', retval)
      end if

      if (tests_quiet .EQ. 0) then
#if (defined(sun) && defined(sparc))
      print *, "Test case 10: start, stop for derived event ",
     *"PAPI_L2_TCM."
#else
      print *, "Test case 10: start, stop for derived event ",
     *"PAPI_L1_TCM."
#endif
 
      print *, "------------------------------------------------------"
      end if
      call PAPIf_get_domain(es1, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_get_domain', retval)
      end if

      call stringify_domain(domain, domainstr)
      if (tests_quiet .EQ. 0) then
      write (*,900) "Default domain is:", domain, domainstr
 900  format(a20, i3, " ", a20)
      end if

      call PAPIf_get_granularity(es1, granularity, PAPI_GET_DEFGRN, 
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
      print *, "Using", NUM_FLOPS, " iterations of c += b*c"
      print *, "------------------------------------------------------"
      
      write (*,500) "Test type", 1, 2, 3
#if (defined(sun) && defined(sparc))
      write (*,500) "PAPI_L2_TCM", values(1), 0, 0
      write (*,500) "PAPI_L2_TCA", 0, values(3), 0
      write (*,500) "PAPI_L2_TCH", 0, 0, values(5)
      print *, "------------------------------------------------",
     *"------"
 
      print *, "Verification:"
      print *, "First number row 1 approximately equals (2,2) - (3,3) ", 
     *"or ",(values(3)-values(5))
#else
      write (*,500) "PAPI_L1_TCM", values(1), 0, 0
      write (*,500) "PAPI_L1_ICM", 0, values(3), 0
      write (*,500) "PAPI_L1_DCM", 0, 0, values(5)
      print *, "------------------------------------------------",
     *"------"
 
      print *, "Verification:"
      print *, "First number row 1 approximately equals (2,2) + (3,3) ",
     *"or ", (values(3)+values(5))
#endif
      end if
 500  format(A13, ": ", I10, I10, I10)

      call ftests_pass(__FILE__)
      end
