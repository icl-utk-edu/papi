#include "fpapi.h"

      program fifth
      implicit none
      
      integer events(2)
      integer es1, es2
      integer*8 values(10), min, max
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      character*(PAPI_MAX_STR_LEN) name
      integer retval

      Integer len_trim, n
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

	call PAPIf_query_event(PAPI_FP_INS, retval)
	if (retval .NE. PAPI_OK) then
        events(1) = PAPI_TOT_INS 
	else
        events(1) = PAPI_FP_INS 
	end if
      events(2) = PAPI_TOT_CYC

      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( es1, events(1), retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if

      call PAPIf_add_event( es1, events(2), retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if

      call PAPIf_create_eventset(es2, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( es2, events(1), retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if
      
      call PAPIf_add_event( es2, events(2), retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_add_event', retval)
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call PAPIf_reset(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_reset', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if

C     call do_flops(NUM_FLOPS)

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if

      call PAPIf_rem_event( es1, events(2), retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_rem_event', retval)
      end if

      call PAPIf_rem_event( es1, events(1), retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_rem_event', retval)
      end if

      call PAPIf_rem_event( es2, events(2), retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_rem_event', retval)
      end if

      call PAPIf_rem_event( es2, events(1), retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_rem_event', retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Test case 5: Overlapping start and stop of 2 eventsets 
     *with reset."

      print *, "--------------------------------------------------------
     *-------------------"
      end if
      call PAPIf_get_domain(es1, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_get_domain', retval)
      end if

      call stringify_domain(domain, domainstr)
      if (tests_quiet .EQ. 0) then
      write (*,900) "Default domain is:", domain, domainstr
      end if
 900  format(a20, i3, " ", a20)

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
      end if
 800  format(a25, i3, " ", a20)

      if (tests_quiet .EQ. 0) then
         call PAPIf_event_code_to_name (events(1), name, retval)
         if ( retval.NE.PAPI_OK) then
            call ftest_fail(__FILE__, __LINE__, 
     *		'PAPIf_event_code_to_name', retval)
         end if
         n=len_trim(name)
         print *, "Using", NUM_FLOPS, " iterations of c += b*c"

         print *, "-----------------------------------------------------
     *--------------------"
 
         write (*,100) "Test type", 1, 2, 3, 4
         write (*,100) name(1:n), values(1), values(3), 
     *     values(5), values(7)
         write (*,100) "PAPI_TOT_CYC", values(2), values(4), 
     *     values(6), values(8)
 100     format(a15, ":", i12, i12, i12, i12)
         print *, "-----------------------------------------------------
     *--------------------"
 
         print *, "Verification:"
         print *, "Column 1 approximately equals column 2"
         print *, "Column 3 approximately equals two times column 2"
         print *, "Column 4 approximately equals column 2"
      end if

      min = values(3)*0.9
      max = values(3)*1.1
      if ((values(1) .gt. max) .OR. (values(1) .lt. min) .OR. 
     *    (values(5) .gt. (max*2)) .OR. (values(5) .lt. (min*2)) .OR. 
     *    (values(7) .gt. max) .OR. (values(7) .lt. min)) then
         call ftest_fail(__FILE__, __LINE__,
     .	 name, 1)
      end if            

      min = values(4)*0.9
      max = values(4)*1.1
      if ((values(2) .gt. max) .OR. (values(2) .lt. min) .OR. 
     *    (values(6) .gt. (max*2)) .OR. (values(6) .lt. (min*2)) .OR. 
     *    (values(8) .gt. max) .OR. (values(8) .lt. min)) then
         call ftest_fail(__FILE__, __LINE__,
     .	 'PAPI_TOT_CYC', 1)
      end if            

      call ftests_pass(__FILE__)
      end
