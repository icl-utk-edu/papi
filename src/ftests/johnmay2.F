#include "fpapi.h"

      program johnmay2
      implicit none

      integer*8 values(10)
      integer es
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

#if defined(NO_FLOPS)
      call PAPIf_query_event(PAPI_TOT_INS, retval)
#else
      call PAPIf_query_event(PAPI_FP_INS, retval)
#endif
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_query_event', retval)
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

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_start', retval)
      end if

      call PAPIf_cleanup_eventset(es, retval)
      if (retval .NE. PAPI_EISRUN) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_cleanup_eventset', 
     *retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if (retval .NE. PAPI_EISRUN) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_destroy_eventset', 
     *retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_stop', retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if (retval .NE. PAPI_EINVAL) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_destroy_eventset', 
     *retval)
      end if

      call PAPIf_cleanup_eventset(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_cleanup_eventset', 
     *retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_destroy_eventset', 
     *retval)
      end if

      if (es .NE. PAPI_NULL) then
        call ftest_fail(__FILE__, __LINE__,
     .	 'PAPIf_destroy_eventset', 
     *retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Test case John May 2: cleanup / ",
     *"destroy eventset."
      print *, "--------------------------------",
     *"-----------------"
      print *, "Test run    : 1"
#if defined(NO_FLOPS)
      print *, "PAPI_TOT_INS : ", values(1)
#else
      print *, "PAPI_FP_INS : ", values(1)
#endif
      print *, "----------------------------------",
     *"---------------"
 
      print *, "Verification:"
      print *, "These error messages:"
      print *, "PAPI Error Code -10: PAPI_EISRUN: ", 
     *"EventSet is currently counting"
      print *, "PAPI Error Code -10: PAPI_EISRUN: ",
     *"EventSet is currently counting"
      print *, "PAPI Error Code -1: PAPI_EINVAL: ",
     *"Invalid argument"
      end if
      
      call ftests_pass(__FILE__)
      end
