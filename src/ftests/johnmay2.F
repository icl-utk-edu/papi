#include "fpapi.h"

      program johnmay2
      implicit none

      integer*8 values(10)
      integer es
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      Integer len_trim
      External len_trim
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

#if defined(NO_FLOPS)
      call PAPIf_query_event(PAPI_TOT_INS, retval)
#else
      call PAPIf_query_event(PAPI_FP_INS, retval)
#endif
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_query_event')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_create_eventset(es, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

#if defined(NO_FLOPS)
      call PAPIf_add_event( es, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_TOT_INS: '//
     *  errorstring(1:len_trim(errorstring))
      end if
#else
      call PAPIf_add_event( es, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_FP_INS: '//
     *  errorstring(1:len_trim(errorstring))
      end if
#endif

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_cleanup_eventset(es, retval)
      if (retval .NE. PAPI_EISRUN) then
        call ftests_perror(__LINE__,'error in PAPIf_cleanup_eventset')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if (retval .NE. PAPI_EISRUN) then
        call ftests_perror(__LINE__,'error in PAPIf_destroy_eventset')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if (retval .NE. PAPI_EINVAL) then
        call ftests_perror(__LINE__,'error in PAPIf_destroy_eventset')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_cleanup_eventset(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_cleanup_eventset')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_destroy_eventset')
        call ftests_fail(__FILE__,retval)
      end if

      if (es .NE. PAPI_NULL) then
        call ftests_perror(__LINE__,'error in PAPIf_destroy_eventset')
        call ftests_fail(__FILE__,retval)
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
