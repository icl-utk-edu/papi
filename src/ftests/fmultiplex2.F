#include "fpapi.h"
#define SUCCESS 1
#define NUM 100

#define MAX_TO_ADD 9

#define PAPI_MAX_PRESET_EVENTS 128

      program multiplex2
      IMPLICIT NONE

      integer retval
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()
      if (tests_quiet .EQ. 0) then
         write (*, 100) NUM
 100     FORMAT ("multiplex2: Using ", I3, " iterations")
         write (*,*) "case1: Does PAPI_multiplex_init() handle", 
     *" lots of events?"
      end if
      call case1(tests_quiet, retval)
      call ftests_pass(__FILE__)
      end

      subroutine init_papi(tests_quiet)
      IMPLICIT NONE
      integer retval
      integer tests_quiet

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPI_library_init', retval)
      end if
      if (tests_quiet .EQ. 0) then
         call PAPIf_set_debug(PAPI_VERB_ECONT, retval)
         if ( retval.NE.PAPI_OK) then
            call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_set_debug', retval)
         end if
      end if
      end

      subroutine case1(tests_quiet, ret)
      IMPLICIT NONE
      integer tests_quiet
      integer retval
      integer i, ret, fd
      integer EventCode
      character*(PAPI_MAX_STR_LEN) event_name, event_descr,
     *event_label, event_note
      integer avail_flag, flags, check
      integer EventSet
      integer*8 values(MAX_TO_ADD*2)

      EventSet = PAPI_NULL
      call init_papi(tests_quiet)

      call PAPIf_multiplex_init(retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'papif_multiplex_init', retval)
      end if

      call PAPIf_create_eventset(EventSet, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_create_eventset',
     *retval)
      end if

      call PAPIf_set_multiplex(EventSet, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'papif_set_multiplex', retval)
      end if

      if (tests_quiet .EQ. 0) then
         print *, "Checking for available events..."
      end if

      EventCode = 0
      do i=1, MAX_TO_ADD, 1
         avail_flag=0
         do while ((avail_flag.EQ.0).AND.
     *        (EventCode.LT.PAPI_MAX_PRESET_EVENTS))
            call papif_query_event_verbose(EventCode+z'80000000', 
     *      event_name, event_descr, event_label, avail_flag, 
     *      event_note, flags, check)
            EventCode = EventCode + 1
         end do

         if (tests_quiet .EQ. 0) then
            write (*, 200) "  Adding Event ", event_name 
 200        FORMAT(A13, A11)
         end if

         call PAPIf_add_event( EventSet, EventCode+z'80000000'-1,
     * retval )
         if ( retval .NE. PAPI_OK ) then
            call ftest_fail(__FILE__, __LINE__,
     *           'PAPIf_add_event', retval)
            stop
         end if

         if (tests_quiet .EQ. 0) then
            write (*, 200) "  Added Event ", event_name 
         end if
      end do
      
      call PAPIf_start(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_start', retval)
      end if

      fd = 1
      do i=0, NUM, 1
         call do_flops(1000000)
         call do_reads(1000)
      end do

      call PAPIf_stop(EventSet, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_stop', retval)
      end if

      call PAPIf_cleanup_eventset(EventSet, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_cleanup_eventset',
     *retval)
      end if

      call PAPIf_destroy_eventset(EventSet, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_destroy_eventset',
     *retval)
      end if

      ret = SUCCESS

      end

