#include "fpapi.h"

      program description
      implicit none

      integer es1, number
      integer*8 values(10)
      integer events(2), eventlist(2)
      integer eventtotal
      integer i
      character*PAPI_MAX_STR_LEN name
      integer status
      integer retval

      Integer len_trim
      External len_trim
      integer tests_quiet

      print *, 'Run in quiet mode (enter 1 for yes, 0 for no)?'
      read(5,*) tests_quiet

      if (tests_quiet .EQ. 0) then
      print *, "Test case descriptions: Test of functions:\n",
     *"\t\tPAPI_add_events, PAPI_rem_events,\n",
     *"\t\tPAPI_list_events, PAPI_describe_event,\n",
     *"\t\tPAPI_state."
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

      eventtotal=5
      call PAPIf_list_events(es1, eventlist, eventtotal, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in listing EventSets')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "\nEvent List:"
      print *, "---------------------------------------",
     *"---------------------------"
      print *, "Event Name\tCode"
      end if
      do i = 1, eventtotal
         call PAPIf_event_code_to_name (eventlist(i), name, retval)
         if ( retval.NE.PAPI_OK) then
            call ftests_perror(__LINE__,'Error in listing EventSets')
            call ftests_fail(__FILE__,retval)
         end if
         if (tests_quiet .EQ. 0) then
         write (*, 100) name, eventlist(i)
         end if
 100     format(A12,O12)
      end do
      if (tests_quiet .EQ. 0) then
      print *, "---------------------------------------",
     *"---------------------------"
      end if
      
      call PAPIf_state(es1, status, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_state')
        call ftests_fail(__FILE__,retval)
      end if

      if (status .NE. PAPI_STOPPED) then
         print *, "PAPI_state Error"
         stop
      end if
      if (tests_quiet .EQ. 0) then
      print *, "PAPI_state: PAPI_STOPPED"
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if
      if (tests_quiet .EQ. 0) then
      print *, "PAPI_start"
      end if

      call PAPIf_state(es1, status, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_state')
        call ftests_fail(__FILE__,retval)
      end if

      if (status .NE. PAPI_RUNNING) then
         print *, "PAPI_state Error"
         stop
      end if
      if (tests_quiet .EQ. 0) then
      print *, "PAPI_state: PAPI_RUNNING"
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, values, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "PAPI_stop"
      end if

      call PAPIf_state(es1, status, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_state')
        call ftests_fail(__FILE__,retval)
      end if

      if (status .NE. PAPI_STOPPED) then
         print *, "PAPI_state Error"
         stop
      end if
      if (tests_quiet .EQ. 0) then
      print *, "PAPI_state: PAPI_STOPPED"
      end if

      call PAPIf_rem_events( es1, events, number, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_events')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "\nResults:"
      print *, "---------------------------------------",
     *"---------------------------"
      print *, "Test type   : \t1"
      print *, "PAPI_FP_INS : \t", values(1)
      print *, "PAPI_TOT_CYC: \t", values(2)
      print *, "---------------------------------------",
     *"---------------------------"

      print *, "\nVerification:"
      print *, "1. The events listed by PAPI_describe_event",
     *"should be exactly the same events added by PAPI_add_events."
      print *, "2. The PAPI_state should be PAPI_RUNNING after ",
     *"PAPI_start and before PAPI_stop."
      print *, "It should be PAPI_STOPPED at other time."
      end if

      call ftests_pass(__FILE__)
      end
