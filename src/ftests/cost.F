#include "fpapi.h"

      program cost
      implicit none
      
      integer es, costes
      integer*8 values(10)
      integer*8 ototcyc, ntotcyc
      integer*4 i
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      Integer len_trim
      External len_trim
      integer tests_quiet

      print *, 'Run in quiet mode (enter 1 for yes, 0 for no)?'
      read(5,*) tests_quiet

      if (tests_quiet .EQ. 0) then
      print *, "Cost of execution for PAPI start/stop",
     *" and PAPI read."
      print *, "This test takes a while. Please be patient..."
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

      call PAPIf_query_event(PAPI_TOT_INS, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_query_event')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_create_eventset(es, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( es, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if

      call PAPIf_add_event( es, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_TOT_INS: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if

      call PAPIf_create_eventset(costes, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( costes, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if

      call PAPIf_add_event( costes, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_TOT_INS: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Performing start/stop test..."
      end if

      call PAPIf_start(costes, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_reset(costes, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_get_real_cyc(ototcyc)
      
      do i=0, 50000
         call PAPIf_start(es, retval)
         call PAPIf_stop(es, values(1), retval)
      end do
      
      call PAPIf_stop(costes, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_get_real_cyc(ntotcyc)
      ntotcyc=ntotcyc-ototcyc

      if (tests_quiet .EQ. 0) then
      print *, "User level cost for PAPI_start/stop(2 counters) over",
     *" 50000 iterations:"
      print *, " ", values(1), " total cyc\n", "  ", values(2), 
     *" total ins\n", values(1)/50000.0, " cyc/call pair\n", 
     *values(2)/50000.0, " ins/call pair"
      print *, "Total cost for PAPI_start/stop(2 counters) over",
     *" 50000 iterations:"
      print *, " ", ntotcyc, " total cyc\n", ntotcyc/50001.0, 
     *" cyc/call pair"
      
C     Start the read val
      
      print *, "Performing read test..."
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_reset(costes, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(costes, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_get_real_cyc(ototcyc)

      do i=0, 50000
         call PAPIf_read(es, values(1), retval)
      end do
      
      call PAPIf_stop(costes, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_get_real_cyc(ntotcyc)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      ntotcyc=ntotcyc-ototcyc

      if (tests_quiet .EQ. 0) then
      print *, "User level cost for PAPI_read(2 counters) over",
     *" 50000 iterations:"
      print *, " ", values(1), " total cyc\n", values(2), "  ",
     *" total ins\n", values(1)/50000.0, " cyc/call\n", 
     *values(2)/50000.0, " ins/call"
      print *, "Total cost for PAPI_read(2 counters) over ",
     *"50000 iterations:"
      print *, " ", ntotcyc, " total cyc\n", ntotcyc/50001.0, 
     *" cyc/call"
      end if

      call ftests_pass(__FILE__)
      end

