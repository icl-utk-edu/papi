#include "fpapi.h"
#define SUCCESS 1
#define NUM 10

      program multiplex1
      IMPLICIT NONE

      integer retval
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()
C Skip Alpha till multiplex is fixed. -KSL 
#if defined(__alpha) && defined(__osf__)
      call ftest_skip(__FILE__, __LINE__, 'Broken', PAPI_ENOEVNT)
#endif

      if (tests_quiet .EQ. 0) then
         write (*, 100) NUM
 100     FORMAT ("multiplex1: Using ", I3, " iterations")
         write (*,*) "case1: Does PAPI_multiplex_init() not break",
     *" regular operation?"
      end if   
      call case1(retval, tests_quiet)
      if (tests_quiet .EQ. 0) then
         write (*,*) "case2: Does setmpx/add work?"
      end if
      call case2(retval, tests_quiet)
      if (tests_quiet .EQ. 0) then
         write (*,*) "case3: Does add/setmpx work?"
      end if
      call case3(retval, tests_quiet)
      if (tests_quiet .EQ. 0) then
         write (*,*) "case4: Does add/setmpx/add work?"
      end if
      call case4(retval, tests_quiet)

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPI_library_init', retval)
      end if
      call ftests_pass(__FILE__)

      end

      subroutine init_papi(tests_quiet, event)
      IMPLICIT NONE
      integer retval
      integer tests_quiet, event

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
        call PAPIf_query_event(PAPI_FP_INS, retval)
        if (retval .NE. PAPI_OK) then
        event = PAPI_TOT_CYC 
        else
        event = PAPI_FP_INS 
        end if
      end

C     Tests that PAPI_multiplex_init does not mess with normal operation. 
      subroutine case1(ret, tests_quiet)
      IMPLICIT NONE
      integer ret, tests_quiet, event
      integer retval, i, EventSet
      INTEGER*8 values(4)
      integer fd

      EventSet = PAPI_NULL
      call init_papi(tests_quiet, event)
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

      call PAPIf_add_event( EventSet, event, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if

      call PAPIf_add_event( EventSet, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_add_event( EventSet, PAPI_TOT_IIS, retval )
        if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
        end if
      end if

      if(tests_quiet .EQ. 0) then
        write(*,*) 'Event set list'
        call PrintEventSet(EventSet)
      end if

      call PAPIf_start(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_start', retval)
      end if

      fd = 1
      do i=0, NUM, 1
         call do_flops(1000000)
#ifndef _CRAYT3E
         call do_reads(1000)
#endif
      end do

      call PAPIf_stop(EventSet, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_stop', retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "case1: ", values(1), values(2)
      end if

      call PAPIf_cleanup_eventset(EventSet, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_cleanup_eventset',
     *retval)
      end if

      call PAPIF_shutdown()
      ret = SUCCESS
      end

C     Tests that PAPI_set_multiplex() works before adding events
      subroutine case2(ret, tests_quiet)
      IMPLICIT NONE
      integer ret, tests_quiet, event
      integer retval, i, EventSet
      INTEGER*8 values(4)
      integer fd

      EventSet = PAPI_NULL
      call init_papi(tests_quiet, event)
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

      call PAPIf_add_event( EventSet, event, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if

      call PAPIf_add_event( EventSet, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_add_event( EventSet, PAPI_TOT_IIS, retval)
        if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
        end if
      end if

C  This print-out is disabled until PAPIf_list_event is working
C  for multiplexed event sets (change -4711 to 0 when it is working)
      if(tests_quiet .EQ. -4711) then
        write(*,*) 'Event set list'
        call PrintEventSet(EventSet)
      endif

      call PAPIf_start(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_start', retval)
      end if

      fd = 1
      do i=0, NUM, 1
         call do_flops(1000000)
#ifndef _CRAYT3E
         call do_reads(1000)
#endif
      end do



      call PAPIf_stop(EventSet, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_stop', retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "case2: ", values(1), values(2)
      end if

      call PAPIf_cleanup_eventset(EventSet, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_cleanup_eventset',
     *retval)
      end if

      call PAPIF_shutdown()
      ret = SUCCESS
      end

C     Tests that PAPI_set_multiplex() works after adding events
      subroutine case3(ret, tests_quiet)
      IMPLICIT NONE
      integer ret, tests_quiet, event
      integer retval, i, EventSet
      INTEGER*8 values(4)
      integer fd

      EventSet = PAPI_NULL
      call init_papi(tests_quiet, event)
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

      call PAPIf_add_event( EventSet, event, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if

#ifdef _CRAYT3E
      call PAPIf_add_event( EventSet, PAPI_TLB_DM, retval )
#else
      call PAPIf_add_event( EventSet, PAPI_TOT_INS, retval )
#endif
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if

      if(tests_quiet .EQ. 0) then
        write(*,*) 'Event set before call to PAPIf_set_multiplex:'
        call PrintEventSet(EventSet)
      endif

      call PAPIf_set_multiplex(EventSet, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'papif_set_multiplex', retval)
      end if

      if(tests_quiet .EQ. 0) then
        write(*,*) 'Event set after call to PAPIf_set_multiplex:'
        call PrintEventSet(EventSet)
      endif

      call PAPIf_start(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_start', retval)
      end if

      fd = 1
      do i=0, NUM, 1
         call do_flops(1000000)
#ifndef _CRAYT3E
         call do_reads(1000)
#endif
      end do

      call PAPIf_stop(EventSet, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_stop', retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "case3: ", values(1), values(2)
      end if

      call PAPIf_cleanup_eventset(EventSet, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_cleanup_eventset',
     *retval)
      end if

      call PAPIF_shutdown()
      ret = SUCCESS
      end

C     Tests that PAPI_set_multiplex() works before adding events
C     Tests that PAPI_add_event() works after
C     PAPI_add_event()/PAPI_set_multiplex()
      subroutine case4(ret, tests_quiet)
      IMPLICIT NONE
      integer ret, tests_quiet, event
      integer retval, i, EventSet
      INTEGER*8 values(4)
      integer fd

      EventSet = PAPI_NULL
      call init_papi(tests_quiet, event)
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

      call PAPIf_add_event( EventSet, event, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if

#ifdef _CRAYT3E
      call PAPIf_add_event( EventSet, PAPI_TLB_DM, retval )
#else
      call PAPIf_add_event( EventSet, PAPI_TOT_INS, retval )
#endif
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_add_event( EventSet, PAPI_TOT_IIS, retval )
        if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
        end if
      end if

      if(tests_quiet .EQ. 0) then
        write(*,*) 'Event set before call to PAPIf_set_multiplex:'
        call PrintEventSet(EventSet)
      endif

      call PAPIf_set_multiplex(EventSet, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'papif_set_multiplex', retval)
      end if

      if(tests_quiet .EQ. 0) then
        write(*,*) 'Event set after call to PAPIf_set_multiplex:'
        call PrintEventSet(EventSet)
      endif

#if (defined(i386)&&defined(linux))||defined(mips) || defined(_CRAYT3E) || (defined(__ia64__) && defined(linux)) || (SUBSTR==aix-power)
      call PAPIf_add_event( EventSet, PAPI_L1_DCM, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_add_event( EventSet, PAPI_L2_DCM, retval )
        if ( retval .NE. PAPI_OK ) then
           call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
        end if
      end if
      call PAPIf_add_event( EventSet, PAPI_L1_ICM, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if

#elif (defined(sparc) && defined(sun))
      call PAPIf_add_event( EventSet, PAPI_LD_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if
      call PAPIf_add_event( EventSet, PAPI_SR_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if

#elif (defined(__alpha)&&defined(__osf__))
      call PAPIf_add_event( EventSet, PAPI_RES_STL, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_add_event',
     *retval)
      end if

#else
      print *,'*** Did not match in event selection ***'
#endif

      if(tests_quiet .EQ. 0) then
        write(*,*) 'Updated event set list:'
        call PrintEventSet(EventSet)
      endif

      call PAPIf_start(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_start', retval)
      end if

      fd = 1
      do i=0, NUM, 1
         call do_flops(1000000)
#ifndef _CRAYT3E
         call do_reads(1000)
#endif
      end do

      call PAPIf_stop(EventSet, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_stop', retval)
      end if

      if (tests_quiet .EQ. 0) then
#if (defined(__alpha) && defined(__osf__))
      write (*, *) "case4: ", values(1), values(2), values(3)
#else
      write (*, *) "case4: ", values(1), values(2), values(3), 
     *values(4)
#endif
      end if

      call PAPIf_cleanup_eventset(EventSet, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPIf_cleanup_eventset',
     *retval)
      end if

      call PAPIF_shutdown()
      ret = SUCCESS
      end

