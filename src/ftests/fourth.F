#include "fpapi.h"
 
      program fourth
      implicit none

      integer es1, es2
      integer*8 values(10), min, max
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

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

#if defined(NO_FLOPS)
      call PAPIf_add_event( es1, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Cannot add PAPI_TOT_INS: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if
#else
      call PAPIf_add_event( es1, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Cannot add PAPI_FP_INS: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if
#endif

      call PAPIf_add_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if
      
      call PAPIf_create_eventset(es2, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

#if defined(NO_FLOPS)
      call PAPIf_add_event( es2, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_TOT_INS: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if
#else
      call PAPIf_add_event( es2, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_FP_INS: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if
#endif
 
      call PAPIf_add_event( es2, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)
      call PAPIf_stop(es2, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_rem_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

#if defined(NO_FLOPS)
      call PAPIf_rem_event( es1, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if
#else
      call PAPIf_rem_event( es1, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if
#endif

      call PAPIf_rem_event( es2, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

#if defined(NO_FLOPS)
      call PAPIf_rem_event( es2, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if
#else
      call PAPIf_rem_event( es2, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if
#endif
      
      if (tests_quiet .EQ. 0) then
      print *, "Test case 4: Overlapping start and stop of 2 eventsets."
      print *, "-----------------------------------------------------",
     *"--------------------"
      end if

      call PAPIf_get_domain(es1, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_domain')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_domain(domain, domainstr)
      if (tests_quiet .EQ. 0) then
      write (*,900) "Default domain is:", domain, domainstr
      end if
 900  format(a20, i3, " ", a20)

      call PAPIf_get_granularity(es1, granularity, PAPI_GET_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_granularity')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_granularity(granularity, grnstr)
      if (tests_quiet .EQ. 0) then
      write (*,800) "Default granularity is:", granularity, grnstr
      end if
 800  format(a25, i3, " ", a20)

      if (tests_quiet .EQ. 0) then
      print *, "Using", NUM_FLOPS, " iterations of c += b*c"

      print *, "-----------------------------------------------------",
     *"--------------------"

      write (*,100) "Test type", 1, 2, 3, 4
#if defined(NO_FLOPS)
      write (*,100) "PAPI_TOT_INS", values(1), values(3), 
     * values(5), values(7)
#else
      write(*,100) "PAPI_FP_INS", values(1), values(3), 
     * values(5), values(7)
#endif
      write(*,100) "PAPI_TOT_CYC", values(2), values(4),
     * values(6), values(8)
      print *, "-----------------------------------------------------",
     *"--------------------"
 100  format (a15, ":", i12, i12, i12, i12)
 
      print *, "Verification:"
      print *, "Column 1 approximately equals column 2"
      print *, "Column 3 approximately equals three times column 2"
      print *, "Column 4 approximately equals column 2"
      end if

      min = values(3)*0.9
      max = values(3)*1.1
      if ((values(1) .gt. max) .OR. (values(1) .lt. min) .OR. 
     *    (values(5) .gt. (max*3)) .OR. (values(5) .lt. (min*3)) .OR. 
     *    (values(7) .gt. max) .OR. (values(7) .lt. min)) then
         call ftests_perror(__LINE__,'Verification Failed')
         call ftests_fail(__FILE__,1)
      end if            

      min = values(4)*0.9
      max = values(4)*1.1
      if ((values(2) .gt. max) .OR. (values(2) .lt. min) .OR. 
     *    (values(6) .gt. (max*3)) .OR. (values(6) .lt. (min*3)) .OR. 
     *    (values(8) .gt. max) .OR. (values(8) .lt. min)) then
         call ftests_perror(__LINE__,'Verification Failed')
         call ftests_fail(__FILE__,1)
      end if            

      call ftests_pass(__FILE__)
      end



