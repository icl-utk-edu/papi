#include "fpapi.h"
#include "tests.h"

      program third
      implicit none

      integer*8 values(10), min, max
      character*8 zero
      integer es1, es2
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      Integer len_trim
      External len_trim

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

      call PAPIf_add_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
      end if

      call PAPIf_create_eventset(es2, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( es2, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_FP_INS: '//
     *  errorstring(1:len_trim(errorstring))
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

      call PAPIf_read(es1, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_read')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, values(7), retval)
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

      call PAPIf_stop(es2, values(9), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_rem_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_rem_event( es2, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

      zero="  0     "
      print *, "Test case 3: Overlapping start and stop of 2 eventsets",
     *" with different counters."
      print *, "----------------------------------------------------------
     *---------------------"

      call PAPIf_get_domain(es1, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_domain')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_domain(domain, domainstr)
      print *, "Default domain is:", domain, " ", domainstr

      call PAPIf_get_granularity(es1, granularity, PAPI_GET_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_granularity')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_granularity(granularity, grnstr)
      print *, "Default granularity is:", granularity, " ", grnstr
      print *, "Using", NUM_FLOPS, " iterations of c += b*c"

      print *, "--------------------------------------------------------------
     *---------------------"

      print *, "Test type   : \t  1\t\t  2\t\t  3\t\t  4\t\t  5"
      print *, "PAPI_FP_INS : \t", zero, "\t", zero, "\t", values(5), 
     *"\t", zero, "\t", values(9)
      print *, "PAPI_TOT_CYC: \t", values(1), "\t", values(3), 
     *"\t", zero, "\t", values(7), "\t", zero
      print *, "----------------------------------------------------------
     *---------------------"

      print *, "Verification:"
      print *, "Row 1 approximately equals ", 0, 0, "  N", 0, "  N"
      print *, "Row 2 approximately equals X  X", 0, "  X", 0
      print *, "Column 1 approximately equals column 2"
      print *, "Column 4 approximately equals three times column 1"
      print *, "Column 5 approximately equals column 3"

      min = values(5)*0.9
      max = values(5)*1.1
      if ((values(5) .EQ. 0) .OR. (values(9) .EQ. 0) .OR. 
     *    (values(9) .gt. max) .OR. (values(9) .lt. min)) then
         call ftests_perror(__LINE__,'Verification Failed')
         call ftests_fail(__FILE__,1)
      end if            

      min = values(1)*0.9
      max = values(1)*1.1
      if ((values(1).EQ.0).OR.(values(3).EQ.0).OR.(values(7).EQ.0) .OR.
     *    (values(3) .gt. max) .OR. (values(3) .lt. min) .OR.
     *    (values(7) .gt. (max*3)) .OR. (values(7) .lt. (min*3))) then
         call ftests_perror(__LINE__,'Verification Failed')
         call ftests_fail(__FILE__,1)
      end if            

      call ftests_pass(__FILE__)
      end
