#include "fpapi.h"

      program first
      IMPLICIT NONE
      
      INTEGER retval
      INTEGER*8 values(10)
      INTEGER*8 max, min
      INTEGER EventSet
      INTEGER errorcode    
      CHARACTER*(PAPI_MAX_STR_LEN) errorstring
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr

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
      
      call PAPIf_create_eventset(EventSet, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( EventSet, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_FP_INS: '//
     *  errorstring(1:len_trim(errorstring))
      end if
      
      call PAPIf_add_event( EventSet, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
      end if
      
      call PAPIf_start(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_read(EventSet, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_read')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_reset(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_reset')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_read(EventSet, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_read')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_read(EventSet, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_read')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(EventSet, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_read(EventSet, values(9), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_read')
        call ftests_fail(__FILE__,retval)
      end if

      print *, "TEST CASE 1: Non-overlapping start, stop, read."
      print *, "--------------------------------------------------------------
     *-----------------------------------"

      call PAPIf_get_domain(EventSet, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_domain')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_domain(domain, domainstr)
      print *, "Default domain is:", domain, " ", domainstr

      call PAPIf_get_granularity(eventset, granularity, PAPI_GET_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_granularity')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_granularity(granularity, grnstr)
      print *, "Default granularity is:", granularity, " ", grnstr

      print *, "Using", NUM_FLOPS, " iterations of c += b*c"
      print *, "--------------------------------------------------------------
     *-----------------------------------"
 
      print *, "Test type    : \t1\t\t2\t\t3\t\t4\t\t5"
      print *, "PAPI_FP_INS  :", values(1), "\t", values(3), "\t", 
     *          values(5), "\t", values(7), "\t", values(9)
      print *, "PAPI_TOT_CYC :", values(2), "\t", values(4), "\t", 
     *          values(6), "\t", values(8), "\t", values(10)
      print *, "--------------------------------------------------------------
     *-----------------------------------"

      print *, "Verification:"
      print *, "Column 1 approximately equals column 2"
      print *, "Column 3 approximately equals 2 * column 2"
      print *, "Column 4 approximately equals 3 * column 2"
      print *, "Column 4 exactly equals column 5"

      min = values(3)*0.9
      max = values(3)*1.1

      if ((values(1).gt.max) .OR. (values(1).lt.min) .OR. 
     *(values(5).gt.(max*2)) .OR. (values(5).lt.(min*2)) .OR.
     *(values(7).gt.(max*3)) .OR. (values(7).lt.(min*3)) .OR.
     *(values(7).NE.values(9))) then
        call ftests_perror(__LINE__,'Verification Failed')
        call ftests_fail(__FILE__,1)
      end if

      min = values(4)*0.9
      max = values(4)*1.1
      if ((values(2).gt.max) .OR. (values(2).lt.min) .OR. 
     *(values(6).gt.(max*2)) .OR. (values(6).lt.(min*2)) .OR.
     *(values(8).gt.(max*3)) .OR. (values(8).lt.(min*3)) .OR.
     *(values(8).NE.values(10))) then
        call ftests_perror(__LINE__,'Verification Failed')
        call ftests_fail(__FILE__,1)
      end if

      
      call ftests_pass(__FILE__)

      end
