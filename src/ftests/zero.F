#include "fpapi.h"

      program zero
      implicit none

      integer*8 values(10)
      integer es
      integer*8 uso, usn, cyco, cycn
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
 
      call PAPIf_add_event( es, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
      end if

      call PAPIf_get_real_usec(uso)
      call PAPIf_get_real_cyc(cyco)

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_get_real_usec(usn)
      call PAPIf_get_real_cyc(cycn)

      print *, "Test case 0: start, stop."
      print *, "-----------------------------------------------",
     *"--------------------------"

      call PAPIf_get_domain(es, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_domain')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_domain(domain, domainstr)
      write (*,900) "Default domain is:", domain, domainstr
 900  format(a20, i3, " ", a20)

      call PAPIf_get_granularity(es, granularity, PAPI_GET_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_granularity')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_granularity(granularity, grnstr)
      write (*,800) "Default granularity is:", granularity, grnstr
 800  format(a25, i3, " ", a20)

      print *, "-----------------------------------------------",
     *"--------------------------"
 
      write (*,100) "Test type", 1
#ifdef NO_FLOPS
      write (*,100) "PAPI_FP_INS", values(1)
#else
      write (*,100) "PAPI_TOT_INS", values(1)
#endif
      write (*,100) "PAPI_TOT_CYC", values(2)
      write (*,100) "Real usec", usn-uso
      write (*,100) "Real cycles", cycn-cyco
 100  format(a13, ":", i12)
      print *, "-----------------------------------------------",
     *"--------------------------"
 
      print *, "Verification: none"
     
      call ftests_pass(__FILE__)
      end

