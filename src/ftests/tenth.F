#include "fpapi.h"
#define ITERS 100

      program tenth
      implicit none

      integer*8 values(10)
      integer es1, es2, es3
      integer mask1, mask2, mask3
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      Integer len_trim
      External len_trim

#if defined(sun) .AND. defined(sparc)
      mask1 = PAPI_L2_TCM
      mask2 = PAPI_L2_TCA
      mask3 = PAPI_L2_TCH
#else
      mask1 = PAPI_L1_TCM
      mask2 = PAPI_L1_ICM
      mask3 = PAPI_L1_DCM
#endif

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
      
#ifndef _CRAYT3E
      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( es1, mask1, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

#endif      
      call PAPIf_create_eventset(es2, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( es2, mask2, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

      call PAPIf_create_eventset(es3, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( es3, mask3, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

      call do_l1misses(ITERS)

#ifndef _CRAYT3E
      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_l1misses(ITERS)

      call PAPIf_stop(es1, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

#else
      values(1)=0
#endif

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_l1misses(ITERS)
      call PAPIf_stop(es2, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if
      
      call PAPIf_start(es3, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_l1misses(ITERS)

      call PAPIf_stop(es3, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

#ifndef _CRAYT3E
      call PAPIf_rem_event( es1, mask1, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

#endif
      call PAPIf_rem_event( es2, mask2, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_rem_event( es3, mask3, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

#if defined(sun) .AND. defined(sparc)
      print *, "Test case 10: start, stop for derived event ",
     *"PAPI_L2_TCM."
#else
      print *, "Test case 10: start, stop for derived event ",
     *"PAPI_L1_TCM."
#endif
 
      print *, "------------------------------------------------------"
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
      print *, "------------------------------------------------------"
      
      print *, "Test type   : \t  1\t\t  2\t\t  3"
#if defined(sun) .AND. defined(sparc)
      print *, "PAPI_L2_TCM : \t", values(1), "\t  0\t\t  0"
      print *, "PAPI_L2_TCA : \t  0\t\t", values(3), "\t  0"
      print *, "PAPI_L2_TCH : \t  0\t\t  0\t\t", values(5)
      print *, "------------------------------------------------------"
 
      print *, "Verification:"
      print *, "First number row 1 approximately equals (2,2) - (3,3) ", 
     *"or ",(values(3)-values(5))
#else
      print *, "PAPI_L1_TCM : \t", values(1), "\t\t  0\t\t  0"
      print *, "PAPI_L1_ICM : \t  0\t\t", values(3), "\t\t  0"
      print *, "PAPI_L1_DCM : \t  0\t\t  0\t\t", values(5)
      print *, "------------------------------------------------------"
 
      print *, "Verification:"
      print *, "First number row 1 approximately equals (2,2) + (3,3) ",
     *"or ", (values(3)+values(5))
#endif

      call ftests_pass(__FILE__)
      end
