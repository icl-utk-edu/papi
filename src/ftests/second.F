#include "fpapi.h"

      program second
      implicit none
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer*8 values(10), max, min
      integer es1, es2, es3
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      Integer len_trim
      External len_trim

#if (defined(sgi) && defined(host_mips))
      integer id
      id = getuid()
#endif

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftests_perror(__LINE__,'Error in PAPI_library_init ')
        call ftests_fail(__FILE__,retval)
      end if

#if defined(NO_FLOPS)
      call ftests_pass(__FILE__)
#endif

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

      call PAPIf_add_event( es1, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_FP_INS: '//
     *  errorstring(1:len_trim(errorstring))
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
      
      call PAPIf_add_event( es2, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
      end if

      call PAPIf_create_eventset(es3, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( es3, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_FP_INS: '//
     *  errorstring(1:len_trim(errorstring))
      end if
      
      call PAPIf_add_event( es3, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
      end if

      call PAPIf_set_domain1(es1, PAPI_DOM_ALL, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_set_domain')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_set_domain1(es2, PAPI_DOM_KERNEL, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_set_domain')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_set_domain1(es3, PAPI_DOM_USER, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in PAPIf_set_domain')
        call ftests_fail(__FILE__,retval)
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

      call PAPIf_start(es3, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es3, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_rem_event( es1, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
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
      call PAPIf_rem_event( es2, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_rem_event( es3, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if
      call PAPIf_rem_event( es3, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

      print *, "Test case 2: Non-overlapping start, stop, read",
     *" for all 3 domains."
      print *, "----------------------------------------------------------
     *---------------------"

      call PAPIf_get_domain(es1, domain, PAPI_GET_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_domain')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_domain(domain, domainstr)
      write (*,900) "Default domain is:", domain, domainstr
 900  format(a20, i3, " ", a20)

      call PAPIf_get_granularity(es1, granularity, PAPI_GET_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_get_granularity')
        call ftests_fail(__FILE__,retval)
      end if

      call stringify_granularity(granularity, grnstr)
      write (*,800) "Default granularity is:", granularity, grnstr
 800  format(a25, i3, " ", a20)

      print *, "Using", NUM_FLOPS, " iterations of c += b*c"

      print *, "----------------------------------------------------------
     *---------------------"

      print *, "Test type   : \t  PAPI_DOM_ALL\t  PAPI_DOM_KERNEL\t",
     *"  PAPI_DOM_USER"
      print *, "PAPI_FP_INS : \t", values(1), "\t", values(3), "\t\t\t", 
     *values(5)
      print *, "PAPI_TOT_CYC: \t", values(2), "\t", values(4), "\t\t", 
     *values(6)
      print *, "----------------------------------------------------------
     *---------------------"

      print *, "Verification:"
      print *, "Row 1 approximately equals N 0 N"
      print *, "Column 1 approximately equals column 2 plus column 3"

#if (defined(sgi) && defined(host_mips))
      print *, "* Irix requires root for PAPI_DOM_KERNEL and PAPI_DOM_ALL."
      print *, "* The first two rows will be -3 if not run as root for IRIX."
      if (id.NE.0) then
         min = NUM_FLOPS*0.9
         max = NUM_FLOPS*1.1
         if ((values(5) .lt. min) .OR. (values(5) .gt. max)) then
            call ftests_perror(__LINE__,'Verification Failed')
            call ftests_fail(__FILE__,1)
         end if
      else
         min = values(5)*0.9
         max = values(5)*1.1
         if ((values(1) .lt. min) .OR. (values(1) .gt. max)) then
            call ftests_perror(__LINE__,'Verification Failed')
            call ftests_fail(__FILE__,1)
         end if            
         min = values(2)*0.9
         max = values(2)*1.1
         if (((values(4)+values(6)) .lt. min) .OR. 
        *    ((values(4)+values(6)) .gt. max)) then
            call ftests_perror(__LINE__,'Verification Failed')
            call ftests_fail(__FILE__,1)
         end if            
      endif
#else
      min = values(5)*0.9
      max = values(5)*1.1
      if ((values(1) .lt. min) .OR. (values(1) .gt. max)) then
         call ftests_perror(__LINE__,'Verification Failed')
         call ftests_fail(__FILE__,1)
      end if            

      min = values(2)*0.9
      max = values(2)*1.1
      if (((values(4)+values(6)) .lt. min) .OR. 
     *    ((values(4)+values(6)) .gt. max)) then
         call ftests_perror(__LINE__,'Verification Failed')
         call ftests_fail(__FILE__,1)
      end if            
#endif

      call ftests_pass(__FILE__)
      end
