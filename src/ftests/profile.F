#include "fpapi.h"
#define NUM 1000
#define THR 10000
#define BUFSIZE 81920

      program profile
      implicit none

      integer*8 values(12)
      integer es
      integer*2 profbuf1(BUFSIZE), profbuf2(BUFSIZE), profbuf3(BUFSIZE)
      integer*2 profbuf4(BUFSIZE), profbuf5(BUFSIZE)
      integer*8 length, start, end
      integer i
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      Integer len_trim
      External len_trim
      integer tests_quiet

      character*(PAPI_MAX_STR_LEN) fullname, name, lib_preload_env
      integer text_start, text_end, data_start, data_end, bss_start
      integer bss_end, check

      print *, 'Run in quiet mode (enter 1 for yes, 0 for no)?'
      read(5,*) tests_quiet

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
      
      call papif_get_exe_info(fullname, name, text_start, text_end,
     *data_start, data_end, bss_start, bss_end, lib_preload_env, check)
      start = text_start
      end = text_end
      length = end - start
C      print *, "length=", length

      if(length .GT. BUFSIZE) then
         print *, "Not enough memory!"
         stop
      end if

      do i=1, BUFSIZE
         profbuf1(i)=0
         profbuf2(i)=0
         profbuf3(i)=0
         profbuf4(i)=0
         profbuf5(i)=0
      end do

      call PAPIf_create_eventset(es, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error in creating EventSet')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_add_event( es, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_FP_INS: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if
      
      call PAPIf_add_event( es, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        if (tests_quiet .EQ. 0) then
        print *, 'Can not add PAPI_TOT_CYC: '//
     *  errorstring(1:len_trim(errorstring))
        end if
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_both(NUM)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Test case 7: SVR4 compatible hardware profiling."
      print *, "------------------------------------------------"
 100  format(A12, Z8, A12, Z8, A12, Z8)
      write (*, 100) "Text start:", text_start, "; Text end:", text_end,
     *"; Text length:", length
 200  format(A12, Z8, A12, Z8)
      write (*, 200) "Data start:", data_start, "; Data end:", data_end
      write (*, 200) "BSS start:", bss_start, "; BSS end:", bss_end
      print *, "Dynamic Library Preload Env. Var.:", lib_preload_env
      print *, "------------------------------------------------"
      print *, "Test type   : \t No profiling"
      print *, "PAPI_FP_INS : \t ", values(1)
      print *, "PAPI_TOT_CYC: \t ", values(2)
      print *, "\n"

      print *, "Test type   : \tPAPI_PROFIL_POSIX"
      end if

      call PAPIf_profil(profbuf1, length, start, 65536, es, 
     *PAPI_FP_INS, THR, PAPI_PROFIL_POSIX, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_both(NUM)

      call PAPIf_stop(es, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "PAPI_FP_INS : \t ", values(3)
      print *, "PAPI_TOT_CYC: \t ", values(4)
      print *, "\n"
      end if

      call PAPIf_profil(profbuf1(1), length, start, 65536, es, 
     *PAPI_FP_INS, 0, PAPI_PROFIL_POSIX, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Test type   : \tPAPI_PROFIL_RANDOM"
      end if
      call PAPIf_profil(profbuf2(1), length, start, 65536, es, 
     *PAPI_FP_INS, THR, PAPI_PROFIL_POSIX+PAPI_PROFIL_RANDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_both(NUM)

      call PAPIf_stop(es, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "PAPI_FP_INS : \t ", values(5)
      print *, "PAPI_TOT_CYC: \t ", values(6)
      print *, "\n"
      end if

      call PAPIf_profil(profbuf2(1), length, start, 65536, es, 
     *PAPI_FP_INS, 0, PAPI_PROFIL_POSIX+PAPI_PROFIL_RANDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Test type   : \tPAPI_PROFIL_WEIGHTED"
      end if
      call PAPIf_profil(profbuf3(1), length, start, 65536, es, 
     *PAPI_FP_INS, THR, PAPI_PROFIL_POSIX+PAPI_PROFIL_WEIGHTED, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_both(NUM)

      call PAPIf_stop(es, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "PAPI_FP_INS : \t ", values(7)
      print *, "PAPI_TOT_CYC: \t ", values(8)
      print *, "\n"
      end if
      call PAPIf_profil(profbuf3(1), length, start, 65536, es, 
     *PAPI_FP_INS, 0, PAPI_PROFIL_POSIX+PAPI_PROFIL_WEIGHTED, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Test type   : \tPAPI_PROFIL_COMPRESS"
      end if

      call PAPIf_profil(profbuf4(1), length, start, 65536, es, 
     *PAPI_FP_INS, THR, PAPI_PROFIL_POSIX+PAPI_PROFIL_COMPRESS, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_both(NUM)
      call PAPIf_stop(es, values(9), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "PAPI_FP_INS : \t ", values(9)
      print *, "PAPI_TOT_CYC: \t ", values(10)
      print *, "\n"
      end if
      call PAPIf_profil(profbuf4(1), length, start, 65536, es, 
     *PAPI_FP_INS, 0, PAPI_PROFIL_POSIX+PAPI_PROFIL_COMPRESS, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "Test type   : \tPAPI_PROFIL_<all>"
      end if

      call PAPIf_profil(profbuf5(1), length, start, 65536, es, 
     *PAPI_FP_INS, THR, PAPI_PROFIL_POSIX+PAPI_PROFIL_COMPRESS+
     *PAPI_PROFIL_RANDOM+PAPI_PROFIL_WEIGHTED, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      call do_both(NUM)
      call PAPIf_stop(es, values(11), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "PAPI_FP_INS : \t ", values(11)
      print *, "PAPI_TOT_CYC: \t ", values(12)
      print *, "\n"
      end if
      call PAPIf_profil(profbuf5(1), length, start, 65536, es, 
     *PAPI_FP_INS, 0, PAPI_PROFIL_POSIX+PAPI_PROFIL_COMPRESS+
     *PAPI_PROFIL_RANDOM+PAPI_PROFIL_WEIGHTED, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_profil')
        call ftests_fail(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, "----------------------------------------------"
      print *, "PAPI_profil() hash table."
      print *, "address\tflat\trandom\tweight\tcompress\tall"
      do i=1, length
         if ((profbuf1(i) .NE. 0) .OR. (profbuf2(i) .NE. 0) .OR. 
     *       (profbuf3(i) .NE. 0) .OR. (profbuf4(i) .NE. 0) .OR. 
     *       (profbuf5(i) .NE. 0)) then
          print *, start+2*i, profbuf1(i), "\t",
     *     start+profbuf2(i), "\t", start+profbuf3(i), "\t",
     *     start+profbuf4(i), "\t", start+profbuf5(i)
         end if
      end do

      print *, "----------------------------------------------"
      end if

      call PAPIf_rem_event( es, PAPI_FP_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_rem_event( es, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_rem_event')
        call ftests_fail(__FILE__,retval)
      end if

      call ftests_pass(__FILE__)
      end


