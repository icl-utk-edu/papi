#include "fpapi.h"


      program native
      implicit none

      integer*8 values(10)
      integer nativees, es, point
      integer ncpu, nnodes, totalcpus, vendor
      integer model
      character*20 vendor_string, model_string
      real*4 revision, mhz, rsec
      integer*8 uso, usn 
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

      point = 0

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

#if defined(_AIX)
      print *, "AIX"
C     ICM      
      nativees = 5 * 2**8
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     FPU1CMPL   
      nativees = 35 * 2**8 + 1
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     LDCM     
      nativees = 5 * 2**8 + 2
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     LDCMPL
      nativees = 5 * 2**8 + 3
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     FPU0CMPL
      nativees = 5 * 2**8 + 4
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     CYC 
      nativees = 12 * 2**8 + 5
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     FMA
      nativees = 9 * 2**8 + 6
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     TLB
      nativees = 0 * 2**8 + 7
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

#endif

#if defined(linux)
C     Data mem refs
      nativees = 43 * 2**8 + 0
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     Lines out
      nativees = 47 * 2**8 + 1
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if
#endif

#if (defined(mips) && defined(sgi) && defined(unix))
C     L1 I Miss
      nativees = 9 * 2**8
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     L1 D Miss
      nativees = 9 * 2**8 + 1
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if
#endif

#if defined(_CRAYT3E)
C     Machine cyc
      nativees = 0 * 2**8 + 0
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     Dcache acc
      nativees = 0 * 2**8 + 1
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     CPU cyc
      nativees = 0 * 2**8 + 2
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

#endif

#if (defined(sun) && defined(sparc))
      print *, "Sun Sparc"
C     Load_use
      nativees = z'B' * 2**8
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if
      
C     DC_wr_hit
      nativees = z'A' * 2**8 + 1
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if
      
#endif

#if (defined(__ALPHA) && defined(__osf__))
      call get_instr(nativees)
      if (nativees .NE. -1) then
         call PAPIf_add_event( es, nativees, retval )
         if ( retval .NE. PAPI_OK ) then
            call PAPIf_perror( retval, errorstring, errorcode)
            print *, errorstring(1:len_trim(errorstring))
         end if
      end if

      call get_cyc(nativees)
      if (nativees .NE. -1) then
         call PAPIf_add_event( es, nativees, retval )
         if ( retval .NE. PAPI_OK ) then
            call PAPIf_perror( retval, errorstring, errorcode)
            print *, errorstring(1:len_trim(errorstring))
         end if
      end if
#endif

      call PAPIF_GET_HARDWARE_INFO (ncpu, nnodes, totalcpus, 
     *vendor, vendor_string, model, model_string, revision, mhz)

      call PAPIf_get_real_usec(uso)

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_start')
        call ftests_fail(__FILE__,retval)
      end if

      point = point +1
C     call sleep(1)
      call do_both(1000)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_get_real_usec(usn)
      usn = usn - uso

      if (tests_quiet .EQ. 0) then
      print *, "-------------Monitor Point ", point,
     *"-------------"
      end if
      rsec = usn/1000000.0
      if (tests_quiet .EQ. 0) then
      print *, "Real Elapsed Time in sec.  : ", rsec
#if defined(_AIX)
      csec = values(5)/ (mhz*1000000.0)
      print *, "CPU Elapsed Time in sec.   : ", csec
      print *, "L1 Instruction Cache Misses: ", values(1)
      print *, "FPU1 Instructions          : ", values(2)
      print *, "L1 Data Cache Load Misses  : ", values(3)
      print *, "Load Instructions          : ", values(4)
      print *, "FPU0 Instructions          : ", values(5)
      print *, "CPU Cycles                 : ", values(6)
      print *, "FMA Instructions           : ", values(7)
      print *, "TLB Misses                 : ", values(8)
      print *, "------------------------------------------"
      print *, "CPU MFLOPS                 : ",
     *((values(5)+values(2))/1000000.0)/csec
      print *, "Real MFLOPS                : ",
     *((values(5)+values(2))/1000000.0)/rsec
      print *, "%% L1 Load Hit Rate         : ",
     *100.0*(1.0 - (values(3)/values(4)))
      print *, "%% FMA Instructions         : ",
     *100.0*values(6)/(values(2)+values(5))
#endif

#if defined(linux)
      print *, "DCU Memory references      : ", values(1)
      print *, "DCU Lines out              : ", values(2)
#endif

#if (defined(mips) && defined(sgi))
      print *, "L1 Instruction cache misses       : ", values(1)
      print *, "L1 Data cache misses              : ", values(2)      
#endif

#if defined(_CRAYT3E)
      print *, "Machine Cycles                    : ",values(1)
      print *, "DCache accesses                   : ",values(2)
      print *, "CPU Cycles                        : ",values(3)      
#endif

#if (defined(sun) && defined(sparc))
      print *, "Load_use                   : ", values(1)
      print *, "DC_wr_hit                  : ", values(2)
#endif
      end if
      call ftests_pass(__FILE__)
      end

#if (defined(__ALPHA) && defined(__osf__))
C     NOT TESTED YET
      subroutine get_instr(integer nativees)
      integer cpu
      retval = PAPI_ESBSTR
      call ftests_fail(__FILE__,retval)
c     call GET_CPU_FAMILY(cpu)
c     if (cpu .EQ. EV6_CPU) nativees = PF6_MUX0_RET_INSTRUCTIONS *
c    *                               2**8 + 0
c     end

c     subroutine get_cyc(integer nativees)
c     integer cpu
c     call GET_CPU_FAMILY(cpu)
c     if (cpu .EQ. EV6_CPU) nativees = PF6_MUX1_CYCLES
c    *                               2**8 + 1
c     end
#endif
