#include "fpapi.h"
#include "tests.h"


      program native
      implicit none

      integer*8 values(10)
      integer native, es, point
      integer ncpu, nnodes, totalcpus, vendor
      integer model
      character*20 vendor_string, model_string
      real*4 revision, mhz, rsec
      integer*8 uso, usn 
      integer retval, errorcode
      character*(PAPI_MAX_STR_LEN) errorstring

      Integer len_trim
      External len_trim
 
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

#ifdef _AIX
      print *, "AIX"
C     ICM      
      native = 5 * 2**8
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     FPU1CMPL   
      native = 35 * 2**8 + 1
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     LDCM     
      native = 5 * 2**8 + 2
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     LDCMPL
      native = 5 * 2**8 + 3
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     FPU0CMPL
      native = 5 * 2**8 + 4
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     CYC 
      native = 12 * 2**8 + 5
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     FMA
      native = 9 * 2**8 + 6
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     TLB
      native = 0 * 2**8 + 7
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

#endif

#if defined(linux)
C     Data mem refs
      native = 43 * 2**8 + 0
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     Lines out
      native = 47 * 2**8 + 1
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if
#endif

#if defined(mips) .AND. defined(sgi) .AND. defined(unix)
C     L1 I Miss
      native = 9 * 2**8
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     L1 D Miss
      native = 9 * 2**8 + 1
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if
#endif

#if defined(_CRAYT3E)
C     Machine cyc
      native = 0 * 2**8 + 0
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     Dcache acc
      native = 0 * 2**8 + 1
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

C     CPU cyc
      native = 0 * 2**8 + 2
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if

#endif

#if defined(sun) .AND. defined(sparc)
      print *, "Sun Sparc"
C     Load_use
      native = z'B' * 2**8
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if
      
C     DC_wr_hit
      native = z'A' * 2**8 + 1
      call PAPIf_add_event( es, native, retval )
      if ( retval .NE. PAPI_OK ) then
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring(1:len_trim(errorstring))
      end if
      
#endif

#if defined(__ALPHA) .AND. defined(__osf__)
      call get_instr(native)
      if (native .NE. -1) then
         call PAPIf_add_event( es, native, retval )
         if ( retval .NE. PAPI_OK ) then
            call PAPIf_perror( retval, errorstring, errorcode)
            print *, errorstring(1:len_trim(errorstring))
         end if
      end if

      call get_cyc(native)
      if (native .NE. -1) then
         call PAPIf_add_event( es, native, retval )
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
      call sleep(1)
      call do_both(1000)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'error in PAPIf_stop')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_get_real_usec(usn)
      usn = usn - uso

      print *, "-------------Monitor Point ", point,
     *"-------------"
      rsec = usn/1000000.0;
      print *, "Real Elapsed Time in sec.  : ", rsec
#if defined(_AIX)
      csec = values(5)/ (mhz*1000000.0)
      print *, "CPU Elapsed Time in sec.   : ", csec
      print *, "L1 Instruction Cache Misses: ", values(0)
      printf(stderr,"FPU1 Instructions          : ",values(1)
      print *, "L1 Data Cache Load Misses  : ", values(2)
      print *, "Load Instructions          : ", values(3)
      print *, "FPU0 Instructions          : ", values(4)
      print *, "CPU Cycles                 : ", values(5)
      print *, "FMA Instructions           : ", values(6)
      print *, "TLB Misses                 : ", values(7)
      print *, "------------------------------------------"
      print *, "CPU MFLOPS                 : ",
     *((values(4)+values(1))/1000000.0)/csec
      print *, "Real MFLOPS                : ",
     *((values(4)+values(1))/1000000.0)/rsec
      print *, "%% L1 Load Hit Rate         : ",
     *100.0*(1.0 - (values(2)/values(3)))
      print *, "%% FMA Instructions         : ",
     *100.0*values(6)/(values(1)+values(4))
#endif

#if defined(linux)
      print *, "DCU Memory references      : ", values(0)
      print *, "DCU Lines out              : ", values(1)
#endif

#if defined(mips) .AND. defined(sgi)
      print *, "L1 Instruction cache misses       : ", values(0)
      print *, "L1 Data cache misses              : ", values(1)      
#endif

#if defined(_CRAYT3E)
      print *, "Machine Cycles                    : ",values(0)
      print *, "DCache accesses                   : ",values(1)
      print *, "CPU Cycles                        : ",values(2)      
#endif

#if defined(sun) .AND. defined(sparc)
      print *, "Load_use                   : ", values(1)
      print *, "DC_wr_hit                  : ", values(2)
#endif
      call ftests_pass(__FILE__)
      end

#if defined(__ALPHA) .AND. defined(__osf__)
C     NOT TESTED YET
      subroutine get_instr(integer native)
      integer cpu
      GET_CPU_FAMILY(cpu)
      if (cpu .EQ. EV6_CPU) native = PF6_MUX0_RET_INSTRUCTIONS *
     *                               2**8 + 0
      end

      subroutine get_cyc(integer native)
      integer cpu
      GET_CPU_FAMILY(cpu)
      if (cpu .EQ. EV6_CPU) native = PF6_MUX1_CYCLES
     *                               2**8 + 1
      end
#endif
