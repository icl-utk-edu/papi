#include "fpapi_test.h"


      program native
      implicit none

      integer*8 values(10)
      integer nativees, es, point
      integer ncpu, nnodes, totalcpus, vendor
      integer model
      character*20 vendor_string, model_string
      real*4 revision, mhz, rsec, csec
      integer*8 uso, usn 
      integer retval

      Integer len_trim
      External len_trim
      integer tests_quiet
      character*25 chbuf
#if defined(_CRAYT3E)
      integer ilen, ierror
#endif

      tests_quiet=0
#if defined(_CRAYT3E)
      call pxfgetarg(1,chbuf,ilen,ierror)
#else
      call getarg(1,chbuf)
#endif
      if (LGE(chbuf, 'TESTS_QUIET')) then
         tests_quiet=1
      end if

      point = 0

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__, 'PAPI_library_init', retval)
      end if

      if ( tests_quiet.EQ.0 ) then
          call PAPIf_set_debug(PAPI_VERB_ECONT, retval)
          if ( retval.NE.PAPI_OK) then
            call ftest_fail(__FILE__, __LINE__, 'PAPIf_set_debug', 
     &       retval)
          end if
      end if
      
      call PAPIf_create_eventset(es, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_create_eventset', 
     *retval)
      end if

C Don't know how to make the following #if defined() be TRUE
C So we'll just do AIX initialization in a C subroutine...
      call addAIXevents(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'addAIXevents', retval)
      end if
#if defined(_AIX)
#if defined(_AIXVERSION_510)
      print *, "AIX POWER4"
C     PM_FPU0_DIV      
      nativees = 10 * 2**8
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     PM_FPU1_DIV   
      nativees = 19 * 2**8 + 1
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     PM_FPU0_FRSP_FCONV     
      nativees = 25 * 2**8 + 2
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     PM_FPU1_FRSP_FCONV
      nativees = 29 * 2**8 + 3
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     PM_FPU0_FMA
      nativees = 11 * 2**8 + 4
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     PM_FPU1_FMA 
      nativees = 20 * 2**8 + 5
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     PM_INST_CMPL
      nativees = 78 * 2**8 + 6
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     PM_CYC
      nativees = 74 * 2**8 + 7
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

#else
      print *, "AIX POWER3"
C     ICM      
      nativees = 5 * 2**8
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     FPU1CMPL   
      nativees = 35 * 2**8 + 1
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     LDCM     
      nativees = 5 * 2**8 + 2
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     LDCMPL
      nativees = 5 * 2**8 + 3
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     FPU0CMPL
      nativees = 5 * 2**8 + 4
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     CYC 
      nativees = 12 * 2**8 + 5
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     FMA
      nativees = 9 * 2**8 + 6
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     TLB
      nativees = 0 * 2**8 + 7
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

#endif
#endif

#if defined(linux) && defined(__i386__)
#if defined(PENTIUM4)
      call PAPIf_event_name_to_code(                                    &
     & 'retired_mispred_branch_type_CONDITIONAL', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('resource_stall_SBFULL',            &
     &  nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('tc_ms_xfer_CISC', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('instr_retired_BOGUSNTAG_BOGUSTAG', &
     & nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('BSQ_cache_reference_RD_2ndL_HITS', &
     & nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#endif
#ifdef PENTIUM3
      call PAPIf_event_name_to_code('DATA_MEM_REFS', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('DCU_LINES_IN', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#endif
#ifdef __ATHLON__
      call PAPIf_event_name_to_code('TOT_CYC', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('IC_MISSES', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('DC_ACCESSES', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('DC_MISSES', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#endif
#ifdef __x86_64__
      call PAPIf_event_name_to_code('FP_ADD_PIPE', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('FP_MULT_PIPE', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('FP_ST_PIPE', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

      call PAPIf_event_name_to_code('FP_NONE_RET', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#endif
#endif

#if defined(linux) && defined(__ia64__)
C     Execution latency stall cycles
#if defined(ITANIUM2)
      call PAPIf_event_name_to_code('CPU_CYCLES', nativees, retval )
#else
      call PAPIf_event_name_to_code('DEPENDENCY_SCOREBOARD_CYCLE',      &
     &    nativees, retval )
#endif
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     Combined execution stall cycles
#if defined(ITANIUM2)
      call PAPIf_event_name_to_code('L1I_READS', nativees, retval )
#else
      call PAPIf_event_name_to_code('DEPENDENCY_ALL_CYCLE',             &
     &       nativees, retval )
#endif
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     Combined instruction fetch stall cycles
#if defined(ITANIUM2)
      call PAPIf_event_name_to_code('L1D_READS_SET0', nativees, retval)
#else
      call PAPIf_event_name_to_code('UNSTALLED_BACKEND_CYCLE',          &
     &  nativees, retval )
#endif
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     Combined memory stall cycles
#if defined(ITANIUM2)
      call PAPIf_event_name_to_code('IA64_INST_RETIRED',nativees,retval)
#else
      call PAPIf_event_name_to_code('MEMORY_CYCLE', nativees, retval )
#endif
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#endif
#if (defined(mips) && defined(sgi) && defined(unix))
C     L1 I Miss
C      nativees = 9*2**8
      call PAPIf_event_name_to_code('Primary_instruction_cache_misse',  &
     &  nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     L1 D Miss
C      nativees = 9 * 2**8 + 1
      call PAPIf_event_name_to_code('Primary_data_cache_misses',        &
     &  nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#endif

#if defined(_CRAYT3E)
C     Machine cyc
      nativees = 0 * 2**8 + 0
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     Dcache acc
      nativees = 0 * 2**8 + 1
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

C     CPU cyc
      nativees = 0 * 2**8 + 2
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if

#endif

#if (defined(sun) && defined(sparc))
C     Load_use
C      nativees = z'B' * 2**8
      call PAPIf_event_name_to_code('Load_use', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
      
C     DC_wr_hit
C      nativees = z'A' * 2**8 + 1
      call PAPIf_event_name_to_code('DC_wr_hit', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
      
#endif

#if (defined(__alpha) && defined(__osf__))
      call PAPIf_event_name_to_code('retinst', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
            call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
      call PAPIf_event_name_to_code('cycles', nativees, retval )
      call PAPIf_add_event( es, nativees, retval )
      if ( retval .NE. PAPI_OK ) then
            call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
      end if
#endif

      call PAPIF_GET_HARDWARE_INFO (ncpu, nnodes, totalcpus, 
     *vendor, vendor_string, model, model_string, revision, mhz)

      call PAPIf_get_real_usec(uso)

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
         call ftest_fail(__FILE__, __LINE__, 'PAPIf_start', retval)
      end if

      point = point +1
C     call sleep(1)
      call do_both(1000)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
         call ftest_fail(__FILE__, __LINE__, 'PAPIf_stop', retval)
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
#if defined(_POWER4)
      csec = values(8)/ (mhz*1000000.0)
      print *, "CPU Elapsed Time in sec.   : ", csec
      print *, "FPU0 DIV Instructions      : ", values(1)
      print *, "FPU1 DIV Instructions      : ", values(2)
      print *, "FPU0 FRSP & FCONV Instr.   : ", values(3)
      print *, "FPU1 FRSP & FCONV Instr.   : ", values(4)
      print *, "FPU0 FMA Instructions      : ", values(5)
      print *, "FPU1 FMA Instructions      : ", values(6)
      print *, "Instructions Completed     : ", values(7)
      print *, "CPU Cycles                 : ", values(8)
      print *, "------------------------------------------"
      print *, "CPU MFLOPS                 : ",
     *((values(5)+values(6))/1000000.0)/csec
      print *, "%% FMA Instructions         : ",
     *100.0*(values(5) + values(6))/values(7)
#endif
#if defined(POWER3)
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
     *100.0*values(7)/(values(2)+values(5))
#endif

#if defined(linux) && defined(__i386__)
#ifdef PENTIUM4
      print *, "retired_mispred_branch_type_CONDITIONAL      : ", values(1)
      print *, "resource_stall_SBFULL                        : ", values(2)
      print *, "tc_ms_xfer_CISC                              : ", values(3)
      print *, "instr_retired_BOGUSNTAG_BOGUSTAG             : ", values(4)
      print *, "BSQ_cache_reference_RD_2ndL_HITS             : ", values(5)
#endif
#ifdef PENTIUM3
      print *, "DATA_MEM_REFS      : ", values(1)
      print *, "DCU_LINES_IN       : ", values(2)
#endif
#ifdef __ATHLON__
      print *, "TOT_CYC            : ", values(1)
      print *, "IC_MISSES          : ", values(2)
      print *, "DC_ACCESSES        : ", values(3)
      print *, "DC_MISSES          : ", values(4)
#endif
#ifdef __x86_64__
      print *, "FP_ADD_PIPE          : ", values(1)
      print *, "FP_MULT_PIPE         : ", values(2)
      print *, "FP_ST_PIPE           : ", values(3)
      print *, "FP_NONE_RET          : ", values(4)
#endif
#endif
#if defined(linux) && defined(__ia64__)
#if defined(ITANIUM2)
      print *, "cpu cycles                  : ", values(1)
      print *, "L1 Inst cache reads         : ", values(2)
      print *, "L1 data cache reads         : ", values(3)
      print *, "ia64 instructions retired   : ", values(4)
#else
      print *, "Execution latency stall cyc        : ", values(1)
      print *, "Combined execution stall cycles    : ", values(2)
      print *, "Combined instr. fetch stall cycles : ", values(3)
      print *, "Combined memory stall cycles       : ", values(4)
#endif
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

#if (defined(__alpha) && defined(__osf__))
      print *, "Retired instructions              : ", values(1)
      print *, "Cycles                            : ", values(2)
#endif
#if (defined(sun) && defined(sparc))
      print *, "Load_use                   : ", values(1)
      print *, "DC_wr_hit                  : ", values(2)
#endif
      end if
      call ftests_pass(__FILE__)
      end

