#include "fpapi_test.h"


      program native
      implicit integer (p)

      integer*8 values(10)
      integer i, n
      integer nativees, es, point
      integer ncpu, nnodes, totalcpus, vendor
      integer model
      character*20 vendor_string, model_string
      real*4 revision, mhz, rsec
      integer*8 uso, usn 
      integer retval

      Integer len_trim
      External len_trim
      character*50 eventname(10)
      integer tests_quiet, get_quiet
      external get_quiet

#if defined(POWER3) || defined (_POWER4) || defined(_POWER5) || defined(_POWER5p)
      real*4 csec
#endif

#if defined(_CRAYT3E) || defined(__crayx1)
      integer ilen, ierror
#endif

      tests_quiet = get_quiet()
      es = PAPI_NULL

      point = 0

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__, 'PAPI_library_init', retval)
      end if

      call PAPIf_create_eventset(es, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_create_eventset', 
     *retval)
      end if

#if defined(_CRAYT3E)
      eventname(1) = 'MACHINE_CYCLES'
      eventname(2) = 'DCACHE_ACCESSES'
      eventname(3) = 'CPU_CYC'
      n = 3
#endif

#if defined(__crayx1)
      eventname(1) = 'X1_P_EV_INST_S_FP'
      eventname(2) = 'X1_E_EV_REQUESTS'
      eventname(3) = 'X1_M_EV_REQUESTS'
      n = 3
#endif

#if defined(PPC32)
      eventname(1) = 'CPU_CLK'
      eventname(2) = 'FLOPS'
      eventname(3) = 'TOT_INS'
      eventname(4) = 'BR_MSP'
      n = 4
#endif
#if defined(_POWER5) || defined(_POWER5p)
      eventname(1) = 'PM_FPU_FDIV'
      eventname(2) = 'PM_FPU_FMA'
      eventname(3) = 'PM_FPU_FMOV_FEST'
      eventname(4) = 'PM_FPU_FEST'
#if defined(_POWER5p)
      eventname(5) = 'PM_RUN_INST_CMPL'
#else
      eventname(5) = 'PM_INST_CMPL'
#endif
      eventname(6) = 'PM_RUN_CYC'
      n = 6
#endif
#if defined(_POWER4) || defined(_PPC970)
      eventname(1) = 'PM_FPU0_FDIV'
      eventname(2) = 'PM_FPU1_FDIV'
      eventname(3) = 'PM_FPU0_FRSP_FCONV'
      eventname(4) = 'PM_FPU1_FRSP_FCONV'
      eventname(5) = 'PM_FPU0_FMA'
      eventname(6) = 'PM_FPU1_FMA'
      eventname(7) = 'PM_INST_CMPL'
      eventname(8) = 'PM_CYC'
      n = 8
#endif
#if defined(POWER3)
      eventname(1) = 'PM_IC_MISS'
      eventname(2) = 'PM_FPU1_CMPL'
      eventname(3) = 'PM_LD_MISS_L1'
      eventname(4) = 'PM_LD_CMPL'
      eventname(5) = 'PM_FPU0_CMPL'
      eventname(6) = 'PM_CYC'
      eventname(7) = 'PM_TLB_MISS'
      n = 7
#endif

#if ((defined(linux) && (defined(__i386__) || (defined __x86_64__))) || defined(_WIN32))
#if defined(PENTIUM4)
      eventname(1) = 'retired_mispred_branch_type_CONDITIONAL'
      eventname(2) = 'resource_stall_SBFULL'
      eventname(3) = 'tc_ms_xfer_CISC'
      eventname(4) = 'instr_retired_BOGUSNTAG_BOGUSTAG'
      eventname(5) = 'BSQ_cache_reference_RD_2ndL_HITS'
      n = 5
#elif defined(__x86_64__)
      eventname(1) = 'FP_ADD_PIPE'
      eventname(2) = 'FP_MULT_PIPE'
      eventname(3) = 'FP_ST_PIPE'
      eventname(4) = 'FP_NONE_RET'
      n = 4
#elif defined(__athlon__)
      eventname(1) = 'TOT_CYC'
      eventname(2) = 'IC_MISSES'
      eventname(3) = 'DC_ACCESSES'
      eventname(4) = 'DC_MISSES'
      n = 4
#else
      eventname(1) = 'DATA_MEM_REFS'
      eventname(2) = 'DCU_LINES_IN'
      n = 2
#endif
#endif

#if defined(linux) && defined(__ia64__)
#if defined(ITANIUM2)
      eventname(1) = 'CPU_CYCLES'
      eventname(2) = 'L1I_READS'
      eventname(3) = 'L1D_READS_SET0'
      eventname(4) = 'IA64_INST_RETIRED'
      n = 4
#else
      eventname(1) = 'DEPENDENCY_SCOREBOARD_CYCLE'
      eventname(2) = 'DEPENDENCY_ALL_CYCLE'
      eventname(3) = 'UNSTALLED_BACKEND_CYCLE'
      eventname(4) = 'MEMORY_CYCLE'
      n = 4
#endif
#endif

#if (defined(mips) && defined(sgi) && defined(unix))
      eventname(1) = 'Primary_instruction_cache_misses'
      eventname(2) = 'Primary_data_cache_misses'
      n = 2
#endif

#if (defined(sun) && defined(sparc))
      eventname(1) = 'Cycle_cnt'
      eventname(2) = 'Instr_cnt'
      n = 2
#endif

#if (defined(__alpha) && defined(__osf__))
      eventname(1) = 'retinst'
      eventname(2) = 'cycles'
      n = 2
#endif

      call PAPIF_GET_HARDWARE_INFO (ncpu, nnodes, totalcpus, 
     *vendor, vendor_string, model, model_string, revision, mhz)

      if(model_string(1:10).eq.'Intel Core')then
         eventname(1) = 'UnhltCore_Cycles'
         eventname(2) = 'Instr_Retired'
         n = 2
      end if

      do i = 1, n
      call PAPIf_event_name_to_code(eventname(i), nativees, retval )
      if ( retval .NE. PAPI_OK ) then
         print *, "Bad Event : ", eventname(i)
         call ftest_fail(__FILE__, __LINE__, 
     &    'PAPIf_event_name_to_code', retval)
      end if
      call PAPIf_add_event( es, nativees, retval )
        if ( retval .NE. PAPI_OK ) then
          call ftest_fail(__FILE__, __LINE__, 'PAPIf_add_event', retval)
        end if
      
      end do

      call PAPIf_get_real_usec(uso)

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
         call ftest_fail(__FILE__, __LINE__, 'PAPIf_start', retval)
      end if

      point = point +1
C     call sleep(1)
      call fdo_both(NUM_ITERS)

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

#if defined(_CRAYT3E)
      print *, "Machine Cycles                    : ",values(1)
      print *, "DCache accesses                   : ",values(2)
      print *, "CPU Cycles                        : ",values(3)

#elif defined(_POWER5) || defined(_POWER5p)
      csec = values(5)/ (mhz*1000000.0)
      print *, "CPU Elapsed Time in sec.   : ", csec
      print *, "PM_FPU_FDIV operations     : ", values(1)
      print *, "FPU FMA Instructions       : ", values(2)
      print *, "Instructions Completed     : ", values(5)
      print *, "CPU Cycles                 : ", values(6)
      print *, "------------------------------------------"
      print *, "CPU MFLOPS                 : ",
     *((values(2))/500000.0)/csec
      print *, "%% FMA Instructions         : ",
     *100.0*(values(2))/values(5)
#elif defined(_POWER4)
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
#elif defined(POWER3)
      csec = values(5)/ (mhz*1000000.0)
      print *, "CPU Elapsed Time in sec.   : ", csec
      print *, "L1 Instruction Cache Misses: ", values(1)
      print *, "FPU1 Instructions          : ", values(2)
      print *, "L1 Data Cache Load Misses  : ", values(3)
      print *, "Load Instructions          : ", values(4)
      print *, "FPU0 Instructions          : ", values(5)
      print *, "CPU Cycles                 : ", values(6)
      print *, "TLB Misses                 : ", values(7)
      print *, "------------------------------------------"
      print *, "CPU MFLOPS                 : ",
     *((values(5)+values(2))/1000000.0)/csec
      print *, "Real MFLOPS                : ",
     *((values(5)+values(2))/1000000.0)/rsec
      print *, "%% L1 Load Hit Rate         : ",
     *100.0*(1.0 - (values(3)/values(4)))
      print *, "%% FMA Instructions         : ",
     *100.0*values(7)/(values(2)+values(5))
#else
      do i = 1, n
        print *, eventname(i)," : ", values(i)
      end do
#endif
      end if
      call ftests_pass(__FILE__)
      end

