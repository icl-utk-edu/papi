/*
   This test needs to be reworked for all platforms.
   Now that each substrate contains a native event table,
   the custom code in this file can be constrained to the 
   native event name arrays.
   Also, we can no longer add raw native bit patterns.
   Further, the output can be driven from the native name array
   by either printing the names directly, or printing the descriptions.
   See Pentium4 code for examples.
*/
#include "papi_test.h"


#if defined(__ALPHA) && defined(__osf__)
#include <machine/hal/cpuconf.h>
#include <sys/pfcntr.h>

long get_instr()
{
   int cpu;
   GET_CPU_FAMILY(&cpu);
   if (cpu == EV6_CPU)
      return (PF6_MUX0_RET_INSTRUCTIONS << 8 | 0);
   else
      abort();
}

long get_cyc()
{
   int cpu;
   GET_CPU_FAMILY(&cpu);
   if (cpu == EV6_CPU)
      return (PF6_MUX1_CYCLES << 8 | 1);
   else
      abort();
}
#endif

static int point = 0;
static int EventSet = PAPI_NULL;
static long_long us;
static const PAPI_hw_info_t *hwinfo;
extern int TESTS_QUIET;         /* Declared in test_utils.c */

#if defined(_AIX)
#if defined(_POWER4)
   /* arbitrarily code events from group 28: pm_fpu3 - Floating point events by unit */
   static char *native_name[] =
       { "PM_FPU0_FDIV", "PM_FPU1_FDIV", "PM_FPU0_FRSP_FCONV", "PM_FPU1_FRSP_FCONV",
      "PM_FPU0_FMA", "PM_FPU1_FMA", "PM_INST_CMPL", "PM_CYC", NULL
   };
#else
#ifdef PMTOOLKIT_1_2
   static char *native_name[] = { "PM_IC_MISS", "PM_FPU1_CMPL", "PM_LD_MISS_L1", "PM_LD_CMPL",
      "PM_FPU0_CMPL", "PM_CYC", "PM_FPU_FMA", "PM_TLB_MISS", NULL
   };
#else
   static char *native_name[] = { "PM_IC_MISS", "PM_FPU1_CMPL", "PM_LD_MISS_L1", "PM_LD_CMPL",
      "PM_FPU0_CMPL", "PM_CYC", "PM_TLB_MISS", NULL
   };
#endif
#endif
#endif

#ifdef PENTIUM4
   static char *native_name[] = { "retired_mispred_branch_type_CONDITIONAL", "resource_stall_SBFULL",
      "tc_ms_xfer_CISC", "instr_retired_BOGUSNTAG_BOGUSTAG", "BSQ_cache_reference_RD_2ndL_HITS", NULL
   };
#elif (defined(linux) && (defined(__i386__) || (defined __x86_64__)))
   static char *native_name[5] = { "DATA_MEM_REFS", "DCU_LINES_IN", NULL };
#endif

#if defined(linux) && defined(__ia64__)
#ifdef ITANIUM2
   static char *native_name[] = { "CPU_CYCLES", "L1I_READS", "L1D_READS_SET0",
      "IA64_INST_RETIRED", NULL
   };
#else
   static char *native_name[] = { "DEPENDENCY_SCOREBOARD_CYCLE", "DEPENDENCY_ALL_CYCLE", 
      "UNSTALLED_BACKEND_CYCLE", "MEMORY_CYCLE", NULL
   };
#endif
#endif

#if ( defined(mips) && defined(sgi) && defined(unix) )
   static char *native_name[] = { "Primary_instruction_cache_misses", 
      "Primary_data_cache_misses", NULL
   };
#endif

#if defined(sun) && defined(sparc)
   static char *native_name[] = { "Cycle_cnt", "Instr_cnt", NULL };
#endif

#if defined(__ALPHA) && defined(__osf__)
   static char *native_name[] = { "cycles", "retinst", NULL };
#endif

void papimon_start(void)
{
   int retval;
   int native;
   int i;

   if (EventSet == PAPI_NULL) {
      if ((hwinfo = PAPI_get_hardware_info()) == NULL)
         test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", PAPI_EMISC);
      if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);
      if(!strncmp(hwinfo->model_string, "AMD K7", 6)) {
         native_name[0] = "TOT_CYC";
         native_name[1] = "IC_MISSES";
         native_name[2] = "DC_ACCESSES";
         native_name[3] = "DC_MISSES";
         native_name[4] = NULL;
      }
      else if(!strncmp(hwinfo->model_string, "AMD K8", 6)) {
         native_name[0] = "FP_ADD_PIPE";
         native_name[1] = "FP_MULT_PIPE";
         native_name[2] = "FP_ST_PIPE";
         native_name[3] = "FP_NONE_RET";
         native_name[4] = NULL;
      }
#if defined(_CRAYT3E)
      native = 0 | 0x0 << 8 | 0;        /* Machine cyc */
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      native = 0 | 0xe << 8 | 1;        /* Dcache acc. */
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      native = 0 | 0xC << 8 | 2;        /* CPU cyc */
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
#elif ((defined(_AIX)) || \
       (defined(linux) && ( defined(__i386__) || ( defined __x86_64__) )) || \
       (defined(_WIN32)) || \
       (defined(linux) && defined(__ia64__)) || \
       (defined(mips) && defined(sgi) && defined(unix)) || \
       (defined(sun) && defined(sparc)) || \
       (defined(__ALPHA) && defined(__osf__)))

      for (i = 0; native_name[i] != NULL; i++) {
         retval = PAPI_event_name_to_code(native_name[i], &native);
         /* printf("native_name[%d] = %s; native = 0x%x\n", i, native_name[i], native); */
         if (retval != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      }

#else
#error "Architecture not included in this test file yet."
#endif
   }

   us = PAPI_get_real_usec();
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   point++;
}

void papimon_stop(void)
{
   int i, retval;
   long_long values[8];
   float rsec;
#if defined(_AIX)
   float csec;
#endif

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   us = PAPI_get_real_usec() - us;

   if (!TESTS_QUIET) {
      fprintf(stderr, "-------------Monitor Point %d-------------\n", point);
      rsec = (float)(us / 1000000.0);
      fprintf(stderr, "Real Elapsed Time in sec.  : %f\n", rsec);
#if defined(_AIX)
#if defined(_POWER4)
      csec = (float) values[7] / (hwinfo->mhz * 1000000.0);
      fprintf(stderr, "CPU Elapsed Time in sec.   : %f\n", csec);
      fprintf(stderr, "FPU0 DIV Instructions      : %lld\n", values[0]);
      fprintf(stderr, "FPU1 DIV Instructions      : %lld\n", values[1]);
      fprintf(stderr, "FPU0 FRSP & FCONV Instr.   : %lld\n", values[2]);
      fprintf(stderr, "FPU1 FRSP & FCONV Instr.   : %lld\n", values[3]);
      fprintf(stderr, "FPU0 FMA Instructions      : %lld\n", values[4]);
      fprintf(stderr, "FPU1 FMA Instructions      : %lld\n", values[5]);
      fprintf(stderr, "Instructions Completed     : %lld\n", values[6]);
      fprintf(stderr, "CPU Cycles                 : %lld\n", values[7]);
      fprintf(stderr, "------------------------------------------\n");
      fprintf(stderr, "CPU MFLOPS                 : %.2f\n",
              (((float) values[4] + (float) values[5]) / 500000.0) / csec);
      fprintf(stderr, "%% FMA Instructions         : %.2f\n",
              100.0 * ((float) values[4] + (float) values[5]) / (float) values[6]);
#else
      csec = (float) values[5] / (hwinfo->mhz * 1000000.0);
      fprintf(stderr, "CPU Elapsed Time in sec.   : %f\n", csec);
      fprintf(stderr, "L1 Instruction Cache Misses: %lld\n", values[0]);
      fprintf(stderr, "FPU1 Instructions          : %lld\n", values[1]);
      fprintf(stderr, "L1 Data Cache Load Misses  : %lld\n", values[2]);
      fprintf(stderr, "Load Instructions          : %lld\n", values[3]);
      fprintf(stderr, "FPU0 Instructions          : %lld\n", values[4]);
      fprintf(stderr, "CPU Cycles                 : %lld\n", values[5]);
#ifdef PMTOOLKIT_1_2
      fprintf(stderr, "FMA Instructions           : %lld\n", values[6]);
      fprintf(stderr, "TLB Misses                 : %lld\n", values[7]);
#else
      fprintf(stderr, "TLB Misses                 : %lld\n", values[6]);
#endif
      fprintf(stderr, "------------------------------------------\n");
      fprintf(stderr, "CPU MFLOPS                 : %.2f\n",
              (((float) values[4] + (float) values[1]) / 1000000.0) / csec);
      fprintf(stderr, "Real MFLOPS                : %.2f\n",
              (((float) values[4] + (float) values[1]) / 1000000.0) / rsec);
      fprintf(stderr, "%% L1 Load Hit Rate         : %.2f\n",
              100.0 * (1.0 - ((float) values[2] / (float) values[3])));
      fprintf(stderr, "%% FMA Instructions         : %.2f\n",
              100.0 * (float) values[6] / ((float) values[1] + (float) values[4]));
#endif
#elif ((defined(linux) && ( defined(__i386__) || defined(__x86_64__) )) || \
       (defined(_WIN32)) || \
       (defined(linux) && defined(__ia64__)) || \
       (defined(mips) && defined(sgi)) || \
       (defined(sun) && defined(sparc)) || \
       (defined(__ALPHA) && defined(__osf__)))
      for (i = 0; native_name[i] != NULL; i++) {
         fprintf(stderr, "%-40s: ", native_name[i]);
         fprintf(stderr, LLDFMT, values[i]);
         fprintf(stderr, "\n");
      }
#elif defined(_CRAYT3E)
      fprintf(stderr, "Machine Cycles                    : %lld\n", values[0]);
      fprintf(stderr, "DCache accesses                   : %lld\n", values[1]);
      fprintf(stderr, "CPU Cycles                        : %lld\n", values[2]);
      fprintf(stderr, "Load_use                   : %lld\n", values[0]);
      fprintf(stderr, "DC_wr_hit                  : %lld\n", values[1]);
      fprintf(stderr, "Retired Instructions       : %lld\n", values[0]);
      fprintf(stderr, "Cycles                     : %lld\n", values[1]);
#endif
   }
   test_pass(__FILE__, NULL, 0);
}

int main(int argc, char **argv)
{
   int retval;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET)
      if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);

   papimon_start();

   /*sleep(1); */
   do_both(1000);

   papimon_stop();
   exit(0);
}
