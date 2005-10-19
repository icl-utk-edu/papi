/*
* File:    	native.c
* Mods: 	Maynard Johnson
*			maynardj@us.ibm.com
*/

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

#if defined(_AIX) || defined(linux)
#if defined(_POWER4) || defined(_PPC970)
   /* arbitrarily code events from group 28: pm_fpu3 - Floating point events by unit */
   static char *native_name[] =
       { "PM_FPU0_FDIV", "PM_FPU1_FDIV", "PM_FPU0_FRSP_FCONV", "PM_FPU1_FRSP_FCONV",
      "PM_FPU0_FMA", "PM_FPU1_FMA", "PM_INST_CMPL", "PM_CYC", NULL
   };
#elif defined(_POWER5)
   /* arbitrarily code events from group 78: pm_fpu1 - Floating Point events */
   static char *native_name[] =
       { "PM_FPU_FDIV", "PM_FPU_FMA", "PM_FPU_FMOV_FEST", "PM_FPU_FEST",
       "PM_INST_CMPL", "PM_RUN_CYC", NULL
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


#if defined (__crayx1)
  /* arbitrarily code 1 event from p, e and m chips */
  static char *native_name[] = {"X1_P_EV_INST_S_FP", "X1_E_EV_REQUESTS", "X1_M_EV_REQUESTS", NULL,NULL,NULL};
#endif
#if ((defined(linux) && (defined(__i386__) || (defined __x86_64__))) || defined(_WIN32))
   static char *native_name[] = { "DATA_MEM_REFS", "DCU_LINES_IN", NULL, NULL, NULL, NULL };
#endif

#if defined (_CRAYT3E)
  static char *native_name[] = {"MACHINE_CYCLES", "DCACHE_ACCESSES", "CPU_CYC", NULL};
#endif

#if defined(linux) && defined(__ia64__)
#ifdef ITANIUM2
   static char *native_name[] = { "CPU_CYCLES", "L1I_READS", "L1D_READS_SET0",
				  "IA64_INST_RETIRED", NULL, NULL
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
   static char *native_name[] = { "cycles", "retinst", NULL, NULL, NULL, NULL };
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
      /* printf("Model = %s\n", hwinfo->model_string); */
      if(!strncmp(hwinfo->model_string, "Intel Pentium 4", 15)) {
         native_name[0] = "retired_mispred_branch_type_CONDITIONAL";
         native_name[1] = "resource_stall_SBFULL";
         native_name[2] = "tc_ms_xfer_CISC";
         native_name[3] = "instr_retired_BOGUSNTAG_BOGUSTAG";
         native_name[4] = "BSQ_cache_reference_RD_2ndL_HITS";
         native_name[5] = NULL;
     }
      else if(!strncmp(hwinfo->model_string, "AMD K7", 6)) {
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
#if ((defined(_AIX)) || \
     (defined(linux) && ( defined(__i386__) || ( defined __x86_64__) )) || \
     (defined(_WIN32)) || \
     (defined(linux) && defined(__ia64__)) || \
     (defined(mips) && defined(sgi) && defined(unix)) || \
     (defined(sun) && defined(sparc)) || \
     (defined(__crayx1)) || (defined(_CRAYT3E)) || \
     (defined(__ALPHA) && defined(__osf__)) || \
     (defined(linux) && (defined(_POWER4) || defined(_PPC970) || defined(_POWER5))))

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
#if defined(_AIX) || defined(linux)
   float csec;
#endif

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   us = PAPI_get_real_usec() - us;

   if (!TESTS_QUIET) {
      fprintf(stderr, "-------------Monitor Point %d-------------\n", point);
      rsec = (float)(us / 1000000.0);
      fprintf(stderr, "Real Elapsed Time in sec.  : %f\n", rsec);
#if defined(_AIX)  || defined(linux)
#if defined(_POWER4) || defined(_PPC970)
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
#elif defined (_POWER5)
      csec = (float) values[4] / (hwinfo->mhz * 1000000.0);
      fprintf(stderr, "CPU Elapsed Time in sec.   : %f\n", csec);
      fprintf(stderr, "PM_FPU_FDIV operations      : %lld\n", values[0]);
      fprintf(stderr, "FPU FMA Instructions      : %lld\n", values[1]);
      fprintf(stderr, "CPU run Cycles                 : %lld\n", values[5]);
      fprintf(stderr, "Instructions Completed     : %lld\n", values[4]);
      fprintf(stderr, "------------------------------------------\n");
      fprintf(stderr, "CPU MFLOPS                 : %.2f\n",
              (((float) values[1]) / 500000.0) / csec);
      fprintf(stderr, "%% FMA Instructions         : %.2f\n",
              100.0 * ((float) values[1]) / (float) values[4]);
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
       (defined(__crayx1))|| (defined(_CRAYT3E)) || \
       (defined(__ALPHA) && defined(__osf__)))
      for (i = 0; native_name[i] != NULL; i++) {
         fprintf(stderr, "%-40s: ", native_name[i]);
         fprintf(stderr, LLDFMT, values[i]);
         fprintf(stderr, "\n");
      }
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
