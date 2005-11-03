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

static int EventSet = PAPI_NULL;

#if defined(_PPC32)
static char *native_name[] = { "CPU_CLK", "FLOPS", NULL };
#elif defined(_POWER4) || defined(_PPC970)
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
#elif defined(POWER3)
   static char *native_name[] = { "PM_IC_MISS", "PM_FPU1_CMPL", "PM_LD_MISS_L1", "PM_LD_CMPL",
      "PM_FPU0_CMPL", "PM_CYC", "PM_TLB_MISS", NULL
   };
#elif defined(__crayx1)
  /* arbitrarily code 1 event from p, e and m chips */
  static char *native_name[] = {"X1_P_EV_INST_S_FP", "X1_E_EV_REQUESTS", "X1_M_EV_REQUESTS", NULL,NULL,NULL};
#elif defined(__i386__) || defined(__x86_64__) || defined(_WIN32)
   static char *native_name[] = { "DATA_MEM_REFS", "DCU_LINES_IN", NULL, NULL, NULL, NULL };
#elif defined (_CRAYT3E)
  static char *native_name[] = {"MACHINE_CYCLES", "DCACHE_ACCESSES", "CPU_CYC", NULL};
#elif defined(__ia64__)
#ifdef ITANIUM2
   static char *native_name[] = { "CPU_CYCLES", "L1I_READS", "L1D_READS_SET0",
				  "IA64_INST_RETIRED", NULL, NULL
   };
#else
   static char *native_name[] = { "DEPENDENCY_SCOREBOARD_CYCLE", "DEPENDENCY_ALL_CYCLE",
      "UNSTALLED_BACKEND_CYCLE", "MEMORY_CYCLE", NULL
   };
#endif
#elif defined(mips) && defined(sgi) 
   static char *native_name[] = { "Primary_instruction_cache_misses",
      "Primary_data_cache_misses", NULL
   };
#elif defined(sun) && defined(sparc)
   static char *native_name[] = { "Cycle_cnt", "Instr_cnt", NULL };
#elif defined(__ALPHA) && defined(__osf__)
   static char *native_name[] = { "cycles", "retinst", NULL, NULL, NULL, NULL };
#else
#error "Architecture not supported in test file."
#endif

void papimon_start(void)
{
   int retval;

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
}

void papimon_stop(void)
{
   int i, retval;
   long_long values[8];

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      for (i = 0; native_name[i] != NULL; i++) {
         fprintf(stderr, "%-40s: ", native_name[i]);
         fprintf(stderr, LLDFMT, values[i]);
         fprintf(stderr, "\n");
      }
   }
}

int main(int argc, char **argv)
{
   int i, retval, native;
   const PAPI_hw_info_t *hwinfo;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET)
      if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
     test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);
   
   if ((hwinfo = PAPI_get_hardware_info()) == NULL)
     test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", PAPI_EMISC);

   printf("Architecture %s, %d\n",hwinfo->model_string, hwinfo->model);

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

   for (i = 0; native_name[i] != NULL; i++) {
     retval = PAPI_event_name_to_code(native_name[i], &native);
     if (retval != PAPI_OK)
       test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);
     printf("Adding %s\n",native_name[i]);
     if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
       test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   }

   papimon_start();

   do_both(1000);

   papimon_stop();

   test_pass(__FILE__, NULL, 0);
   exit(0);
}
