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

/*
#if defined(linux) && defined(__ia64__)
	#ifdef ITANIUM2
	  #include "pfmlib_itanium2_priv.h"
      #include "itanium2_events.h"
    #else
	  #include "pfmlib_itanium_priv.h"
	  #include "itanium_events.h"
	#endif
#endif
*/

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
static long long us;
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
      "PM_FPU0_CMPL", "PM_CYC", "PM_EXEC_FMA", "PM_TLB_MISS", NULL
   };
#endif
#endif
#endif

#ifdef PENTIUM4
   static char *native_name[] = { "retired_mispred_branch_type_CONDITIONAL", "resource_stall_SBFULL",
      "tc_ms_xfer_CISC", "instr_retired_BOGUSNTAG_BOGUSTAG", "BSQ_cache_reference_RD_2ndL_HITS", NULL
   };
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

#if defined(_AIX)
      for (i = 0; native_name[i] != NULL; i++) {
         /* printf("native_name[%d] = %s\n", i, native_name[i]); */
         retval = PAPI_event_name_to_code(native_name[i], &native);
         /* printf("native_name[%d] = %s; native = 0x%x\n", i, native_name[i], native); */
         if (retval != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      }
#elif defined(linux) && defined(__i386__)
      if (strncmp("AuthenticAMD", hwinfo->vendor_string, (size_t) 3) == 0) {
         native = 0 | 0x40 << 8 | 0;    /* DCU refs */
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
         native = 0 | 0x41 << 8 | 1;    /* DCU miss */
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      } else if (hwinfo->model >= 11) { /* Pentium 4 */
         for (i = 0; native_name[i] != NULL; i++) {
            retval = PAPI_event_name_to_code(native_name[i], &native);
            /* printf("native_name[%d] = %s; native = 0x%x\n", i, native_name[i], native); */
            if (retval != PAPI_OK)
               test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);
            if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
               test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
         }
      } else {
         native = 0 | 0x43 << 8 | 0;    /* Data mem refs */
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
         native = 0 | 0x47 << 8 | 1;    /* Lines out */
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      }
#elif defined(linux) && defined(__ia64__)
      {

         /* Execution latency stall cycles */
#ifdef ITANIUM2
         PAPI_event_name_to_code("CPU_CYCLES", &native);
#else
         PAPI_event_name_to_code("DEPENDENCY_SCOREBOARD_CYCLE", &native);
#endif
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
         /* Combined execution stall cycles */
#ifdef ITANIUM2
         PAPI_event_name_to_code("L1I_READS", &native);
#else
         PAPI_event_name_to_code("DEPENDENCY_ALL_CYCLE", &native);
#endif
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
         /* Combined instruction fetch stall cycles */
#ifdef ITANIUM2
         PAPI_event_name_to_code("L1D_READS_SET0", &native);
#else
         PAPI_event_name_to_code("UNSTALLED_BACKEND_CYCLE", &native);
#endif
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
         /* Combined memory stall cycles */
#ifdef ITANIUM2
         PAPI_event_name_to_code("IA64_INST_RETIRED", &native);
#else
         PAPI_event_name_to_code("MEMORY_CYCLE", &native);
#endif
         if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      }
#elif defined(mips) && defined(sgi) && defined(unix)
      /* See man r10k_counters */
      PAPI_event_name_to_code("Primary_instruction_cache_misses", &native);
      /* L1 I Miss */
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      PAPI_event_name_to_code("Primary_data_cache_misses", &native);
      /* L1 D Miss */
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
#elif defined(_CRAYT3E)
      native = 0 | 0x0 << 8 | 0;        /* Machine cyc */
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      native = 0 | 0xe << 8 | 1;        /* Dcache acc. */
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      native = 0 | 0xC << 8 | 2;        /* CPU cyc */
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
#elif defined(sun) && defined(sparc)
      PAPI_event_name_to_code("Load_use", &native);
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      PAPI_event_name_to_code("DC_wr_hit", &native);
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
#elif defined(__ALPHA) && defined(__osf__)
      PAPI_event_name_to_code("retinst", &native);
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      PAPI_event_name_to_code("cycles", &native);
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
#elif defined(__LINUX__)
      native = 0x28;
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
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
   long long values[8];
   float rsec;
#if defined(_AIX)
   float csec;
#endif

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   us = PAPI_get_real_usec() - us;

   if (!TESTS_QUIET) {
      fprintf(stderr, "-------------Monitor Point %d-------------\n", point);
      rsec = (float) us / 1000000.0;
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
      fprintf(stderr, "FMA Instructions           : %lld\n", values[6]);
      fprintf(stderr, "TLB Misses                 : %lld\n", values[7]);
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
#elif defined(linux) && defined(__i386__)
      if (strncmp("AuthenticAMD", hwinfo->vendor_string, (size_t) 3) == 0) {
         fprintf(stderr, "DCU cache accesses         : %lld\n", values[0]);
         fprintf(stderr, "DCU cache misses           : %lld\n", values[1]);
      } else if (hwinfo->model >= 11) { /* Pentium 4 */
         for (i = 0; native_name[i] != NULL; i++) {
            fprintf(stderr, "%-40s: %lld\n", native_name[i], values[i]);
         }
      } else {
         fprintf(stderr, "DCU Memory references      : %lld\n", values[0]);
         fprintf(stderr, "DCU Lines out              : %lld\n", values[1]);
      }
#elif defined(linux) && defined(__ia64__)
#ifdef ITANIUM2
      fprintf(stderr, "cpu cycles         : %lld\n", values[0]);
      fprintf(stderr, "L1 Inst cache reads     : %lld\n", values[1]);
      fprintf(stderr, "L1 data cache reads  : %lld\n", values[2]);
      fprintf(stderr, "ia64 instructions retired        : %lld\n", values[3]);
#else
      fprintf(stderr, "Execution latency stall cyc         : %lld\n", values[0]);
      fprintf(stderr, "Combined execution stall cycles     : %lld\n", values[1]);
      fprintf(stderr, "Combined instr. fetch stall cycles  : %lld\n", values[2]);
      fprintf(stderr, "Combined memory stall cycles        : %lld\n", values[3]);
#endif
#elif defined(mips) && defined(sgi)
      fprintf(stderr, "L1 Instruction cache misses       : %lld\n", values[0]);
      fprintf(stderr, "L1 Data cache misses              : %lld\n", values[1]);
#elif defined(_CRAYT3E)
      fprintf(stderr, "Machine Cycles                    : %lld\n", values[0]);
      fprintf(stderr, "DCache accesses                   : %lld\n", values[1]);
      fprintf(stderr, "CPU Cycles                        : %lld\n", values[2]);
#elif defined(sun) && defined(sparc)
      fprintf(stderr, "Load_use                   : %lld\n", values[0]);
      fprintf(stderr, "DC_wr_hit                  : %lld\n", values[1]);
#elif defined(__ALPHA) && defined(__osf__)
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

   sleep(1);
   do_both(1000);

   papimon_stop();
   exit(0);
}
