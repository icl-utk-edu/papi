#include "papi_test.h"

/*  Variable to hold reporting status
	if TRUE, output is suppressed
	if FALSE output is sent to stdout
	initialized to FALSE
	declared here so it can be available globally
*/
int TESTS_QUIET = 0;

long_long **allocate_test_space(int num_tests, int num_events)
{
   long_long **values;
   int i;

   values = (long_long **) malloc(num_tests * sizeof(long_long *));
   if (values == NULL)
      exit(1);
   memset(values, 0x0, num_tests * sizeof(long_long *));

   for (i = 0; i < num_tests; i++) {
      values[i] = (long_long *) malloc(num_events * sizeof(long_long));
      if (values[i] == NULL)
         exit(1);
      memset(values[i], 0x00, num_events * sizeof(long_long));
   }
   return (values);
}

void free_test_space(long_long ** values, int num_tests)
{
   int i;

   for (i = 0; i < num_tests; i++)
      free(values[i]);
   free(values);
}

/* Mask tells us what to select. 
	See test_utils.h for mask definitions
*/

int add_test_events(int *number, int *mask)
{
   int retval;
   int EventSet = PAPI_NULL;

   *number = 0;

   retval = PAPI_get_opt(PAPI_MAX_HWCTRS, NULL);
   if (retval < 1)
      test_fail(__FILE__, __LINE__, "PAPI_get_opt", retval);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if (*mask & MASK_L1_DCA) {
      retval = PAPI_add_event(EventSet, PAPI_L1_DCA);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L1_DCA is not available.\n");
         *mask = *mask ^ MASK_L1_DCA;
      }
   }

   if (*mask & MASK_L1_DCW) {
      retval = PAPI_add_event(EventSet, PAPI_L1_DCW);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L1_DCW is not available.\n");
         *mask = *mask ^ MASK_L1_DCW;
      }
   }

   if (*mask & MASK_L1_DCR) {
      retval = PAPI_add_event(EventSet, PAPI_L1_DCR);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L1_DCR is not available.\n");
         *mask = *mask ^ MASK_L1_DCR;
      }
   }

   if (*mask & MASK_L2_TCH) {
      retval = PAPI_add_event(EventSet, PAPI_L2_TCH);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L2_TCH is not available.\n");
         *mask = *mask ^ MASK_L2_TCH;
      }
   }

   if (*mask & MASK_L2_TCA) {
      retval = PAPI_add_event(EventSet, PAPI_L2_TCA);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L2_TCA is not available.\n");
         *mask = *mask ^ MASK_L2_TCA;
      }
   }

   if (*mask & MASK_L2_TCM) {
      retval = PAPI_add_event(EventSet, PAPI_L2_TCM);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L2_TCM is not available.\n");
         *mask = *mask ^ MASK_L2_TCM;
      }
   }

   if (*mask & MASK_L1_DCM) {
      retval = PAPI_add_event(EventSet, PAPI_L1_DCM);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L1_DCM is not available.\n");
         *mask = *mask ^ MASK_L1_DCM;
      }
   }

   if (*mask & MASK_L1_ICM) {
      retval = PAPI_add_event(EventSet, PAPI_L1_ICM);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L1_ICM is not available.\n");
         *mask = *mask ^ MASK_L1_ICM;
      }
   }

   if (*mask & MASK_L1_TCM) {
      retval = PAPI_add_event(EventSet, PAPI_L1_TCM);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_L1_TCM is not available.\n");
         *mask = *mask ^ MASK_L1_TCM;
      }
   }

   if (*mask & MASK_BR_CN) {
      retval = PAPI_add_event(EventSet, PAPI_BR_CN);
      if (retval == PAPI_OK)
         (*number)++;
      else {

         char errstring[PAPI_MAX_STR_LEN];
         PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN);
         if (!TESTS_QUIET) {
            fprintf(stdout,"Error: %s\n", errstring);
            fprintf(stdout, "PAPI_BR_CN is not available.\n");
         }
         *mask = *mask ^ MASK_BR_CN;
      }
   }
   if (*mask & MASK_BR_MSP) {
      retval = PAPI_add_event(EventSet, PAPI_BR_MSP);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_BR_MSP is not available.\n");
         *mask = *mask ^ MASK_BR_MSP;
      }
   }
   if (*mask & MASK_BR_PRC) {
      retval = PAPI_add_event(EventSet, PAPI_BR_PRC);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_BR_PRC is not available.\n");
         *mask = *mask ^ MASK_BR_PRC;
      }
   }

   if (*mask & MASK_FP_OPS) {
      retval = PAPI_add_event(EventSet, PAPI_FP_OPS);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_FP_OPS is not available.\n");
         *mask = *mask ^ MASK_FP_OPS;
      }
   }

   if (*mask & MASK_FP_INS) {
      retval = PAPI_add_event(EventSet, PAPI_FP_INS);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_FP_INS is not available.\n");
         *mask = *mask ^ MASK_FP_INS;
      }
   }

   if (*mask & MASK_TOT_INS) {
      retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_TOT_INS is not available.\n");
         *mask = *mask ^ MASK_TOT_INS;
      }
   }

   if (*mask & MASK_TOT_IIS) {
      retval = PAPI_add_event(EventSet, PAPI_TOT_IIS);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_TOT_IIS is not available.\n");
         *mask = *mask ^ MASK_TOT_IIS;
      }
   }

   if (*mask & MASK_TOT_CYC) {
      retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
      if (retval == PAPI_OK)
         (*number)++;
      else {
         if (!TESTS_QUIET)
            fprintf(stdout, "PAPI_TOT_CYC is not available.\n");
         *mask = *mask ^ MASK_TOT_CYC;
      }
   }

   return (EventSet);
}

int remove_test_events(int *EventSet, int mask)
{
   int retval = PAPI_OK;

   if (mask & MASK_L1_DCA) {
      retval = PAPI_remove_event(*EventSet, PAPI_L1_DCA);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_L1_DCW) {
      retval = PAPI_remove_event(*EventSet, PAPI_L1_DCW);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_L1_DCR) {
      retval = PAPI_remove_event(*EventSet, PAPI_L1_DCR);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_L2_TCH) {
      retval = PAPI_remove_event(*EventSet, PAPI_L2_TCH);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_L2_TCA) {
      retval = PAPI_remove_event(*EventSet, PAPI_L2_TCA);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_L2_TCM) {
      retval = PAPI_remove_event(*EventSet, PAPI_L2_TCM);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_L1_DCM) {
      retval = PAPI_remove_event(*EventSet, PAPI_L1_DCM);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_L1_ICM) {
      retval = PAPI_remove_event(*EventSet, PAPI_L1_ICM);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_L1_TCM) {
      retval = PAPI_remove_event(*EventSet, PAPI_L1_TCM);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_FP_OPS) {
      retval = PAPI_remove_event(*EventSet, PAPI_FP_OPS);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_FP_INS) {
      retval = PAPI_remove_event(*EventSet, PAPI_FP_INS);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_TOT_INS) {
      retval = PAPI_remove_event(*EventSet, PAPI_TOT_INS);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_TOT_IIS) {
      retval = PAPI_remove_event(*EventSet, PAPI_TOT_IIS);
      if (retval < PAPI_OK)
         return (retval);
   }

   if (mask & MASK_TOT_CYC) {
      retval = PAPI_remove_event(*EventSet, PAPI_TOT_CYC);
      if (retval < PAPI_OK)
         return (retval);
   }

   return (PAPI_destroy_eventset(EventSet));
}

char *stringify_domain(int domain)
{
   switch (domain) {
   case PAPI_DOM_USER:
      return ("PAPI_DOM_USER");
   case PAPI_DOM_KERNEL:
      return ("PAPI_DOM_KERNEL");
   case PAPI_DOM_OTHER:
      return ("PAPI_DOM_OTHER");
   case PAPI_DOM_ALL:
      return ("PAPI_DOM_ALL");
   default:
      abort();
   }
   return (NULL);
}

char *stringify_granularity(int granularity)
{
   switch (granularity) {
   case PAPI_GRN_THR:
      return ("PAPI_GRN_THR");
   case PAPI_GRN_PROC:
      return ("PAPI_GRN_PROC");
   case PAPI_GRN_PROCG:
      return ("PAPI_GRN_PROCG");
   case PAPI_GRN_SYS_CPU:
      return ("PAPI_GRN_SYS_CPU");
   case PAPI_GRN_SYS:
      return ("PAPI_GRN_SYS");
   default:
      abort();
   }
   return (NULL);
}

void tests_quiet(int argc, char **argv)
{
   if (argc > 1) {
      if (!strcmp(argv[1], "TESTS_QUIET"))
         TESTS_QUIET = 1;
   }
}

void test_pass(char *file, long_long ** values, int num_tests)
{
   fprintf(stdout,"%-40s PASSED\n", file);
   if (values)
      free_test_space(values, num_tests);
   PAPI_set_debug(PAPI_QUIET);  /* Prevent error messages on Alpha */
   if ( PAPI_is_initialized() ) PAPI_shutdown();
   exit(0);
}

void test_fail(char *file, int line, char *call, int retval)
{
   char buf[128];

   if (retval == PAPI_ESBSTR || retval == PAPI_ENOEVNT || retval == PAPI_ECNFLCT)
      test_skip(file, line, call, retval);
#ifdef PENTIUM4
   /* This can be removed when the P4 substrate is finished */
   if (retval == PAPI_EINVAL)
      test_skip(file, line, call, retval);
#endif
   memset(buf, '\0', sizeof(buf));
   if (retval != 0)
      fprintf(stdout,"%-40s FAILED\nLine # %d\n", file, line);
   else {
      fprintf(stdout,"%-40s SKIPPED\n", file);
      if (!TESTS_QUIET)
         fprintf(stdout,"Line # %d\n", line);
   }
   if (retval == PAPI_ESYS) {
      sprintf(buf, "System error in %s:", call);
      perror(buf);
   } else if (retval > 0) {
      fprintf(stdout,"Error calculating: %s\n", call);
   } else if (retval == 0) {
      fprintf(stdout,"SGI requires root permissions for this test\n");
   } else {
      char errstring[PAPI_MAX_STR_LEN];
      PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN);
      fprintf(stdout,"Error in %s: %s\n", call, errstring);
   }
   fprintf(stdout,"\n");
   if ( PAPI_is_initialized() ) PAPI_shutdown();
   exit(1);
}

void test_skip(char *file, int line, char *call, int retval)
{
   char buf[128];

   memset(buf, '\0', sizeof(buf));
   fprintf(stdout,"%-40s SKIPPED\n", file);
   if (!TESTS_QUIET) {
      if (retval == PAPI_ESYS) {
         fprintf(stdout,"Line # %d\n", line);
         sprintf(buf, "System error in %s:", call);
         perror(buf);
      } else if (retval >= 0) {
         fprintf(stdout,"Line # %d\n", line);
         fprintf(stdout,"Error calculating: %s\n", call);
      } else if (retval < 0) {
         char errstring[PAPI_MAX_STR_LEN];
         fprintf(stdout,"Line # %d\n", line);
         PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN);
         fprintf(stdout,"Error in %s: %s\n", call, errstring);
      }
      fprintf(stdout,"\n");
   }
   if ( PAPI_is_initialized() ) PAPI_shutdown();
   exit(0);
}

#ifdef _WIN32
#undef exit
void wait(char *prompt)
{
   HANDLE hStdIn;
   BOOL bSuccess;
   INPUT_RECORD inputBuffer;
   DWORD dwInputEvents;         /* number of events actually read */

   printf(prompt);
   hStdIn = GetStdHandle(STD_INPUT_HANDLE);
   do {
      bSuccess = ReadConsoleInput(hStdIn, &inputBuffer, 1, &dwInputEvents);
   } while (!(inputBuffer.EventType == KEY_EVENT && inputBuffer.Event.KeyEvent.bKeyDown));
}

int wait_exit(int retval)
{
   if (!TESTS_QUIET)
      wait("Press any key to continue...\n");
   exit(retval);
}

#define exit wait_exit
#endif

void test_print_event_header(char *call, int evset)
{
   int ev_ids[PAPI_MAX_HWCTRS + PAPI_MPX_DEF_DEG];
   int i, nev;
   int retval;
   char evname[PAPI_MAX_STR_LEN];

   nev = PAPI_MAX_HWCTRS + PAPI_MPX_DEF_DEG;
   retval = PAPI_list_events(evset, ev_ids, &nev);

   if (*call)
      fprintf(stdout,"%s", call);
   if (retval == PAPI_OK) {
      for (i = 0; i < nev; i++) {
         PAPI_event_code_to_name(ev_ids[i], evname);
         printf(ONEHDR, evname);
      }
   } else {
      fprintf(stdout,"Can not list event names.");
   }
   fprintf(stdout,"\n");
}
