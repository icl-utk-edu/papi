/* This file performs the following test: hardware info and which native events are available */

#include "papi_test.h"
extern int TESTS_QUIET;         /* Declared in test_utils.c */

static void print_help(char **argv)
{
   printf("Usage: %s [-dh]\n",argv[0]);
   printf("Options:\n\n");
   printf("\t-d            Display detailed information about all native events\n");
   printf("\t-h            Print this help message\n");
   printf("\n");
   printf("This program provides information about PAPI native events.\n");
}

int main(int argc, char **argv)
{
   int i, j, k;
   int retval;
   PAPI_event_info_t info;
   const PAPI_hw_info_t *hwinfo = NULL;
#ifdef _POWER4
   int group = 0;
#endif
#ifdef PENTIUM4
   int l;
#endif
   int print_event_info = 0;
   char *name = NULL;
   int print_tabular = 0;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   for (i = 0; i < argc; i++)
     {
       if (strstr(argv[i], "-e")) 
	 {
	   print_event_info = 1;
	   name = argv[i+1];
	   if ((name == NULL) || (strlen(name) == 0))
	     {
	       print_help(argv);
	       exit(1);
	     }
	 }
       else if (strstr(argv[i], "-d"))
	 print_tabular = 1;
       else if (strstr(argv[i], "-h")) {
	 print_help(argv);
	 exit(1);
       }
     }

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET) {
      retval = PAPI_set_debug(PAPI_VERB_ECONT);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   }

   if ((hwinfo = PAPI_get_hardware_info()) == NULL)
      test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   if (!TESTS_QUIET) {
      printf
          ("Available native events and hardware information.\n");
      printf
          ("-------------------------------------------------------------------------\n");
      printf("Vendor string and code   : %s (%d)\n", hwinfo->vendor_string,
             hwinfo->vendor);
      printf("Model string and code    : %s (%d)\n", hwinfo->model_string, hwinfo->model);
      printf("CPU Revision             : %f\n", hwinfo->revision);
      printf("CPU Megahertz            : %f\n", hwinfo->mhz);
      printf("CPU Clock Megahertz      : %d\n", hwinfo->clock_mhz);
      printf("CPU's in this Node       : %d\n", hwinfo->ncpu);
      printf("Nodes in this System     : %d\n", hwinfo->nnodes);
      printf("Total CPU's              : %d\n", hwinfo->totalcpus);
      printf("Number Hardware Counters : %d\n", PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
      printf("Max Multiplex Counters   : %d\n", PAPI_get_opt(PAPI_MAX_MPX_CTRS, NULL));
      printf
          ("-------------------------------------------------------------------------\n");

      printf("The following correspond to fields in the PAPI_event_info_t structure.\n\n");
      
      printf("%-32s %-12s %s\n Register Name[n]\n Register Value[n]\n\n","Symbol","Event Code","Long Description");

   }
   i = 0 | PAPI_NATIVE_MASK;
   j = 0;
#ifdef __crayx1
   PAPI_enum_event(&i, 0);
#endif
   do {
#if defined(__crayx2)						/* CRAY X2 */
	/*	M chips on the Cray X2 must be qualified as to the
	 *	M chip ID upon which the event is being counted.
	 *	Therefore a chip ID must always be present. To get
	 *	the ball rolling set the bit for chip ID 0 when
	 *	the first M chip event is encountered.
	 */
      if ((i & 0x00000FFF) >= (32*4 + 16*4) && ((i & 0x0FFFF000) == 0)) {
	i |= 0x00001000;
      }
#endif
#ifdef _POWER4
      group = (i & 0x00FF0000) >> 16;
      if (group) {
         if (!TESTS_QUIET)
            printf("%4d", group - 1);
      } else {
         if (!TESTS_QUIET)
            printf("\n\n");
#endif
         j++;
	 memset(&info,0,sizeof(info));
         retval = PAPI_get_event_info(i, &info);

	 /* This event may not exist */
	 if (retval == PAPI_ENOEVNT)
	   continue;

	 printf("%-32s 0x%-10x %s\n",
		info.symbol,
		info.event_code,
		info.long_descr);

     if (print_tabular)
       {
	 for (k=0;k<(int)info.count;k++)
	   {
	     printf(" Register Name[%d]: %s\n",k,info.name[k]);
	     printf(" Register Value[%d]: 0x%-10x\n",k,info.code[k]);
	   }
	 if (k) printf("\n");
       }
#ifdef _POWER4
         if (!TESTS_QUIET)
            printf("Groups: ");
      }
#endif
#ifdef PENTIUM4
      k = i;
      if (PAPI_enum_event(&k, PAPI_PENT4_ENUM_BITS) == PAPI_OK) {
         l = strlen(info.long_descr);
         do {
            j++;
            retval = PAPI_get_event_info(k, &info);
            if (!TESTS_QUIET && retval == PAPI_OK) {
               printf("%-26s 0x%-10x %s\n",
                      info.symbol, info.event_code, info.long_descr + l);
            }
         } while (PAPI_enum_event(&k, PAPI_PENT4_ENUM_BITS) == PAPI_OK);
      }
      if (!TESTS_QUIET && retval == PAPI_OK)
         printf("\n");
   } while (PAPI_enum_event(&i, PAPI_PENT4_ENUM_GROUPS) == PAPI_OK);
#elif defined(_POWER4)
/* this function would return the next native event code.
    modifer = PAPI_ENUM_ALL
		 it simply returns next native event code
    modifer = PAPI_PWR4_ENUM_GROUPS
		 it would return information of groups this native event lives
                 0x400000ed is the native code of PM_FXLS_FULL_CYC,
		 before it returns 0x400000ee which is the next native event's
		 code, it would return *EventCode=0x400400ed, the digits 16-23
		 indicate group number
   function return value:
     PAPI_OK successful, next event is valid
     PAPI_ENOEVNT  fail, next event is invalid
*/
   } while (PAPI_enum_event(&i, PAPI_PWR4_ENUM_GROUPS) == PAPI_OK);
#elif defined(__crayx2)					/* CRAY X2 */
   } while (PAPI_enum_event(&i, PAPI_ENUM_UMASKS_CRAYX2) == PAPI_OK);
#else
   } while (PAPI_enum_event(&i, PAPI_ENUM_ALL) == PAPI_OK);
#endif

   if (!TESTS_QUIET) {
      printf
          ("\n-------------------------------------------------------------------------\n");
      printf("Total events reported: %d\n", j);
   }
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
