/* This file performs the following test: hardware info and which events are available */

#include "papi_test.h"
extern int TESTS_QUIET;         /* Declared in test_utils.c */


int main(int argc, char **argv)
{
   int i,j, k;
   int retval;
   int print_full_info = 0;
   char *name = NULL;
   int print_avail_only = 0;
   PAPI_event_info_t info;
   const PAPI_hw_info_t *hwinfo = NULL;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */
   if (argv[1]) {
      if (strstr(argv[1], "-f")) {
         print_full_info = 1;
         name = argv[2];
      }
   }
   for (i = 0; i < argc; i++)
      if (argv[i]) {
         if (strstr(argv[i], "-a"))
            print_avail_only = PAPI_PRESET_ENUM_AVAIL;
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
      printf("Test case 8: Available events and hardware information.\n");
      printf
          ("-------------------------------------------------------------------------\n");
      printf("Vendor string and code   : %s (%d)\n", hwinfo->vendor_string,
             hwinfo->vendor);
      printf("Model string and code    : %s (%d)\n", hwinfo->model_string, hwinfo->model);
      printf("CPU Revision             : %f\n", hwinfo->revision);
      printf("CPU Megahertz            : %f\n", hwinfo->mhz);
      printf("CPU's in this Node       : %d\n", hwinfo->ncpu);
      printf("Nodes in this System     : %d\n", hwinfo->nnodes);
      printf("Total CPU's              : %d\n", hwinfo->totalcpus);
      printf("Number Hardware Counters : %d\n", PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
      printf("Max Multiplex Counters   : %d\n", PAPI_MPX_DEF_DEG);
      printf
          ("-------------------------------------------------------------------------\n");

      if (print_full_info) {
         if (PAPI_event_name_to_code(name, &i) == PAPI_OK) {
            if (PAPI_get_event_info(i, &info) == PAPI_OK) {
               PAPI_event_info_t n_info;
               if (i & PAPI_PRESET_MASK) {
                  printf("Event name:\t\t\t%s\nEvent Code:\t\t\t0x%-10x\nNumber of Native Events:\t%d\n",
		               info.symbol, info.event_code, info.count);
                  printf("Short Description:\t\t|%s|\nLong Description:\t\t|%s|\nDeveloper's Notes:\t\t|%s|\n",
		               info.short_descr, info.long_descr, info.note);
                     printf("Derived Type:\t\t\t|%s|\nPostfix Processing String:\t|%s|\n",
                     info.derived, info.postfix);
                  for (j=0;j<info.count;j++) {
                     printf(" |Native Code[%d]: 0x%x  %s|\n",j,info.code[j], info.name[j]);
                     PAPI_get_event_info(info.code[j], &n_info);
                     printf(" |Number of Register Values: %d|\n", n_info.count);
                     for (k=0;k<n_info.count;k++)
                        printf(" |Register[%d]: 0x%-10x  %s|\n",k, n_info.code[k], n_info.name[k]);
                     printf(" |Native Event Description: |%s|\n\n", n_info.long_descr);
                 }
               }
               else { /* must be a native event code */
                  printf("%s\t0x%x\n |%s|\n",
		               info.symbol,
		               info.event_code,
		               info.long_descr);
                  for (k=0;k<info.count;k++)
                     printf(" |Register Value[%d]: 0x%-10x  %s|\n",k,info.code[k], info.name[k]);
               }
	         }
         }
         else printf("Sorry, an event by the name '%s' could not be found. Is it typed correctly?\n\n",name);
      }
      else {
         printf("The following correspond to fields in the PAPI_event_info_t structure.\n");
         
         printf("Symbol\tEvent Code\tCount\n |Short Description|\n |Long Description|\n |Developer's Notes|\n Derived|\n |PostFix|\n");
         printf("The count field indicates whether it is a) available (count >= 1) and b) derived (count > 1)\n");

         i = PAPI_PRESET_MASK;
         do {
            if (PAPI_get_event_info(i, &info) == PAPI_OK) 
	         {
	            printf("%s\t0x%x\t%d\n |%s|\n |%s|\n |%s|\n |%s|\n |%s|\n",
		            info.symbol,
		            info.event_code,
		            info.count,
		            info.short_descr,
		            info.long_descr,
		            info.note,
                  info.derived,
                  info.postfix);
               for (j=0;j<info.count;j++) printf(" |Native Code[%d]: 0x%x  %s|\n",j,info.code[j], info.name[j]);
	         }
         } while (PAPI_enum_event(&i, print_avail_only) == PAPI_OK);
      }
      printf
          ("-------------------------------------------------------------------------\n");
   }

   test_pass(__FILE__, NULL, 0);
   exit(1);
}
