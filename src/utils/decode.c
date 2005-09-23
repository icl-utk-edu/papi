/* This file decodes the preset events into a csv format file */

#include "papi_test.h"
extern int TESTS_QUIET;         /* Declared in test_utils.c */

static void print_help(void)
{
   printf("This is the PAPI decode utility program.\n");
   printf("It decodes PAPI preset events into csv formatted text.\n");
   printf("By default all presets are decoded.\n");
   printf("The text goes to stdout, but can be piped to a file.\n");
   printf("Such a file can be edited in a text editor or spreadsheet.\n");
   printf("It can also be parsed by PAPI_encode_events.\n");
   printf("Usage:\n\n");
   printf("    decode [options]\n\n");
   printf("Options:\n\n");
   printf("  -a            decode only available PAPI preset events\n");
   printf("  -h            print this help message\n");
   printf("\n");
}

int main(int argc, char **argv)
{
   int i,j;
   int retval;
   int print_avail_only = 0;
   PAPI_event_info_t info;
   const PAPI_hw_info_t *hwinfo = NULL;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */
   for (i = 0; i < argc; i++)
      if (argv[i]) {
         if (strstr(argv[i], "-a"))
            print_avail_only = PAPI_PRESET_ENUM_AVAIL;
         if (strstr(argv[i], "-h")) {
            print_help();
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

   i = PAPI_PRESET_MASK;
   printf("name,derived,postfix,short_descr,long_descr,note,[native,...]\n\n");

   do {
      if (PAPI_get_event_info(i, &info) == PAPI_OK) {
	      printf("%s,%s,%s,", info.symbol, info.derived, info.postfix);
         if (info.short_descr[0]) { printf("\"%s\",", info.short_descr); }
         else { printf(","); }
         if (info.long_descr[0]) { printf("\"%s\",", info.long_descr); }
         else { printf(","); }
	      if (info.note[0]) printf("\"%s\"", info.note);

         for (j=0;j<(int)info.count;j++) printf(",%s",info.name[j]);
         printf("\n");
	   }
   } while (PAPI_enum_event(&i, print_avail_only) == PAPI_OK);
   exit(1);
}
