/* This test file reads data from a csv file and uses the PAPI_set_event_info
   funtion to change preset event settings.
*/

#include "papi_test.h"
extern int TESTS_QUIET;         /* Declared in test_utils.c */

static void print_help(void)
{
   printf("This is the PAPI encode test program.\n");
   printf("It reads a cvs formatted file and loads the PAPI preset event table.\n");
   printf("It then displays the modified table according to the optional parameters.\n");
   printf("Usage:\n\n");
   printf("    encode [options] filename\n");
   printf("    encode TESTS_QUIET\n\n");
   printf("Options:\n\n");
   printf("  -a            display only available PAPI preset events\n");
   printf("  -d            display PAPI preset event info in detailed format\n");
   printf("  -h            print this help message\n");
   printf("  -n            no display of preset table\n");
   printf("  -r            replace existing presets\n");
   printf("  -t            display PAPI preset event info in tabular format (default)\n");
   printf("\n");
}

#if 0
static void print_this_event_info(PAPI_event_info_t *info) {
   PAPI_event_info_t n_info;
   int j, k;

   printf("Event name:\t\t\t%s\nEvent Code:\t\t\t0x%-10x\nNumber of Native Events:\t%d\n",
	   info->symbol, info->event_code, info->count);
   printf("Short Description:\t\t|%s|\nLong Description:\t\t|%s|\nDeveloper's Notes:\t\t|%s|\n",
	   info->short_descr, info->long_descr, info->note);
      printf("Derived Type:\t\t\t|%s|\nPostfix Processing String:\t|%s|\n",
      info->derived, info->postfix);
   for (j=0;j<(int)info->count;j++) {
      printf(" |Native Code[%d]: 0x%x  %s|\n",j,info->code[j], info->name[j]);
      PAPI_get_event_info(info->code[j], &n_info);
      printf(" |Number of Register Values: %d|\n", n_info.count);
      for (k=0;k<(int)n_info.count;k++)
         printf(" |Register[%d]: 0x%-10x  %s|\n",k, n_info.code[k], n_info.name[k]);
      printf(" |Native Event Description: |%s|\n\n", n_info.long_descr);
   }
}
#endif

static char *quotcpy(char *d, char *s) {
   if (s && *s == '\"') {
      s++;
      s[strlen(s)-1]=0;
   }
   strcpy(d, s);
   return (d);
}

int main(int argc, char **argv)
{
   int i, j, line, field;
   int retval;
   int replace_event = 0;
   char *name = NULL;
   int print_avail_only = PAPI_PRESET_ENUM_AVAIL;
   int print_tabular = 0;
   int print_table = 1;
   long long ec;
   PAPI_event_info_t info;
   const PAPI_hw_info_t *hwinfo = NULL;
	FILE *file = NULL;
   char buf[PAPI_HUGE_STR_LEN];
   char *b, *arg[20];
   int quote;


   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */
   if (TESTS_QUIET) test_skip(__FILE__, __LINE__, "test_skip", 1);

   for (i = 1; i < argc; i++) {
      if (argv[i][0] == '-') {
         if (strstr(argv[i], "a")) {
            print_avail_only = PAPI_PRESET_ENUM_AVAIL;
         }
         if (strstr(argv[i], "d")) {
            print_tabular = 0;
        }
         if (strstr(argv[i], "n")) {
            print_table = 0;
         }
         if (strstr(argv[i], "r")) {
            replace_event = 1;
         }
         if (strstr(argv[i], "t")) {
            print_tabular = 1;
        }
         if (strstr(argv[i], "h")) {
            print_help();
            exit(1);
         }
      }
      else name = argv[i];
   }
	
   if (name == NULL) {
      if (!TESTS_QUIET) print_help();
      test_skip(__FILE__, __LINE__, "fopen: no filename given", PAPI_EINVAL);
   }
   file = fopen(name, "r");
	if (file == NULL) {
      test_fail(__FILE__, __LINE__, "fopen", PAPI_EINVAL);
   }
   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if ((hwinfo = PAPI_get_hardware_info()) == NULL)
      test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   line = 0;
   while (fgets(buf, sizeof(buf), file)) {
      field = 0;
      quote = 0;
      b = buf;
      arg[field] = b;
      do {
         switch (*b) {
            case '\r':
            case '\n':
               /* strip line endings if not inside quotes */
               if (!quote) *b = 0;
               break;
            case '\"':
               /* toggle quoted state */
               quote = !quote;
               break;
            case ',':
               /* change fields if not inside quotes */
               if (!quote) {
                  *b = 0;
                  arg[++field] = b+1;
                  arg[field+1] = NULL;
               }
               break;
            default:
               break;
         }
      } while (*(++b));

      if (line == 0) {
      }
      else if (line > 1) {
         quotcpy(info.symbol, arg[0]);
         quotcpy(info.derived, arg[1]);
         quotcpy(info.postfix, arg[2]);
         quotcpy(info.short_descr, arg[3]);
         quotcpy(info.long_descr, arg[4]);
         quotcpy(info.note, arg[5]);
         info.event_code = PAPI_PRESET_MASK;
         info.count = 0;
         for (j = 0; j < 8; j++) {
            if (!arg[j+6]) break;
            quotcpy(info.name[j], arg[j+6]);
            info.count++;
         }
/*         print_this_event_info(&info);*/
         printf("Setting preset event info for: %s\n", info.symbol);
         retval = PAPI_set_event_info(&info, &ec, replace_event);
         printf("Returns: %d\n", retval);
         if (retval == PAPI_ESBSTR) printf("This feature isn't supported!\n");
         if (retval == PAPI_EPERM) printf("Non-modifiable event conflict!\n");
         if (retval == PAPI_EISRUN) printf("Event is referenced from an EventSet!\n");
         if (retval == PAPI_EINVAL) printf("This event data doesn't make sense!\n");
         if (retval == PAPI_ENOTPRESET) printf("No room in preset table!\n");
         if (retval == PAPI_ENOEVNT) printf("Event isn't a preset!\n");

      }
      line++;
   }
	fclose(file);

   if (print_table) {
      ec = PAPI_PRESET_MASK;
      printf ("-------------------------------------------------------------------------\n");
      if (print_tabular) {
         if (print_avail_only) {
            printf("Name\t\tDerived\tDescription (Mgr. Note)\n");
         } else {
            printf("Name\t\tCode\t\tAvail\tDeriv\tDescription (Note)\n");
         }
      }
      else {
         printf("The following correspond to fields in the PAPI_event_info_t structure.\n");
         printf("Symbol\tEvent Code\tCount\n |Short Description|\n |Long Description|\n |Developer's Notes|\n Derived|\n |PostFix|\n");
      }
      printf ("-------------------------------------------------------------------------\n");
      do {
         if (PAPI_get_event_info(ec, &info) == PAPI_OK) {
            if (print_tabular) {
               if (print_avail_only) {
		            printf("%s\t%s\t%s (%s)\n",
			            info.symbol,
			            (info.count > 1 ? "Yes" : "No"),
			            info.long_descr, (info.note ? info.note : ""));
               } else {
		            printf("%s\t0x%llx\t%s\t%s\t%s (%s)\n",
		                  info.symbol,
		                  info.event_code,
		                  (info.count ? "Yes" : "No"),
		                  (info.count > 1 ? "Yes" : "No"),
		                  info.long_descr, (info.note ? info.note : ""));
               }
            } else {
	            printf("\n%s\t0x%llx\t%d\n |%s|\n |%s|\n |%s|\n |%s|\n |%s|\n",
		            info.symbol,
		            info.event_code,
		            info.count,
		            info.short_descr,
		            info.long_descr,
		            info.note,
                  info.derived,
                  info.postfix);
               for (j=0;j<(int)info.count;j++) printf(" |Native Code[%d]: 0x%llx  %s|\n",j,info.code[j], info.name[j]);
            }
	      }
      } while (PAPI_enum_event(&ec, print_avail_only) == PAPI_OK);
      printf ("-------------------------------------------------------------------------\n");
   }
   test_pass(__FILE__, NULL, 0);
   exit(1);
}


