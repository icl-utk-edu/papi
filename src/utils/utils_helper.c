#include <stdio.h>
#include <sys/utsname.h>
#include <string.h>

#include "papi.h"

/*  Support routine to display header information to the screen
        from the hardware info data structure. The same code was duplicated
        in a number of tests and utilities. Seems to make sense to refactor.
        This may not be the best place for it to live, but it works for now.
 */
int
papi_print_header( char *prompt, const PAPI_hw_info_t ** hwinfo )
{
        int cnt, mpx;
        struct utsname uname_info;
        PAPI_option_t options;
  memset(&options,0,sizeof(PAPI_option_t));

        if ( ( *hwinfo = PAPI_get_hardware_info(  ) ) == NULL ) {
                return PAPI_ESYS;
        }

        PAPI_get_opt(PAPI_COMPONENTINFO,&options);

        uname(&uname_info);

        printf( "%s", prompt );
        printf
                ( "--------------------------------------------------------------------------------\n" );
        printf( "PAPI version             : %d.%d.%d.%d\n",
                        PAPI_VERSION_MAJOR( PAPI_VERSION ),
                        PAPI_VERSION_MINOR( PAPI_VERSION ),
                        PAPI_VERSION_REVISION( PAPI_VERSION ),
                        PAPI_VERSION_INCREMENT( PAPI_VERSION ) );
        printf( "Operating system         : %s %s\n",
                uname_info.sysname, uname_info.release);
        printf( "Vendor string and code   : %s (%d, 0x%x)\n",
                        ( *hwinfo )->vendor_string,
                        ( *hwinfo )->vendor,
                        ( *hwinfo )->vendor );
        printf( "Model string and code    : %s (%d, 0x%x)\n",
                        ( *hwinfo )->model_string,
                        ( *hwinfo )->model,
                        ( *hwinfo )->model );
        printf( "CPU revision             : %f\n", ( *hwinfo )->revision );
        if ( ( *hwinfo )->cpuid_family > 0 ) {
                printf( "CPUID                    : Family/Model/Stepping %d/%d/%d, "
                        "0x%02x/0x%02x/0x%02x\n",
                        ( *hwinfo )->cpuid_family,
                        ( *hwinfo )->cpuid_model,
                        ( *hwinfo )->cpuid_stepping,
                        ( *hwinfo )->cpuid_family,
                        ( *hwinfo )->cpuid_model,
                        ( *hwinfo )->cpuid_stepping );
        }
        printf( "CPU Max MHz              : %d\n", ( *hwinfo )->cpu_max_mhz );
        printf( "CPU Min MHz              : %d\n", ( *hwinfo )->cpu_min_mhz );
        printf( "Total cores              : %d\n", ( *hwinfo )->totalcpus );

        if ( ( *hwinfo )->threads > 0 )
                printf( "SMT threads per core     : %d\n", ( *hwinfo )->threads );
        if ( ( *hwinfo )->cores > 0 )
                printf( "Cores per socket         : %d\n", ( *hwinfo )->cores );
        if ( ( *hwinfo )->sockets > 0 )
                printf( "Sockets                  : %d\n", ( *hwinfo )->sockets );
        printf( "Cores per NUMA region    : %d\n", ( *hwinfo )->ncpu );
        printf( "NUMA regions             : %d\n", ( *hwinfo )->nnodes );
        printf( "Running in a VM          : %s\n", ( *hwinfo )->virtualized?
                "yes":"no");
        if ( (*hwinfo)->virtualized) {
           printf( "VM Vendor                : %s\n", (*hwinfo)->virtual_vendor_string);
        }
        cnt = PAPI_get_opt( PAPI_MAX_HWCTRS, NULL );
        mpx = PAPI_get_opt( PAPI_MAX_MPX_CTRS, NULL );

  int numcmp = PAPI_num_components(  );
  int perf_event = 0, cid;
        for ( cid = 0; cid < numcmp; cid++ ) {
          const PAPI_component_info_t* cmpinfo = PAPI_get_component_info( cid );
          if (cmpinfo->disabled) continue;
    if (strcmp(cmpinfo->name, "perf_event")== 0) perf_event = 1;
        }

        if ( cnt >= 0 ) {
                printf( "Number Hardware Counters : %d\n", cnt );
        } else if ( perf_event == 0 ) {
                printf( "Number Hardware Counters : NA\n" );
        } else {
                printf( "Number Hardware Counters : PAPI error %d: %s\n", cnt, PAPI_strerror(cnt));
        }
        if ( mpx >= 0 ) {
                printf( "Max Multiplex Counters   : %d\n", mpx );
        } else {
                printf( "Max Multiplex Counters   : PAPI error %d: %s\n", mpx, PAPI_strerror(mpx));
        }
  if (options.cmp_info != NULL) {
    printf("Fast counter read (rdpmc): %s\n",
        options.cmp_info->fast_counter_read ? "yes" : "no");
  }
        printf( "--------------------------------------------------------------------------------\n" );
        printf( "\n" );
        return PAPI_OK;
}

/**@class check_for_utility_components
 * @brief See if a component is a 'utility' component and if so
 *        return 1 signifying that it is.
 *
 *        Note: A 'utility' component is a component that does not have
 *        native events e.g. sysdetect.
 *
 * @param *component_name
 *   Name of component to check.
 */
int check_for_utility_components(const char *component_name)
{
    char *array_of_utility_cmps[PAPI_MIN_STR_LEN] = {"sysdetect", NULL};

    int entry;
    for (entry = 0; array_of_utility_cmps[entry] != NULL; entry++) {
        if (strcasecmp(array_of_utility_cmps[entry], component_name) == 0) {
            return 1;
        }   
    }   

    return 0;
}
