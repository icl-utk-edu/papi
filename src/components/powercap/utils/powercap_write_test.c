/** 
 * @author Philip Vaccar
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"

#define MAX_EVENTS 128

char events[MAX_EVENTS][BUFSIZ];
char filenames[MAX_EVENTS][BUFSIZ];

int ompcpuloadprimes( int limit ) 
{
    int num, primes=0;
#pragma omp parallel for schedule(dynamic) reduction(+ : primes)
    for (num = 1; num <= limit; num++) { 
        int i = 2; 
        while(i <= num) { 
            if(num % i == 0)
                break;
            i++; 
        }
        if(i == num)
            primes++;
    }
    return primes;
}


int main (int argc, char **argv)
{
    int retval,cid,rapl_cid=-1,numcmp;
    int EventSet = PAPI_NULL;
    long long values[MAX_EVENTS];
    int i,code,enum_retval;
    const PAPI_component_info_t *cmpinfo = NULL;
    long long start_time,write_start_time,write_end_time,read_start_time,read_end_time;
    char event_name[BUFSIZ];
    long long event_value_ll;
    static int num_events=0;
    FILE *fileout;
    
    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        fprintf(stderr,"PAPI_library_init failed\n");
	exit(1);
    }
    
    /* Find the powercap component */
    numcmp = PAPI_num_components();
    for(cid=0; cid<numcmp; cid++) {
	if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
            fprintf(stderr,"PAPI_get_component_info failed\n");
            exit(1);
	}
	if (strstr(cmpinfo->name,"powercap")) {
            rapl_cid=cid;
            printf("Found powercap component at cid %d\n", rapl_cid);
            if (cmpinfo->disabled) {
                fprintf(stderr,"No powercap events found: %s\n", cmpinfo->disabled_reason);
                exit(1);
            }
            break;
	}
    }

    /* Component not found */    
    if (cid==numcmp) {
        fprintf(stderr,"No powercap component found\n");
        exit(1);
    }
    
    /* Find events in the component */
    code = PAPI_NATIVE_MASK;
    enum_retval = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, cid );
    while ( enum_retval == PAPI_OK ) {
        retval = PAPI_event_code_to_name( code, event_name );
        if ( retval != PAPI_OK ) {
            printf("Error translating %#x\n",code);
            exit(1);
        }
        strncpy(events[num_events],event_name,BUFSIZ);
        sprintf(filenames[num_events],"results.%s",event_name);
        num_events++;
        if (num_events==MAX_EVENTS) {
            printf("Too many events! %d\n",num_events);
            exit(1);
        }
        enum_retval = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, cid );
    }
    if (num_events==0) {
        printf("Error!  No powercap events found!\n");
	exit(1);
    }

    /* Open output file */
    char fileoutname[]="powercap_write_test_output.tsv";
    fileout=fopen( fileoutname ,"w" );
    if ( fileout==NULL) { fprintf( stderr,"Could not open %s\n",fileoutname ); exit(1); }

    /* Create EventSet */
    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK) {
        fprintf(stderr,"Error creating eventset!\n");
    }
    
    for(i=0;i<num_events;i++) {
        retval = PAPI_add_named_event( EventSet, events[i] );
        if (retval != PAPI_OK) fprintf(stderr,"Error adding event %s\n",events[i]); 
    }

    start_time=PAPI_get_real_nsec();

    /* Grab the initial values for the events */
    retval = PAPI_start( EventSet);
    if (retval != PAPI_OK) { fprintf(stderr,"PAPI_start() failed\n"); exit(1); }
    /* Initial checking read */
    retval = PAPI_read( EventSet, values);
    if (retval != PAPI_OK) { fprintf(stderr,"PAPI_read() failed\n"); exit(1); }

    /* Write a header line */
    fprintf( fileout, "time \tevents\t");
    for(i=0; i<num_events; i++) 
        fprintf( fileout, "%s\t", events[i]/*+9*/ );
    fprintf( fileout, "\n" );

    /* Read the initial values */
    retval = PAPI_read( EventSet, values);
    if (retval != PAPI_OK) { fprintf(stderr,"PAPI_read() failed\n"); exit(1); }
    fprintf( fileout, "%8.3f\t",((double)(PAPI_get_real_nsec()-start_time))/1.0e9);
    fprintf( fileout, "INIT\t");

    for(i=0; i<num_events; i++) {
        event_value_ll = values[i];
        fprintf( fileout, "%lld\t", event_value_ll );
    }
    fprintf( fileout, "\n" );

    int rpt=0;
    long long powerLimitAIncr =1000000;
    long long powerLimitA     =135000000;
    long long powerLimitBIncr =1000000;
    long long powerLimitB     =162000000;
      
    printf("\n"); 
    while(rpt++<50) {
        if ( rpt % 10 == 0 )  {
            for (i=0; i<num_events; i++) {
                /* Get current power_limit value */
                event_value_ll = values[i];
                if ( strstr( events[i], "POWER_LIMIT_A" )) event_value_ll=powerLimitA;
                else if ( strstr( events[i], "POWER_LIMIT_B" )) event_value_ll=powerLimitB;
                else { event_value_ll=-1;}
                values[i]=event_value_ll;
            }
            printf("updating vals...\n"); /* For next time */
            powerLimitA -= powerLimitAIncr;
            powerLimitB -= powerLimitBIncr;

            write_start_time=PAPI_get_real_nsec();
            retval = PAPI_write( EventSet, values );
            write_end_time=PAPI_get_real_nsec();
            if (retval != PAPI_OK) { fprintf(stderr,"PAPI_write() failed\n"); exit(1); }
            
            fprintf( fileout, "%8.3f\t",((double)(PAPI_get_real_nsec()-start_time))/1.0e9);
            fprintf( fileout, "SET\t");
            for(i=0; i<num_events; i++) {
                event_value_ll = values[i];
//                if(event_value_ll != -1) {
                    fprintf( fileout, "%lld\t", event_value_ll );
                    //printf( "SET\t");
//                }
//                else
//                  fprintf( fileout, "\t" );
            }
            fprintf( fileout, "\n" );
        }
        
        /* DO SOME WORK TO USE ENERGY */
        //usleep(100000);
        ompcpuloadprimes( 100000 );
        
        /* Read and output the values */
        read_start_time=PAPI_get_real_nsec();
        retval = PAPI_read( EventSet, values );
        read_end_time=PAPI_get_real_nsec();
        if (retval != PAPI_OK) { fprintf(stderr,"PAPI_read() failed\n"); exit(1); }
         fprintf( fileout, "%8.3f\t",((double)(PAPI_get_real_nsec()-start_time))/1.0e9);
        fprintf( fileout, "READ\t");
        for(i=0; i<num_events; i++) {
            event_value_ll = values[i];
            fprintf( fileout, "%lld\t", event_value_ll );
        }
        fprintf( fileout, "\n" );
    }

    retval = PAPI_stop( EventSet, values);
    printf("\nPASSED\n");
    return 0;
}

