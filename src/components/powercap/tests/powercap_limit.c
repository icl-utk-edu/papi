/**
 * @author Philip Vaccaro
 * Test case for powercap component
 * @brief
 *   Tests basic functionality of powercap component
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#define MAX_powercap_EVENTS 64

int main ( int argc, char **argv )
{
  (void) argv;
  (void) argc;
  int retval,cid,powercap_cid=-1,numcmp;
  int EventSet = PAPI_NULL;
  long long values[MAX_powercap_EVENTS];
  int num_limits=0;
  int code;
  char power_limit_eventnames[MAX_powercap_EVENTS][PAPI_MAX_STR_LEN];
  int r,i;

  const PAPI_component_info_t *cmpinfo = NULL;

  /* PAPI Initialization */
  retval = PAPI_library_init( PAPI_VER_CURRENT );
  if ( retval != PAPI_VER_CURRENT )
    test_fail( __FILE__, __LINE__,"PAPI_library_init()\n",retval );

  if ( !TESTS_QUIET ) printf( "Trying all powercap events\n" );

  numcmp = PAPI_num_components();

  for( cid=0; cid<numcmp; cid++ ) {

    if ( ( cmpinfo = PAPI_get_component_info( cid ) ) == NULL )
      test_fail( __FILE__, __LINE__,"PAPI_get_component_info()\n", 0 );

    if ( strstr( cmpinfo->name,"powercap" ) ) {
      powercap_cid=cid;
      if ( !TESTS_QUIET ) printf( "Found powercap component at cid %d\n",powercap_cid );
      if ( cmpinfo->disabled ) {
        if ( !TESTS_QUIET ) {
          printf( "powercap component disabled: %s\n",
                  cmpinfo->disabled_reason );
        }
        test_skip( __FILE__,__LINE__,"powercap component disabled",0 );
      }
      break;
    }
  }

  /* Component not found */
  if ( cid==numcmp )
    test_skip( __FILE__,__LINE__,"No powercap component found\n",0 );

  /* Skip if component has no counters */
  if ( cmpinfo->num_cntrs==0 )
    test_skip( __FILE__,__LINE__,"No counters in the powercap component\n",0 );

  /* Create EventSet */
  retval = PAPI_create_eventset( &EventSet );
  if ( retval != PAPI_OK )
    test_fail( __FILE__, __LINE__, "PAPI_create_eventset()",retval );

  /* Add all package limit events */
  code = PAPI_NATIVE_MASK;
  r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, powercap_cid );

  /* Find all package power limit events */
  while (r == PAPI_OK) {
    char powercap_eventname[PAPI_MAX_STR_LEN] = { 0 };
    retval = PAPI_event_code_to_name( code, powercap_eventname );
    if ( retval != PAPI_OK ) 
      test_fail( __FILE__, __LINE__,"PAPI_event_code_to_name()", retval );

    if (!(strstr(powercap_eventname, "SUBZONE")) && (strstr(powercap_eventname, "POWER_LIMIT"))) {
      retval = PAPI_add_named_event(EventSet, powercap_eventname);
      if (retval != PAPI_OK)
        break; /* We've hit an event limit */

      int strLen = snprintf(power_limit_eventnames[num_limits], sizeof(power_limit_eventnames[num_limits]), "%s", powercap_eventname);
      if (strLen < 0 || (size_t) strLen >= sizeof(power_limit_eventnames[num_limits])) {
          fprintf(stderr, "Failed to fully write event name %s into buffer at index %d.\n", powercap_eventname, num_limits);
          exit(EXIT_FAILURE);
      }
      num_limits++;
    }

    r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, powercap_cid );
  }

  if (num_limits == 0) {
    fprintf(stderr, "No power limit events detected.\n");
    test_skip( __FILE__, __LINE__, "", 0 );
  }

  /* start collecting power data */
  retval = PAPI_start( EventSet );
  if ( retval != PAPI_OK )
      test_fail( __FILE__, __LINE__, "PAPI_start()",retval );

  /* initial read of package limits */
  retval = PAPI_read( EventSet, values );
  if ( retval != PAPI_OK )
      test_fail( __FILE__, __LINE__, "PAPI_read()",retval );

  printf("\nCURRENT LIMITS\n");
  for( i=0; i<num_limits; i++ ) {
      printf("EVENT: %s\tLIMIT: %0.2lf Watts\n", power_limit_eventnames[i], ((double)values[i]*1e-6));
      values[i] = values[i] - (10 * 1e6); //minus 10 Watts
  }
  usleep(10000);

  printf("\nSETTING LIMITS 10 WATTS BELOW CURRENT LIMITS\n");
  retval = PAPI_write( EventSet, values );
  if ( retval != PAPI_OK )
      test_fail( __FILE__, __LINE__, "PAPI_write()",retval );

  usleep(10000);

  printf("\nREADING LIMITS TO MAKE SURE THEY ARE SET\n");
  retval = PAPI_read( EventSet, values );
  if ( retval != PAPI_OK )
      test_fail( __FILE__, __LINE__, "PAPI_read()",retval );
  usleep(10000);

  printf("\nNEW LIMITS\n");
  for( i=0; i<num_limits; i++ ) {
      printf("EVENT: %s\tLIMIT: %0.2lf Watts\n", power_limit_eventnames[i], ((double)values[i]*1e-6));
      values[i] = values[i] + (10 * 1e6); //plus 10 Watts
  }

  printf("\nRESET LIMITS BEFORE EXITING...");
  retval = PAPI_write( EventSet, values );
  if ( retval != PAPI_OK )
      test_fail( __FILE__, __LINE__, "PAPI_write()",retval );
  usleep(10000);

  printf("\nREADING RESET LIMITS TO MAKE SURE THEY ARE SET\n");
  retval = PAPI_read( EventSet, values );
  if ( retval != PAPI_OK )
      test_fail( __FILE__, __LINE__, "PAPI_read()",retval );
  usleep(10000);

  printf("\nRESET LIMITS\n");
  for( i=0; i<num_limits; i++ ) {
      printf("EVENT: %s\tLIMIT: %0.2lf Watts\n", power_limit_eventnames[i], ((double)values[i]*1e-6));
  }

  printf("done\n");

  retval = PAPI_stop( EventSet, values );
  if ( retval != PAPI_OK )
      test_fail( __FILE__, __LINE__, "PAPI_stop()",retval );

  /* Done, clean up */
  retval = PAPI_cleanup_eventset( EventSet );
  if ( retval != PAPI_OK )
    test_fail( __FILE__, __LINE__,"PAPI_cleanup_eventset()",retval );

  retval = PAPI_destroy_eventset( &EventSet );
  if ( retval != PAPI_OK )
    test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset()",retval );

  test_pass( __FILE__ );

  return 0;
}

