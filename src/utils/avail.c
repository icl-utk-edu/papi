/* This file performs the following test: hardware info and which events are available */

#include "papi_test.h"
int TESTS_QUIET=0; /* Tests in Verbose mode? */


int main(int argc, char **argv) 
{
  int i;
  int retval;
  const PAPI_preset_info_t *info = NULL;
  const PAPI_hw_info_t *hwinfo = NULL;
  char *tmp,buf[128];

  memset( buf, '\0', sizeof(buf) );
  if ( argc > 1 ) {
	if ( !strcmp( argv[1], "TESTS_QUIET" ) )
	   TESTS_QUIET=1;
  }
  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
	tmp = strdup("PAPI_library_init");
     	goto FAILED;
  }

  if ((retval=PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK){
	tmp = strdup("PAPI_set_debug");
     	goto FAILED;
  }

  if ((info = PAPI_query_all_events_verbose()) == NULL){
	retval = 1;
	tmp = strdup("PAPI_query_all_events_verbose");
	goto FAILED;
  }
  if ((hwinfo = PAPI_get_hardware_info()) == NULL){
	retval = 2;
	tmp = strdup("PAPI_get_hardware_info");
	goto FAILED;
  }

  if ( !TESTS_QUIET ) {
  printf("Test case 8: Available events and hardware information.\n");
  printf("-------------------------------------------------------------------------\n");
  printf("Vendor string and code   : %s (%d)\n",hwinfo->vendor_string,hwinfo->vendor);
  printf("Model string and code    : %s (%d)\n",hwinfo->model_string,hwinfo->model);
  printf("CPU revision             : %f\n",hwinfo->revision);
  printf("CPU Megahertz            : %f\n",hwinfo->mhz);
  printf("CPU's in an SMP node     : %d\n",hwinfo->ncpu);
  printf("Nodes in the system      : %d\n",hwinfo->nnodes);
  printf("Total CPU's in the system: %d\n",hwinfo->totalcpus);
  printf("-------------------------------------------------------------------------\n");
  printf("Name\t\tCode\t\tAvail\tDeriv\tDescription (Note)\n");
  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    if (info[i].event_name)
      printf("%s\t0x%x\t%s\t%s\t%s (%s)\n",
	     info[i].event_name,
	     info[i].event_code,
	     (info[i].avail ? "Yes" : "No"),
	     (info[i].flags & PAPI_DERIVED ? "Yes" : "No"),
	     info[i].event_descr,
	     (info[i].event_note ? info[i].event_note : ""));
  printf("-------------------------------------------------------------------------\n");
  }
  printf("avail:		PASSED\n");
  exit(0);
FAILED:
  printf("avail:		FAILED\n");
  if ( retval == PAPI_ESYS ) {
	sprintf(buf, "System error in %s:", tmp );
	perror(buf);
  }
  else if ( retval > 0 ) 
	printf("Error in %s, no further information available\n", tmp );
  else {
	char errstring[PAPI_MAX_STR_LEN];
	PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN );
	printf("Error in %s: %s\n", tmp, errstring );
  }
  free(tmp);
  exit(1);
}
