#include <papi.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

 
#define NUM_EVENTS 4
 
int main(int argc, char** argv) {
  int Events[NUM_EVENTS]; 
  const char* names[NUM_EVENTS] = {"READ.CALLS", "READ.BYTES","WRITE.CALLS","WRITE.BYTES"};
  long long values[NUM_EVENTS];

  char *infile = "/etc/group";

  int version = PAPI_library_init (PAPI_VER_CURRENT);
  if (version != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI_library_init version mismatch\n");
    exit(1);
  }

  int fdin;
  fprintf(stderr, "This program will read %s and write it to stdout\n", infile);
  fdin=open(infile, O_RDONLY);
  if (fdin < 0) perror("Could not open file for reading: \n");
  int bytes = 0;
  char buf[1024];

  int retval;
  int e;
  for (e=0; e<NUM_EVENTS; e++) {
    retval = PAPI_event_name_to_code((char*)names[e], &Events[e]);
    if (retval != PAPI_OK) {
      fprintf(stderr, "Error getting code for %s\n", names[e]);
      exit(2);
    } 
  }

  /* Start counting events */
  if (PAPI_start_counters(Events, NUM_EVENTS) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_start_counters\n");
    exit(1);
  }
 
//if (PAPI_read_counters(values, NUM_EVENTS) != PAPI_OK)
//   handle_error(1);
//printf("After reading the counters: %lld\n",values[0]);

  while ((bytes = read(fdin, buf, 1024)) > 0) {
    write(1, buf, bytes);
  }


  /* Stop counting events */
  if (PAPI_stop_counters(values, NUM_EVENTS) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_stop_counters\n");
  }
  
  printf("----\n");
  for (e=0; e<NUM_EVENTS; e++)  
    printf("%s: %lld\n", names[e], values[e]);
  return 0;
}
