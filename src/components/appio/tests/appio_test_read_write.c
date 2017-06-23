/* 
 * Test case for appio
 * Author: Tushar Mohan
 *         tusharmohan@gmail.com
 * 
 * Description: This test case reads from standard linux /etc/group
 *              and writes the output to  stdout.
 *              Statistics are printed at the end of the run.,
 */
#include <papi.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"
 
#define NUM_EVENTS 12
 
int main(int argc, char** argv) {
  int Events[NUM_EVENTS]; 
  const char* names[NUM_EVENTS] = {"OPEN_CALLS", "OPEN_FDS", "READ_CALLS", "READ_BYTES", "READ_USEC", "READ_ERR", "READ_INTERRUPTED", "READ_WOULD_BLOCK", "WRITE_CALLS","WRITE_BYTES","WRITE_USEC","WRITE_WOULD_BLOCK"};
  long long values[NUM_EVENTS];

  char *infile = "/etc/group";

  /* Set TESTS_QUIET variable */
  tests_quiet( argc, argv );

  int version = PAPI_library_init (PAPI_VER_CURRENT);
  if (version != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI_library_init version mismatch\n");
    exit(1);
  }

  int fdin;
  if (!TESTS_QUIET) printf("This program will read %s and write it to /dev/null\n", infile);
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

  fdin=open(infile, O_RDONLY);
  if (fdin < 0) perror("Could not open file for reading: \n");
  int fdout;
  fdout=open("/dev/null", O_WRONLY);
  if (fdout < 0) perror("Could not open file for writing: \n");
  int bytes = 0;
  char buf[1024];

 
//if (PAPI_read_counters(values, NUM_EVENTS) != PAPI_OK)
//   handle_error(1);
//printf("After reading the counters: %lld\n",values[0]);

  while ((bytes = read(fdin, buf, 1024)) > 0) {
    write(fdout, buf, bytes);
  }

  /* Closing the descriptors before doing the PAPI_stop
     means, OPEN_FDS will be reported as zero, which is
     right, since at the time of PAPI_stop, the descriptors
     we opened have been closed */
  close (fdin);
  close (fdout);

  /* Stop counting events */
  if (PAPI_stop_counters(values, NUM_EVENTS) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_stop_counters\n");
  }

  if (!TESTS_QUIET) {
    printf("----\n");
    for (e=0; e<NUM_EVENTS; e++)
      printf("%s: %lld\n", names[e], values[e]);
  }
  test_pass( __FILE__ );
  return 0;
}
