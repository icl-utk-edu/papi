#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "papi.h"

#define NUM_EVENTS 4
const char* names[NUM_EVENTS] = {"READ.CALLS", "READ.BYTES","WRITE.CALLS","WRITE.BYTES"};

#define NUM_INFILES 4
static const char* files[NUM_INFILES] = {"/etc/passwd", "/etc/group", "/etc/protocols", "/etc/nsswitch.conf"};

void *ThreadIO(void *arg) {
  unsigned long tid = (unsigned long)pthread_self();
  printf("\nThread 0x%lx: will read %s\n", tid,(const char*) arg);
  int Events[NUM_EVENTS]; 
  long long values[NUM_EVENTS];
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

  int fdin = open((const char*)arg, O_RDONLY);
  if (fdin < 0) perror("Could not open file for reading: \n");

  int bytes = 0;
  char buf[1024];

  int fdout = open("/dev/null", O_WRONLY);
  if (fdout < 0) perror("Could not open /dev/null for writing: \n");
  while ((bytes = read(fdin, buf, 1024)) > 0) {
    write(fdout, buf, bytes);
  }
  close(fdout);

  /* Stop counting events */
  if (PAPI_stop_counters(values, NUM_EVENTS) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_stop_counters\n");
  }

  for (e=0; e<NUM_EVENTS; e++)  
    printf("Thread 0x%lx: %s: %lld\n", tid, names[e], values[e]);
  return(NULL);
}

int main(void) {
  pthread_t *callThd;
  int i, numthrds;
  int retval;
  pthread_attr_t attr;

  int version = PAPI_library_init (PAPI_VER_CURRENT);
  if (version != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI_library_init version mismatch\n");
    exit(1);
  }


  pthread_attr_init(&attr);
  if (PAPI_thread_init(pthread_self) != PAPI_OK) {
    fprintf(stderr, "PAPI_thread_init returned an error\n");
    exit(1);
  }
#ifdef PTHREAD_CREATE_UNDETACHED
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
  retval = pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
  if (retval != 0) {
    fprintf(stderr,"This system does not support kernel scheduled pthreads.\n");
    exit(1);
  }
#endif

  numthrds = NUM_INFILES;
  if (getenv("NUM_THREADS")) {
    numthrds = atoi(getenv("NUM_THREADS"));
  }
  printf("%d threads\n",numthrds);
  if (numthrds > NUM_INFILES) {
    fprintf(stderr, "This test can only test a maximum of %d threads. Setting num threads to %d\n", NUM_INFILES, NUM_INFILES);
    numthrds = NUM_INFILES;
  }
  callThd = (pthread_t *)malloc(numthrds*sizeof(pthread_t));

  for (i=0;i<(numthrds-1);i++) {
    pthread_create(callThd+i, &attr, ThreadIO, (void *) files[i]);
  }
  ThreadIO((void *)files[numthrds-1]);
  pthread_attr_destroy(&attr);

  for (i=0;i<(numthrds-1);i++)
    pthread_join(callThd[i], NULL);

  exit(0);
}
