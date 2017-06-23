#include <stdio.h>
#include <string.h>
#include <stdlib.h> /* exit() */
#include <errno.h> /* herror() */
#include <netdb.h> /* gethostbyname() */
#include <sys/types.h> /* bind() accept() */
#include <sys/socket.h> /* bind() accept() */
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#define PORT 3490
#define NUM_EVENTS 6

main(int argc, char *argv[]) {
  int Events[NUM_EVENTS]; 
  const char* names[NUM_EVENTS] = {"RECV_CALLS", "RECV_BYTES", "RECV_USEC", "RECV_ERR", "RECV_INTERRUPTED", "RECV_WOULD_BLOCK"};
  long long values[NUM_EVENTS];

  /* Set TESTS_QUIET variable */
  tests_quiet( argc, argv );

  int version = PAPI_library_init (PAPI_VER_CURRENT);
  if (version != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI_library_init version mismatch\n");
    exit(1);
  }

  if (!TESTS_QUIET) printf("This program will listen on port 3490, and write data received to standard output\n");
  int retval;
  int e;
  for (e=0; e<NUM_EVENTS; e++) {
    retval = PAPI_event_name_to_code((char*)names[e], &Events[e]);
    if (retval != PAPI_OK) {
      fprintf(stderr, "Error getting code for %s\n", names[e]);
      exit(2);
    } 
  }

  int bytes = 0;
  char buf[1024];

  int sockfd, n_sockfd, sin_size, len;
  char *host_addr, *recv_msg;
  struct sockaddr_in my_addr;
  struct sockaddr_in their_addr;
  my_addr.sin_family = AF_INET;
  my_addr.sin_port = htons(PORT);
  my_addr.sin_addr.s_addr = INADDR_ANY;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    perror("socket");
    exit(1);
  }
  if ((bind(sockfd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr))) == -1) {
    perror("bind");
    exit(1);
  }
  listen(sockfd, 10);
  if ((n_sockfd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size)) == -1) {
    perror("accept");
    exit(1);
  }
  close(sockfd);

  /* Start counting events */
  if (PAPI_start_counters(Events, NUM_EVENTS) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_start_counters\n");
    exit(1);
  }

  while ((bytes = recv(n_sockfd, buf, 1024, 0)) > 0) {
    write(1, buf, bytes);
  }

  close(n_sockfd);

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
