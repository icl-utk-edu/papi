#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include "libperfctr.h"

static unsigned nrctrs; /* > 1 if we have PMCs in addition to a TSC */
extern void do_flops();

void do_init(struct vperfctr **handle)
{
    struct perfctr_info info;

    if( perfctr_info(&info) < 0 ) {
	perror("perfctr_info");
	exit(1);
    }
    nrctrs = perfctr_cpu_nrctrs();
    /* printf("\nPerfCtr Info:\n");
    printf("driver_version\t\t%u.%u", info.version_major, info.version_minor);
    if( info.version_micro )
	printf(".%u", info.version_micro);
    printf("\n");
    printf("nrcpus\t\t\t%u\n", info.nrcpus);
    printf("cpu_type\t\t%u (%s)\n", info.cpu_type, perfctr_cpu_name());
    printf("cpu_features\t\t0x%x\n", info.cpu_features);
    printf("cpu_khz\t\t\t%lu\n", info.cpu_khz);
    printf("nrctrs\t\t\t%u\n", nrctrs); */

    if((*handle = perfctr_attach_rdwr_self()) == NULL ) {
	perror("perfctr_attach_rdwr_self");
	exit(1);
    }
}

void do_read(struct vperfctr *handle)
{
    struct vperfctr_state state;

    if( perfctr_read_self(handle, &state) < 0 ) {
	 perror("perfctr_read_self");
	 exit(1);
    }
    printf("\nCurrent Sample:\n");
    printf("status\t\t\t%d\n", state.status);
    if( nrctrs > 1 )
      {
	printf("control.evntsel[0]\t0x%08X\n", state.control.evntsel[0]);
	printf("control.evntsel[1]\t0x%08X\n", state.control.evntsel[1]);
      }
    printf("tsc\t\t\t0x%016llX\n", state.sum.ctr[0]);
    if( nrctrs > 1 )
      {
	printf("pmc[0]\t\t\t0x%016llX %lld\n", state.sum.ctr[1], state.sum.ctr[1]);
	printf("pmc[1]\t\t\t0x%016llX %lld\n", state.sum.ctr[2], state.sum.ctr[2]);
      }
}

void do_enable(struct vperfctr *handle)
{
    struct perfctr_control control;

    memset(&control, 0, sizeof control);
    if( nrctrs > 1 )
      {
	control.evntsel[0] = 0x4700C1;
	control.evntsel[1] = 0x700C0;
      }
    if( perfctr_control_self(handle, &control) < 0 ) {
	perror("perfctr_control_self");
	exit(1);
    }
}

void do_close(struct vperfctr *handle)
{
  perfctr_close_self(handle);
}

void *Thread(void *arg)
{
  struct vperfctr *handle = NULL;
  printf("Thread %ld\n",pthread_self());
  do_init(&handle);
  do_read(handle);
  do_read(handle);
  do_enable(handle);
  do_flops(100000);
  do_read(handle);
  do_close(handle);
  return(NULL);
}

int main()
{
  pthread_t e_th;
  pthread_t f_th;
  int rc;
  double count1, count2, count3;

  count1 = 20.0;
  rc = pthread_create(&e_th, NULL, &Thread, &count1);
  if (rc)
    exit(-1);

  count2 = 15.0;
  rc = pthread_create(&f_th, NULL, &Thread, &count2);
  if (rc)
    exit(-1); 

  count3 = 10.0;
  Thread(&count3);

  pthread_join(f_th, NULL);
   pthread_join(e_th, NULL);
   pthread_exit(NULL); 

  exit(0);
}
