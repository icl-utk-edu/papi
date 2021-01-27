#include <pthread.h>
#include <sys/signal.h>
#include <sys/times.h>
#include <assert.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/unistd.h>
#include <syscall.h>
#include <stdlib.h>

#if !defined( SYS_gettid )
#define SYS_gettid 1105
#endif

struct timespec threadone = { 0, 0 };
struct timespec threadtwo = { 0, 0 };
pthread_t threadOne, threadTwo;
volatile int done = 0;

int gettid() {
    return syscall( SYS_gettid );
}

void *doThreadOne( void * v ) {
    while (!done)
    sleep(1);
    if (syscall(__NR_clock_gettime,CLOCK_THREAD_CPUTIME_ID,&threadone) == -1) {
        perror("clock_gettime(CLOCK_THREAD_CPUTIME_ID)");
        exit(1);
    }
    return 0;
}

void *doThreadTwo( void * v ) {
    long i, j = 0xdeadbeef;
    for( i = 0; i < 0xFFFFFFF; ++i ) { j = j ^ i; }

    if (syscall(__NR_clock_gettime,CLOCK_THREAD_CPUTIME_ID,&threadtwo) == -1) {
        perror("clock_gettime(CLOCK_THREAD_CPUTIME_ID)");
        exit(1);
    }
    done = 1;
    return j;
} 

int main( int argc, char ** argv ) {
    int status = pthread_create( & threadOne, NULL, doThreadOne, NULL );
    assert( status == 0 );
    status = pthread_create( & threadTwo, NULL, doThreadTwo, NULL );
    assert( status == 0 );  
    status = pthread_join( threadTwo, NULL );
    assert( status == 0 );
    status = pthread_join( threadOne, NULL );
    assert( status == 0 );
    if ((threadone.tv_sec != threadtwo.tv_sec) || (threadone.tv_nsec != threadtwo.tv_nsec))
        exit(0);
    else {	
        fprintf(stderr,"T1 %ld %ld T2 %ld %ld\n",threadone.tv_sec,threadone.tv_nsec,threadtwo.tv_sec,threadtwo.tv_nsec);
        exit(1);
    }
}
