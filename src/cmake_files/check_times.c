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

long threadone = 0, threadtwo = 0;
pthread_t threadOne, threadTwo;
volatile int done = 0;

int gettid() {
    return syscall( SYS_gettid );
}

int doThreadOne( void * v ) {
    struct tms tm;
    int status;
    while (!done)
        sleep(1);
    status = times( & tm );
    assert( status != -1 );
    threadone = tm.tms_utime;
    return 0;
}

int doThreadTwo( void * v ) {
    struct tms tm;
    long i, j = 0xdeadbeef;
    int status;
    for( i = 0; i < 0xFFFFFFF; ++i ) { j = j ^ i; }
    status = times( & tm );
    assert( status != -1 );
    threadtwo = tm.tms_utime;
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
    return (threadone == threadtwo);
}
