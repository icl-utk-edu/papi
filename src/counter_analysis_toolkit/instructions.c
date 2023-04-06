#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <papi.h>
#include "instr.h"

int sum_i32=0;
float sum_f32=0.0;
double sum_f64=0.0;

void test_int_add(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    int i32_00, i32_01, i32_02, i32_03, i32_04, i32_05, i32_06, i32_07, i32_08, i32_09;

    /* Initialize the variables with values that the compiler cannot guess. */
    i32_00 =  p/2;
    i32_01 = -p/3;
    i32_02 =  p/4;
    i32_03 = -p/5;
    i32_04 =  p/6;
    i32_05 = -p/7;
    i32_06 =  p/8;
    i32_07 = -p/9;
    i32_08 =  p/10;
    i32_09 = -p/11;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # INT_ADD_count: %lld (%.3lf)\n", N, ev_values[0], 50LL*N*M, ev_values[0]/(50.0*N*M));

    sum_i32 += i32_00 + i32_01 + i32_02 + i32_03 + i32_04 + i32_05 + i32_06 + i32_07 + i32_08 + i32_09;

clean_up:

    return;
}


////////////////////////////////////////////////////////////////////////////////
// f64 ADDITION

void test_f64_add(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  (double)p/1.02;
    f64_01 = -(double)p/1.03;
    f64_02 =  (double)p/1.04;
    f64_03 = -(double)p/1.05;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define FADD_BLOCK() {f64_01 += f64_00; f64_02 += f64_01; f64_03 += f64_02; f64_00 += f64_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            FADD_BLOCK();
            FADD_BLOCK();
            FADD_BLOCK();
            FADD_BLOCK();
            FADD_BLOCK();
            FADD_BLOCK();
            FADD_BLOCK();
            FADD_BLOCK();
            FADD_BLOCK();
            FADD_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld # FP_ADD_count: %lld (%.3lf)\n", N, ev_values[0], fp_op_count, (double)ev_values[0]/fp_op_count);

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03;

clean_up:

    return;
}


void test_f64_add_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/1.2;
    f64_01 = -p/1.3;
    f64_02 =  p/1.4;
    f64_03 = -p/1.5;
    f64_04 =  p/1.6;
    f64_05 = -p/1.7;
    f64_06 =  p/1.8;
    f64_07 = -p/1.9;
    f64_08 =  p/2.0;
    f64_09 = -p/2.1;
    f64_10 =  p/2.2;
    f64_11 = -p/2.3;

    f64_100 =  0.00100;
    f64_101 = -0.00101;
    f64_102 =  0.00102;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f64_100 *= 1.045;
        f64_101 *= 1.054;
        f64_102 *= 1.067;
    }

#define F64_ADDS(_X) {f64_00 += _X; f64_01 += _X; f64_02 += _X; f64_03 += _X; f64_04 += _X; f64_05 += _X; f64_06 += _X; f64_07 += _X; f64_08 += _X; f64_09 += _X; f64_10 += _X; f64_11 += _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_ADDS(f64_100); F64_ADDS(f64_101); F64_ADDS(f64_102);
            if( p < 4 ){
                F64_ADDS(f64_00); F64_ADDS(f64_01); F64_ADDS(f64_02);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_ADD_count_ILP12: %lld (%.3lf)\n", N, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// Vector double precision ADD

void test_f64_add_DVEC128(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+2
    double a[BUFFER_SIZE];
    double b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-2);
    for(int i=0; i<UB; i++){
        for(int j=2; j<BUFFER_SIZE; j++){
            a[j] = a[j-2] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_ADD_DVEC128_count: %lld (%.3lf)\n", N, ev_values[0], N*M/2LL, (double)ev_values[0]/(N*M/2.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += a[j];
    }

clean_up:

    return;
}

void test_f64_add_DVEC256(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+4
    double a[BUFFER_SIZE];
    double b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-4);
    for(int i=0; i<UB; i++){
        for(int j=4; j<BUFFER_SIZE; j++){
            a[j] = a[j-4] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_ADD_DVEC256_count: %lld (%.3lf)\n", N, ev_values[0], N*M/4LL, (double)ev_values[0]/(N*M/4.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += a[j];
    }

clean_up:

    return;
}

void test_f64_add_DVEC512(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+8
    double a[BUFFER_SIZE];
    double b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-8);
    for(int i=0; i<UB; i++){
        for(int j=8; j<BUFFER_SIZE; j++){
            a[j] = a[j-8] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_ADD_DVEC512_count: %lld (%.3lf)\n", N, ev_values[0], N*M/8LL, (double)ev_values[0]/(N*M/8.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += a[j];
    }

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// Vector single precision ADD

void test_f64_add_SVEC128(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+2
    float a[BUFFER_SIZE];
    float b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-2);
    for(int i=0; i<UB; i++){
        for(int j=2; j<BUFFER_SIZE; j++){
            a[j] = a[j-2] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_ADD_SVEC128_count: %lld (%.3lf)\n", N, ev_values[0], N*M/2LL, (double)ev_values[0]/(N*M/2.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += (float)a[j];
    }

clean_up:

    return;
}

void test_f64_add_SVEC256(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+4
    float a[BUFFER_SIZE];
    float b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-4);
    for(int i=0; i<UB; i++){
        for(int j=4; j<BUFFER_SIZE; j++){
            a[j] = a[j-4] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_ADD_SVEC256_count: %lld (%.3lf)\n", N, ev_values[0], N*M/4LL, (double)ev_values[0]/(N*M/4.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += (float)a[j];
    }

clean_up:

    return;
}

void test_f64_add_SVEC512(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+8
    float a[BUFFER_SIZE];
    float b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-8);
    for(int i=0; i<UB; i++){
        for(int j=8; j<BUFFER_SIZE; j++){
            a[j] = a[j-8] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_ADD_SVEC512_count: %lld (%.3lf)\n", N, ev_values[0], N*M/8LL, (double)ev_values[0]/(N*M/8.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += (float)a[j];
    }

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// f64 SUB

void test_f64_sub(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  (double)p/1.02;
    f64_01 = -(double)p/1.03;
    f64_02 =  (double)p/1.04;
    f64_03 = -(double)p/1.05;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define FSUB_BLOCK() {f64_01 -= f64_00; f64_02 -= f64_01; f64_03 -= f64_02; f64_00 -= f64_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            FSUB_BLOCK();
            FSUB_BLOCK();
            FSUB_BLOCK();
            FSUB_BLOCK();
            FSUB_BLOCK();
            FSUB_BLOCK();
            FSUB_BLOCK();
            FSUB_BLOCK();
            FSUB_BLOCK();
            FSUB_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld # FP_SUB_count: %lld (%.3lf)\n", N, ev_values[0], fp_op_count, (double)ev_values[0]/fp_op_count);

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03;

clean_up:

    return;
}


void test_f64_sub_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/1.2;
    f64_01 = -p/1.3;
    f64_02 =  p/1.4;
    f64_03 = -p/1.5;
    f64_04 =  p/1.6;
    f64_05 = -p/1.7;
    f64_06 =  p/1.8;
    f64_07 = -p/1.9;
    f64_08 =  p/2.0;
    f64_09 = -p/2.1;
    f64_10 =  p/2.2;
    f64_11 = -p/2.3;

    f64_100 =  0.00100;
    f64_101 = -0.00101;
    f64_102 =  0.00102;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f64_100 *= 1.045;
        f64_101 *= 1.054;
        f64_102 *= 1.067;
    }

#define F64_SUBS(_X) {f64_00 -= _X; f64_01 -= _X; f64_02 -= _X; f64_03 -= _X; f64_04 -= _X; f64_05 -= _X; f64_06 -= _X; f64_07 -= _X; f64_08 -= _X; f64_09 -= _X; f64_10 -= _X; f64_11 -= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_SUBS(f64_100); F64_SUBS(f64_101); F64_SUBS(f64_102);
            if( p < 4 ){
                F64_SUBS(f64_00); F64_SUBS(f64_01); F64_SUBS(f64_02);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_SUB_count_ILP12: %lld (%.3lf)\n", N, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// f64 MULTIPLICATION

void test_f64_mul(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  (double)p/1.02;
    f64_01 =  1.03/(double)p;
    f64_02 =  (double)p/1.04;
    f64_03 =  1.05/(double)p;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define FMUL_BLOCK() {f64_01 *= f64_00; f64_02 *= f64_01; f64_03 *= f64_02; f64_00 *= f64_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            FMUL_BLOCK();
            FMUL_BLOCK();
            FMUL_BLOCK();
            FMUL_BLOCK();
            FMUL_BLOCK();
            FMUL_BLOCK();
            FMUL_BLOCK();
            FMUL_BLOCK();
            FMUL_BLOCK();
            FMUL_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld # FP_MUL_count: %lld (%.3lf)\n", N, ev_values[0], fp_op_count, (double)ev_values[0]/fp_op_count);

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03;

clean_up:

    return;
}


void test_f64_mul_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/431.2;
    f64_01 = -p/431.3;
    f64_02 =  p/431.4;
    f64_03 = -p/431.5;
    f64_04 =  p/431.6;
    f64_05 = -p/431.7;
    f64_06 =  p/431.8;
    f64_07 = -p/431.9;
    f64_08 =  p/432.0;
    f64_09 = -p/432.1;
    f64_10 =  p/432.2;
    f64_11 = -p/432.3;

    f64_100 =  1.00001;
    f64_101 = -1.00002;
    f64_102 =  1.00003;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f64_100 *= 1.0045;
        f64_101 *= 1.0054;
        f64_102 *= 1.0067;
    }

#define F64_MULS(_X) {f64_00 *= _X; f64_01 *= _X; f64_02 *= _X; f64_03 *= _X; f64_04 *= _X; f64_05 *= _X; f64_06 *= _X; f64_07 *= _X; f64_08 *= _X; f64_09 *= _X; f64_10 *= _X; f64_11 *= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_MULS(f64_100); F64_MULS(f64_101); F64_MULS(f64_102);
            if( p < 4 ){
                F64_MULS(f64_00); F64_MULS(f64_01); F64_MULS(f64_02);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_MUL_count_ILP12: %lld (%.3lf)\n", N, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// f64 DIVISION

void test_f64_div(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  1.0 + 1.0/(1000.1*(double)p);
    f64_01 =  1.0 + 1.0/(1000.2*(double)p);
    f64_02 =  1.0 + 1.0/(1000.3*(double)p);
    f64_03 =  1.0 + 1.0/(1000.4*(double)p);

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define FDIV_BLOCK() {f64_01 /= f64_00; f64_02 /= f64_01; f64_03 /= f64_02; f64_00 /= f64_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            FDIV_BLOCK();
            FDIV_BLOCK();
            FDIV_BLOCK();
            FDIV_BLOCK();
            FDIV_BLOCK();
            FDIV_BLOCK();
            FDIV_BLOCK();
            FDIV_BLOCK();
            FDIV_BLOCK();
            FDIV_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld # FP_DIV_count: %lld (%.3lf)\n", N, ev_values[0], fp_op_count, (double)ev_values[0]/fp_op_count);

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03;

clean_up:

    return;
}


void test_f64_div_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/431.2;
    f64_01 = -p/431.3;
    f64_02 =  p/431.4;
    f64_03 = -p/431.5;
    f64_04 =  p/431.6;
    f64_05 = -p/431.7;
    f64_06 =  p/431.8;
    f64_07 = -p/431.9;
    f64_08 =  p/432.0;
    f64_09 = -p/432.1;
    f64_10 =  p/432.2;
    f64_11 = -p/432.3;

    f64_100 =  1.00001;
    f64_101 = -1.00002;
    f64_102 =  1.00003;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f64_100 *= 1.0045;
        f64_101 *= 1.0054;
        f64_102 *= 1.0067;
    }

#define F64_DIVS(_X) {f64_00 /= _X; f64_01 /= _X; f64_02 /= _X; f64_03 /= _X; f64_04 /= _X; f64_05 /= _X; f64_06 /= _X; f64_07 /= _X; f64_08 /= _X; f64_09 /= _X; f64_10 /= _X; f64_11 /= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_DIVS(f64_100); F64_DIVS(f64_101); F64_DIVS(f64_102);
            if( p < 4 ){
                F64_DIVS(f64_00); F64_DIVS(f64_01); F64_DIVS(f64_02);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # FP_DIV_count_ILP12: %lld (%.3lf)\n", N, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// MEM ops

void test_mem_ops_serial_RO(int p, int M, int N, int EventSet, FILE *fp){
    int i, ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 256
    int buffer[BUFFER_SIZE];

    /* Initialize the buffer with values that the compiler cannot guess. */
    for(i=0; i<BUFFER_SIZE; i++){
        buffer[i] = p/(1223+i);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    uintptr_t index = buffer[0];
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            index = buffer[index%BUFFER_SIZE];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # MEM_OPS_RO_count: %lld (%.3lf)\n", N, ev_values[0], 1LL*N*M, ev_values[0]/(1.0*N*M));

    sum_i32 += index;

clean_up:

    return;
}

void test_mem_ops_serial_RW(int p, int M, int N, int EventSet, FILE *fp){
    int i, ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE (256+1)
    int buffer[BUFFER_SIZE];

    /* Initialize the buffer with values that the compiler cannot guess. */
    for(i=0; i<BUFFER_SIZE; i++){
        buffer[i] = p/(1223+i);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            uintptr_t index = j%(BUFFER_SIZE-1);
            buffer[index+1] += buffer[index];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # MEM_OPS_RW_count: %lld (%.3lf)\n", N, ev_values[0], 2LL*N*M, ev_values[0]/(2.0*N*M));

    sum_i32 += buffer[0] + buffer[BUFFER_SIZE/2] + buffer[BUFFER_SIZE-1];

clean_up:

    return;
}

void test_mem_ops_parallel_RO(int p, int M, int N, int EventSet, FILE *fp){
    int i, ret;
    long long int ev_values[2];
    int c0, c1, c2, c3;
#undef BUFFER_SIZE
#define BUFFER_SIZE (256+8)
    int buffer[BUFFER_SIZE];

    /* Initialize the buffer with values that the compiler cannot guess. */
    for(i=0; i<BUFFER_SIZE; i++){
        buffer[i] = p/(223+i);
    }

    /* Initialize the variables with values that the compiler cannot guess. */
    c0 = (int)((5+p)/(12+1));
    c1 = (int)((7+p)/(12+2));
    c2 = (int)((11+p)/(12+3));
    c3 = (int)((13+p)/(12+4));

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    for(int i=0; i<M; i++){
        // compute some junk value.
        uintptr_t base = i*(c0+c1)/(c2+c3+1);
        for(int j=0; j<N; j++){
            uintptr_t offset = (base+j)%(BUFFER_SIZE-8);

            c0 += buffer[offset+1];
            c1 += buffer[offset+2];
            c2 += buffer[offset+3];
            c3 += buffer[offset+4];

            c0 += buffer[offset+5];
            c1 += buffer[offset+6];
            c2 += buffer[offset+7];
            c3 += buffer[offset+8];

        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # MEM_OPS_RO_count(par): %lld (%.3lf)\n", N, ev_values[0], 8LL*N*M, ev_values[0]/(8.0*N*M));

    sum_i32 += c0+c1+c2+c3;

clean_up:

    return;
}

void test_mem_ops_parallel_WO(int p, int M, int N, int EventSet, FILE *fp){
    int i, ret;
    long long int ev_values[2];
    int sum=0;
#undef BUFFER_SIZE
#define BUFFER_SIZE 256
    int buffer[BUFFER_SIZE];

    /* Initialize the buffer with values that the compiler cannot guess. */
    for(i=0; i<BUFFER_SIZE; i++){
        buffer[i] = p/(1223+i);
        sum += buffer[i];
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            buffer[j%BUFFER_SIZE] = sum;
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld # MEM_OPS_WO_count(par): %lld (%.3lf)\n", N, ev_values[0], 1LL*N*M, ev_values[0]/(1.0*N*M));

    sum_i32 += buffer[0] + buffer[BUFFER_SIZE/2] + buffer[BUFFER_SIZE-1];

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
//////// Main driver

void instr_test(int EventSet, FILE *fp) {
    int i, j, M, N;
    int minM=64, minN=64;
    double f[4] = {1.0, 1.1892, 1.4142, 1.6818};
    int p = (int)getpid();

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_mem_ops_serial_RO(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_mem_ops_serial_RW(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((9+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_mem_ops_parallel_RO(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_mem_ops_parallel_WO(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((50.0+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_int_add(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((40+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_add(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((12.0*3+5)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_add_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((40+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_sub(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((12.0*3+5)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_sub_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((40+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_mul(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((12.0*3+5)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_mul_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((40+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM/4);
            N = (int)(i*f[j]*minN/4);
            test_f64_div(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((12.0*3+5)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM/4);
            N = (int)(i*f[j]*minN/4);
            test_f64_div_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_SVEC128(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_SVEC256(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_SVEC512(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_DVEC128(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_DVEC256(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp, "# (((2+3)*N)+3)*M\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_DVEC512(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    if( sum_i32 == 12345 && sum_f64 == 12.345)
        fprintf(stderr, "Side-effect to disable dead code elimination by the compiler. Please ignore.\n");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void instr_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir)
{
    int retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".instr";
    char *papiFileName;

    (void)hw_desc;

    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);
    if (NULL == (papiFileName = (char *)calloc( 1+l, sizeof(char)))) {
        return;
    }
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        goto error0;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Failed to open file %s.\n", papiFileName);
        goto error0;
    }
  
    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

    retval = PAPI_add_named_event( EventSet, papi_event_name );
    if (retval != PAPI_OK ){
        goto error1;
    }

    retval = PAPI_OK;

    instr_test(EventSet, ofp_papi);

    retval = PAPI_cleanup_eventset( EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }
    retval = PAPI_destroy_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

error1:
    fclose(ofp_papi);
error0:
    free(papiFileName);
    return;
}
