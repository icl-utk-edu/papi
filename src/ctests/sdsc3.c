/*
 * $Id$
 *
 * Test example for multiplex functionality, originally 
 * provided by Timothy Kaiser, SDSC. It was modified to fit the 
 * PAPI test suite by Nils Smeds, <smeds@pdc.kth.se>.
 *
 * This example verifies the that sharing base events between 
 * multiplexing event sets works.
 */

#include "papi_test.h"
#include <stdio.h>
#include <math.h>

#define REPEATS 5
#define MAXEVENTS 9
#define MINCOUNTS 100000
#define RELTOLERANCE 0.08
#define SLEEPTIME 100

static double dummy3(double x,int iters);

int main(int argc, char **argv) {
  char des[128];
  int i, j, retval;
  int iters=10000000;
  double x,y,dtmp;
  long_long t1,t2;  
  long_long values[2*MAXEVENTS],refvals[MAXEVENTS];
#ifdef STARTSTOP
  long_long dummies[MAXEVENTS];
#endif
  long_long *val2;
  int sleep_time = SLEEPTIME; 
  double valsqsum[2*MAXEVENTS];
  double valsum[2*MAXEVENTS];
  double spread[2*MAXEVENTS];
  double *val2sum,*val2sqsum,*spread2;
  int nevents=MAXEVENTS,nev1,nev2;
  int eventset=PAPI_NULL,eset1=PAPI_NULL,eset2=PAPI_NULL;
  int events[MAXEVENTS];

  val2=&values[MAXEVENTS];
  val2sum=&valsum[MAXEVENTS];
  val2sqsum=&valsqsum[MAXEVENTS];
  spread2=&spread[MAXEVENTS];

  events[0]=PAPI_FP_INS;
  events[1]=PAPI_TOT_INS;
  events[2]=PAPI_INT_INS;
  events[3]=PAPI_TOT_CYC;
  events[4]=PAPI_STL_CCY;
  events[5]=PAPI_BR_INS;
  events[6]=PAPI_SR_INS;
  events[7]=PAPI_LD_INS;
  events[8]=PAPI_TOT_IIS;

  for(i=0;i<2*MAXEVENTS;i++) {
    values[i]=0.;
    valsqsum[i]=0;
    valsum[i]=0;
  }

  if ( argc > 1 ){
    if ( !strcmp( argv[1], "TESTS_QUIET") )
      tests_quiet(argc, argv);
    else {
      sleep_time = atoi(argv[1]);
      if ( sleep_time <= 0 )
	sleep_time = SLEEPTIME;
    }
  }

  if ( !TESTS_QUIET ) {
    printf("\nAccuracy check of multiplexing routines.\n");
    printf("Multiple multiplexing eventsets with some shared event(s).\n\n");
  }

  if((retval = PAPI_library_init(PAPI_VER_CURRENT))!=PAPI_VER_CURRENT)
    test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  if((retval=PAPI_create_eventset(&eventset)))
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

#ifdef MPX
  if((retval=PAPI_multiplex_init()))
    test_fail(__FILE__,__LINE__,"PAPI_multiplex_init",retval);

  if((retval=PAPI_set_multiplex(&eventset)))
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);
#endif

  nevents=MAXEVENTS;
  for(i=0;i<nevents;i++) {
    if((retval=PAPI_add_event(&eventset, events[i]))) {
      for(j=i;j<MAXEVENTS;j++)
	events[j]=events[j+1];
      nevents--;
      i--;
    }
  }
  if(nevents<3)
    test_skip(__FILE__,__LINE__,"Not enough events left...",0);

  /* Find a reasonable number of iterations (each 
   * event active 20 times) during the measurement
   */
  t2=10000*20*nevents; /* Target: 10000 usec/multiplex, 20 repeats */
  if(t2>30e6)
    test_skip(__FILE__,__LINE__,"This test takes too much time",retval);

  /* Measure one run */
  t1=PAPI_get_real_usec();
  y=dummy3(x,iters);  
  t1=PAPI_get_real_usec()-t1;

  if(t2>t1) /* Scale up execution time to match t2 */
    iters=iters*t2/t1;
  else if(t1>30e6) /* Make sure execution time is < 30s per repeated test */
    test_skip(__FILE__,__LINE__,"This test takes too much time",retval);

  
  if((retval=PAPI_create_eventset(&eset1)))
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);
#ifdef MPX
  if((retval=PAPI_set_multiplex(&eset1)))
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);
#endif

  nev1=nevents/2+1;
  for(i=0;i<nev1;i++) {
    if((retval=PAPI_add_event(&eset1, events[i]))) 
      test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
  }
  
  if((retval=PAPI_create_eventset(&eset2)))
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);
#ifdef MPX
  if((retval=PAPI_set_multiplex(&eset2)))
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);
#endif

  nev2=nevents/2+1;
  for(i=0;i<nev2;i++) {
    if((retval=PAPI_add_event(&eset2, events[nevents-i-1]))) 
      test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
  }
  

  if((retval=PAPI_start(eset1)))
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);
  if((retval=PAPI_start(eset2)))
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for(i=1;i<=REPEATS;i++) {
    x=1.0;

#ifndef STARTSTOP
    if((retval=PAPI_reset(eset1)))
      test_fail(__FILE__,__LINE__,"PAPI_reset",retval);
    if((retval=PAPI_reset(eset2)))
      test_fail(__FILE__,__LINE__,"PAPI_reset",retval);
#else
    if((retval=PAPI_stop(eset1,dummies)))
      test_fail(__FILE__,__LINE__,"PAPI_stop",retval);
    if((retval=PAPI_stop(eset2,dummies)))
      test_fail(__FILE__,__LINE__,"PAPI_stop",retval);
    if((retval=PAPI_start(eset2)))
      test_fail(__FILE__,__LINE__,"PAPI_start",retval);
    if((retval=PAPI_start(eset1)))
      test_fail(__FILE__,__LINE__,"PAPI_start",retval);
#endif

    if ( !TESTS_QUIET )
      printf("\nTest %d (of %d):\n",i,REPEATS);
    t1=PAPI_get_real_usec();
    y=dummy3(x,iters);  
    PAPI_read(eset1,values);
    PAPI_read(eset2,val2);
    t2=PAPI_get_real_usec();

    if ( !TESTS_QUIET ) {
      printf("\n(calculated independent of PAPI)\n");
      printf("\tOperations= %.1f Mflop",y*1e-6);  
      printf("\t(%g Mflop/s)\n\n",((float)y/(t2-t1)));
      printf("PAPI measurements:\n");
    }
    for (j=0; j<nev1; j++) {
      PAPI_label_event(events[j],des);
      if ( !TESTS_QUIET )
	printf("%20s = %lld\n", des, values[j]);
      dtmp = (double) values[j];
      valsum[j] += dtmp;
      valsqsum[j] += dtmp * dtmp;
    }
    if ( !TESTS_QUIET )
      printf("\n");
    for (j=0; j<nev2; j++) {
      PAPI_label_event(events[nevents-j-1],des);
      if ( !TESTS_QUIET )
	printf("%20s = %lld\n", des, val2[j]);
      dtmp = (double) val2[j];
      val2sum[j] += dtmp;
      val2sqsum[j] += dtmp * dtmp;
    }
    if ( !TESTS_QUIET )
      printf("\n");
  }

  if ( !TESTS_QUIET ) {
    printf("\n\nEstimated variance relative to average counts:\n");
    for (j=0;j<((nev1>nev2)?nev1:nev2);j++)
      printf("   Event %.2d",j);
    printf("\n");
  }

  i=nev1+nev2;
  for (j=0;j<nev1;j++) {
    spread[j]=valsqsum[j];
    spread[j]-=(valsum[j]*valsum[j])/REPEATS;
    spread[j]=sqrt(spread[j]/(REPEATS-1));
    if(valsum[j]>0.9)
      spread[j]=REPEATS*spread[j]/valsum[j];
    values[j]=valsum[j]/REPEATS;
    if ( !TESTS_QUIET )
      printf("%9.2g  ",spread[j]);
    /* Make sure that NaN get counted as errors */
    if(spread[j]<RELTOLERANCE) i--;
  }
  if ( !TESTS_QUIET )
    printf("Event set 1\n");

  for (j=0;j<nev2;j++) {
    spread2[j]=val2sqsum[j];
    spread2[j]-=(val2sum[j]*val2sum[j])/REPEATS;
    spread2[j]=sqrt(spread2[j]/(REPEATS-1));
    if(val2sum[j]>0.9)
      spread2[j]=REPEATS*spread2[j]/val2sum[j];
    val2[j]=val2sum[j]/REPEATS;
    if ( !TESTS_QUIET )
      printf("%9.2g  ",spread2[j]);
    /* Make sure that NaN get counted as errors */
    if(spread2[j]<RELTOLERANCE) i--;
  }
  if ( !TESTS_QUIET )
    printf("Event set 2\n\n");

  if ( i )
    test_fail(__FILE__,__LINE__,"Values outside threshold", i);

  if ( !TESTS_QUIET )
    printf("\nVerification run with one event set:\n");
  if((retval=PAPI_start(eventset)))
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);
  x=1.0;
  t1=PAPI_get_real_usec();
  y=dummy3(x,iters);  
  if((retval=PAPI_stop(eventset,refvals)))
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);;
  t2=PAPI_get_real_usec();

  if ( !TESTS_QUIET ) {
    printf("\n(calculated independent of PAPI)\n");
    printf("\tOperations= %.1f Mflop",y*1e-6);  
    printf("\t(%g Mflop/s)\n\n",((float)y/(t2-t1)));
    printf("PAPI reference (multiplexed) measurements:\n");
  }
  for (j=0; j<nevents; j++) {
    PAPI_label_event(events[j],des);
    if ( !TESTS_QUIET )
      printf("%20s = %lld\n", des, refvals[j]);
  }
  if ( !TESTS_QUIET )
    printf("\n");

  if ( !TESTS_QUIET ) {
    printf("\n\nRelative error of the mean value of the mixed run\n"
	   "to the reference run:\n");
    for (j=0;j<((nev1>nev2)?nev1:nev2);j++)
      printf("   Event %.2d",j);
    printf("\n");
  }
  i=nev1+nev2;
  for (j=0;j<nev1;j++) {
    dtmp= abs(values[j]-refvals[j]);
    if(refvals[j])
      dtmp/=refvals[j];
    /* Make sure that NaN get counted as errors */
    if(dtmp<RELTOLERANCE)
      i--;
    else if(refvals[j]<MINCOUNTS) /* Neglect inprecise results with low counts */
      i--;
    if ( !TESTS_QUIET )
      printf("%9.2g  ",dtmp);
  }
  if ( !TESTS_QUIET )
    printf("\tEvent set 1\n");
  for (j=0;j<nev2;j++) {
    dtmp=abs(val2[j]-refvals[nevents-j-1]);
    if(refvals[nevents-j-1])
      dtmp/=refvals[nevents-j-1];
    /* Make sure that NaN get counted as errors */
    if(dtmp<RELTOLERANCE)
      i--;
    else if(val2[j]<MINCOUNTS) /* Neglect inprecise results with low counts */
      i--;
    if ( !TESTS_QUIET )
      printf("%9.2g  ",dtmp);
  }
  if ( !TESTS_QUIET )
    printf("\tEvent set 2\n\n");
  
  if ( i )
    test_fail(__FILE__,__LINE__,"Values differ from reference", i);
  else 
    test_pass(__FILE__,NULL, 0);

  return 0;
}

static double dummy3(double x,int iters) {  
  int i;  
  double w,y,z,a,b,c,d,e,f,g,h;  
  double one;  
  one=1.0;  
  w=x; y=x; z=x;
  a=x; b=x; c=x; d=x; e=x;  
  f=x; g=x; h=x;  
  for (i=1;i<=iters;i++) {  
    w=w*1.000000000001+one;   
    y=y*1.000000000002+one;   
    z=z*1.000000000003+one;   
    a=a*1.000000000004+one;   
    b=b*1.000000000005+one;   
    c=c*0.999999999999+one;   
    d=d*0.999999999998+one;   
    e=e*0.999999999997+one;  
    f=f*0.999999999996+one;  
    g=h*0.999999999995+one;  
    h=h*1.000000000006+one;  
  }  
  return 2.0*(a+b+c+d+e+f+w+x+y+z+g+h);  
}  
