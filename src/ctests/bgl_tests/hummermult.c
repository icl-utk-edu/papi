#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "bgl_perfctr_events.h"


#define N 8
#define NCOUNTS 5

int main(int argc, char* argv[]) {
  double v1[N], v2[N], v3[N], r1[N], r2[N];
  double a=1.01,b=1.02,c=1.03,t=0.0,t2=0.0;
  int i, rank;
  int retval, perr, ev_set = PAPI_NULL;
  int encoding;
  long long counts[NCOUNTS];

#include "bglpersonality.h"
#include "rts.h"

  if(PAPI_VER_CURRENT!=(perr=PAPI_library_init(PAPI_VER_CURRENT)))
    printf("\nPAPI_library_init failed. %s\n",PAPI_strerror(perr));

 {
   BGLPersonality me;
   rts_get_personality(&me,sizeof(me));
   if(me.xCoord != 0 ) goto fine;
   if(me.yCoord != 0 ) goto fine;
   if(me.zCoord != 0 ) goto fine;
 }

  for(i=0;i<N;i++) {
    v1[i]=1.01+0.01*i;
    v2[i]=2.01+0.01*i;
    v3[i]=3.01+0.01*i;
    r1[i]=v1[i]*v2[i]+v3[i];
  }

  if((perr=PAPI_create_eventset(&ev_set)))
    printf("\nPAPI_create_eventset failed. %s\n",PAPI_strerror(perr));

  /*
  encoding=( BGL_FPU_ARITH_MULT_DIV & 0x3FF );
  encoding=( BGL_FPU_ARITH_ADD_SUBTRACT & 0x3FF );
  encoding=( BGL_FPU_ARITH_TRINARY_OP & 0x3FF );
  */

  if((perr=PAPI_add_event(ev_set,PAPI_TOT_CYC)))
    printf("PAPI_add_event failed. %s\n",PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_FPU_ARITH_OEDIPUS_OP", &encoding);
  if (retval != PAPI_OK)
      printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);

  if((perr=PAPI_add_event(ev_set,encoding)))
    printf("\nPAPI_add_event failed. %s\n",PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_2NDFPU_ARITH_OEDIPUS_OP", &encoding);
  if (retval != PAPI_OK)
      printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  
  if((perr=PAPI_add_event(ev_set,encoding)))
    printf("\nPAPI_add_event failed. %s\n",PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_FPU_LDST_QUAD_LD", &encoding);
  if (retval != PAPI_OK)
      printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);

  if((perr=PAPI_add_event(ev_set,encoding)))
    printf("\nPAPI_add_event failed. %s\n",PAPI_strerror(perr));


  retval = PAPI_event_name_to_code("BGL_2NDFPU_LDST_QUAD_LD", &encoding);
  if (retval != PAPI_OK)
      printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  
  if((perr=PAPI_add_event(ev_set,encoding)))
    printf("\nPAPI_add_event failed. %s\n",PAPI_strerror(perr));


  printf("\nAssigning a vector of length %1d and computing "
	 "A()=B()*C()+D().\n",N);

  if((perr=PAPI_start(ev_set)))
    printf("\nPAPI_start_event failed. %s\n",PAPI_strerror(perr));

  for(i=0;i<N;i++) r2[i]=-1.001;
  fpmaddv(N,v1,v2,v3,r2);

  if((perr=PAPI_read(ev_set,counts)))
    printf("PAPI_read failed. %s\n",PAPI_strerror(perr));

  printf("Counts registered: ");
  for(i=0;i<NCOUNTS;i++) printf("  %12llu",counts[i]);
  printf("\n");


  for(i=0;i<N;i++) {
    printf(" %g * %g + % g = %g  (%g)\n",
	   v1[i],v2[i],v3[i],r2[i],r1[i]);
  }

  for(i=0;i<N;i++) r2[i]=-1.001;

  printf("\nResetting the running counter and computing "
	 "A(1:%1d)=B()*C()+D().\n",N);

  if((perr=PAPI_reset(ev_set)))
    printf("\nPAPI_reset failed. %s\n",PAPI_strerror(perr));

  fpmaddv(N,v1,v2,v3,r2);


  if((perr=PAPI_stop(ev_set,counts)))
    printf("PAPI_stop failed. %s\n",PAPI_strerror(perr));

  for(i=0;i<N;i++) {
    printf(" %g * %g + % g = %g  (%g)\n",
	   v1[i],v2[i],v3[i],r2[i],v1[i]*v2[i]+v3[i]);
  }

  printf("Testing to read stopped counters\n");
  if((perr=PAPI_read(ev_set,counts)))
    printf("PAPI_read failed. %s\n",PAPI_strerror(perr));
    
  printf("Counts registered: ");
  for(i=0;i<NCOUNTS;i++) printf("  %12llu",counts[i]);
  printf("\n");

 fine:
  PAPI_shutdown();
  return 0;
}
