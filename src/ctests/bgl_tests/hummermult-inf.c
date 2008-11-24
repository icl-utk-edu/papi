#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "bgl_perfctr_events.h"
#include "linux-bgl-events.c"

#undef FPMA  // Use the FPM version of the computation

#define N 4000000
#define NITER 1100
#define NCOUNTS 5

int main(int argc, char* argv[]) {
  double v1[N], v2[N], v3[N], r1[N], r2[N];
  double a=1.01,b=1.02,c=1.03,t=0.0,t2=0.0;
  int i, rank, iter;
  int perr, ev_set = PAPI_NULL;
  int encoding;
  long long counts[NCOUNTS];

#include "bglpersonality.h"
#include "rts.h"

  if(PAPI_VER_CURRENT!=(perr=PAPI_library_init(PAPI_VER_CURRENT)))
    printf("PAPI_library_init failed. %s\n",PAPI_strerror(perr));

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

  for(i=0;i<N;i++) r2[i]=-1.001;


  if((perr=PAPI_create_eventset(&ev_set)))
    printf("PAPI_create_eventset failed. %s\n",PAPI_strerror(perr));

  if((perr=PAPI_add_event(ev_set,PAPI_TOT_CYC)))
    printf("PAPI_add_event failed. %s\n",PAPI_strerror(perr));

  encoding=( PNE_BGL_FPU_ARITH_TRINARY_OP );
  if((perr=PAPI_add_event(ev_set,encoding)))
    printf("\nPAPI_add_event failed. %s\n",PAPI_strerror(perr));

  encoding=( PNE_BGL_2NDFPU_ARITH_TRINARY_OP );
  if((perr=PAPI_add_event(ev_set,encoding)))
    printf("\nPAPI_add_event failed. %s\n",PAPI_strerror(perr));

  encoding=( PNE_BGL_FPU_LDST_DBL_LD );
  if((perr=PAPI_add_event(ev_set,encoding)))
    printf("\nPAPI_add_event failed. %s\n",PAPI_strerror(perr));

  encoding=( PNE_BGL_2NDFPU_LDST_DBL_LD );
  if((perr=PAPI_add_event(ev_set,encoding)))
    printf("\nPAPI_add_event failed. %s\n",PAPI_strerror(perr));

  if((perr=PAPI_start(ev_set)))
    printf("PAPI_start_event failed. %s\n",PAPI_strerror(perr));


  printf("\n\nPerforming %d iterations of vector operations for\n"
	 "a total of %lld (0x%llx) number of FMAs\n",
	 NITER,((long long)NITER)*N,((long long)NITER)*N);

  for(iter=0;iter<NITER;iter++) {

    if(iter%100==0)
      printf("\t----  Iteration %4.4d of %4.4d ----\n",iter,NITER);

#ifdef FPMA
    fpmaddv(N,v1,v2,v3,r2);
#else
    fmaddv(N,v1,v2,v3,r2);
#endif

  }

  if((perr=PAPI_stop(ev_set,counts)))
    printf("PAPI_stop failed. %s\n",PAPI_strerror(perr));
  
  printf("Counts registered: ");
  for(i=0;i<NCOUNTS;i++) printf("  %12llu",counts[i]);
  printf("\n");

 fine:
  PAPI_shutdown();
  return 0;
}
