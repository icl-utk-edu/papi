#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "bgl_perfctr_events.h"
#include "rts.h"
#include "bgllockbox.h"
#include "bglpersonality.h"
#include "rts.h"

#define N 64*1024*1024
#define NCOUNTS 6

int main(int argc, char* argv[]) {
  unsigned char v[N];
  int i, k, ncnt;
  int perr, ev_set = PAPI_NULL;
  int encoding;
  long long counts[NCOUNTS];
  int evlist[NCOUNTS];
  char evname[NCOUNTS][PAPI_MAX_STR_LEN];
  BGL_Barrier *barrier;
  FILE *fh;

  if(PAPI_VER_CURRENT!=(perr=PAPI_library_init(PAPI_VER_CURRENT)))
    printf("\nPAPI_library_init failed. %s\n",PAPI_strerror(perr));

 {
   BGLPersonality me;
   rts_get_personality(&me,sizeof(me));
   if(me.xCoord != 0 ) goto fine;
   if(me.yCoord != 0 ) goto fine;
   if(me.zCoord != 0 ) goto fine;
 }

#ifdef DEBUG
 sprintf(evname[0],"event_dump_%.3d",rts_get_processor_id());
 fh=fopen(evname[0],"w");
#endif

#ifdef VN
 { 
   BGL_Barrier ** volatile dirtytrick=(BGL_Barrier **) 0xFFFFFFF0;
   if(rts_get_processor_id()) {
     printf("Virtual node mode, waiting for lock reference through SRAM\n");
     while(! *dirtytrick) {};
     barrier=*dirtytrick;
     printf("Virtual node mode, got lock reference through SRAM\n");
   } else {
     barrier=rts_allocate_barrier();
     if(!barrier)
       printf("Barrier allocation failed...\n");
     printf("Virtual node mode, trying to pass lock reference through SRAM\n");
     *dirtytrick=barrier;
   }
 }
#endif

 printf("Cache warm up/flush: Writing %d chars into array.\n",N);
 for(i=0;i<N;i++) {
   v[i]=i % 256;
 }

  if((perr=PAPI_create_eventset(&ev_set)))
    printf("\nPAPI_create_eventset failed. %s\n",PAPI_strerror(perr));

  if((perr=PAPI_add_event(ev_set,PAPI_L2_DCA)))
    printf("[%d] PAPI_add_event failed. %s\n",__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_add_event(ev_set,PAPI_L2_DCH)))
    printf("[%d] PAPI_add_event failed. %s\n",__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_add_event(ev_set,PAPI_L3_TCH)))
    printf("[%d] PAPI_add_event failed. %s\n",__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_add_event(ev_set,PAPI_L3_TCM)))
    printf("[%d] PAPI_add_event failed. %s\n",__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_add_event(ev_set,PAPI_L3_STM)))
    printf("[%d] PAPI_add_event failed. %s\n",__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_add_event(ev_set,PAPI_L3_LDM)))
    printf("[%d] PAPI_add_event failed. %s\n",__LINE__,PAPI_strerror(perr));

  ncnt=NCOUNTS;
  if((perr=PAPI_list_events(ev_set,evlist,&ncnt)))
    printf("[%d] PAPI_list_events failed. %s\n",__LINE__,PAPI_strerror(perr));
  for(i=0;i<ncnt;i++)
    if((perr=PAPI_event_code_to_name(evlist[i],evname[i])) == PAPI_ENOTPRESET)
      sprintf(evname[i],"%s",BGL_PERFCTR_event_table[evlist[i]].event_name);
    else if(perr!=PAPI_OK)
      printf("[%d] Naming event failed. %s [i=%d  event=%d]\n",__LINE__,
	     PAPI_strerror(perr),i,evlist[i]);

  printf("The following %d events were named:\n",ncnt);
  for(i=0;i<ncnt;i++) printf("  %12s",evname[i]);
  printf("\n");
  
  printf("Writing %d chars into array\n",N);

#ifdef VN
  BGL_Barrier_Pass(barrier);
#endif

  if((perr=PAPI_start(ev_set)))
    printf("\nPAPI_start_event failed. %s\n",PAPI_strerror(perr));

#ifdef DEBUG
#ifdef VN
  BGL_Barrier_Pass(barrier);
#endif
  bgl_perfctr_dump_state(fh);
#ifdef VN
  BGL_Barrier_Pass(barrier);
#endif
#endif

  for(i=0;i<N;i++) {
    v[i]=256-v[i];
  }

#ifdef VN  
  BGL_Barrier_Pass(barrier);
#endif
  if((perr=PAPI_read(ev_set,counts)))
    printf("PAPI_read failed. %s\n",PAPI_strerror(perr));
#ifdef DEBUG
#ifdef VN  
  BGL_Barrier_Pass(barrier);
#endif
  bgl_perfctr_dump_state(fh);
#ifdef VN
  BGL_Barrier_Pass(barrier);
#endif
#endif

  printf("Counts registered: \n");
  for(i=0;i<NCOUNTS;i++) printf("  %12s",evname[i]);
  printf("\n");
  for(i=0;i<NCOUNTS;i++) printf("  %12llu",counts[i]);
  printf("\n");

  printf("Writing %d chars into array twice\n",N);

#ifdef VN
  BGL_Barrier_Pass(barrier);
#endif

  if((perr=PAPI_reset(ev_set)))
    printf("\nPAPI_reset_event failed. %s\n",PAPI_strerror(perr));

  for(k=0;k<2;k++)
    for(i=0;i<N;i++) 
      v[i]=256-v[i];

#ifdef VN
  BGL_Barrier_Pass(barrier);
#endif

  if((perr=PAPI_read(ev_set,counts)))
    printf("PAPI_read failed. %s\n",PAPI_strerror(perr));

  printf("Counts registered: \n");
  for(i=0;i<NCOUNTS;i++) printf("  %12s",evname[i]);
  printf("\n");
  for(i=0;i<NCOUNTS;i++) printf("  %12llu",counts[i]);
  printf("\n");

  printf("Writing %d chars into array again\n",N);

#ifdef VN
  BGL_Barrier_Pass(barrier);
#endif

  if((perr=PAPI_reset(ev_set)))
    printf("\nPAPI_reset_event failed. %s\n",PAPI_strerror(perr));

  for(i=0;i<N;i++) 
    v[i]=256-v[i];

#ifdef VN
  BGL_Barrier_Pass(barrier);
#endif

  if((perr=PAPI_read(ev_set,counts)))
    printf("PAPI_read failed. %s\n",PAPI_strerror(perr));

  printf("Counts registered: \n");
  for(i=0;i<NCOUNTS;i++) printf("  %12s",evname[i]);
  printf("\n");
  for(i=0;i<NCOUNTS;i++) printf("  %12llu",counts[i]);
  printf("\n");

#ifdef DEBUG
  fclose(fh);
#endif

 fine:
  PAPI_shutdown();
  return 0;
}
