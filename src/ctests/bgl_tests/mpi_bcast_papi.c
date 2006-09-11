#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "mpi.h"
#include "papiStdEventDefs.h"
#include <bgl_perfctr.h>

#define N (1024*1024)
#define ROOT 0
#define NCOUNTS 6

int Pt2Pt_MPI_Bcast(char *buf, int sz, MPI_Datatype dt, int root, MPI_Comm comm)
{
  int i,localrank,localsize;
  int perr=MPI_SUCCESS;
  MPI_Status status;

  if((perr=MPI_Comm_rank(MPI_COMM_WORLD,&localrank)))
    {
      printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Comm_rank failed. err=%d\n",__FILE__,__LINE__,perr);
      return perr;
    }

  if((perr=MPI_Comm_size(MPI_COMM_WORLD,&localsize)))
    {
      printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Comm_size failed. err=%d\n",__FILE__,__LINE__,perr);
      return perr;
    }

  if (localrank==root)
    {
      for (i=0; i<localsize; i++)
      {
        if (i!=localrank) perr=MPI_Send(buf,sz,dt,i,42,comm);
        if (perr!=MPI_SUCCESS) return perr;
     }
    }
  else
    {
      perr=MPI_Recv(buf,sz,dt,root,42,comm,&status);
    }
  MPI_Barrier(comm);
  return perr;
}

int main(int argc, char* argv[]) {
  char v[N];
  int i, rank;
  int perr, ev_set = PAPI_NULL, ev_set2 = PAPI_NULL;
  int nev,nev2;
  long_long counts[NCOUNTS], counts2[NCOUNTS];
  int evlist[NCOUNTS];
  char evname[NCOUNTS][PAPI_MAX_STR_LEN], evname2[NCOUNTS][PAPI_MAX_STR_LEN];
  int EventCode, retval;
  
  if(PAPI_VER_CURRENT!=(perr=PAPI_library_init(PAPI_VER_CURRENT)))
    printf("\nPAPI_library_init failed. %s\n",PAPI_strerror(perr));

  if((perr=PAPI_create_eventset(&ev_set)))
    printf("\nPAPI_create_eventset failed. %s\n",PAPI_strerror(perr));
  if((perr=PAPI_create_eventset(&ev_set2)))
    printf("\nPAPI_create_eventset 2 failed. %s\n",PAPI_strerror(perr));

  if((perr=PAPI_add_event(ev_set,PAPI_BGL_TS_32B)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TS_XM_32B_CHUNKS", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TS_XP_32B_CHUNKS", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TS_YM_32B_CHUNKS", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TS_YP_32B_CHUNKS", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TS_ZM_32B_CHUNKS", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TS_ZP_32B_CHUNKS", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  nev=NCOUNTS;
  if((perr=PAPI_list_events(ev_set,evlist,&nev)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_list_events failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  for(i=0;i<nev;i++)
    if((perr=PAPI_event_code_to_name(evlist[i],evname[i])) == PAPI_ENOTPRESET)
      sprintf(evname[i],"%s",BGL_PERFCTR_event_table[evlist[i] & 0x3FF].event_name);
    else if(perr!=PAPI_OK)
      printf("\n[ERROR! ERROR! ERROR! %s:%d] Naming event failed. %s [i=%d  event=%d]\n",__FILE__,__LINE__,
	     PAPI_strerror(perr),i,evlist[i]);

  printf("The following %d events were named:\n",nev);
  for(i=0;i<nev;i++) printf("  %12s",evname[i]);
  printf("\n");
  
  nev2=NCOUNTS;
  if((perr=PAPI_list_events(ev_set2,evlist,&nev2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_list_events failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  for(i=0;i<nev2;i++)
    if((perr=PAPI_event_code_to_name(evlist[i],evname2[i])) == PAPI_ENOTPRESET)
      sprintf(evname2[i],"%s",BGL_PERFCTR_event_table[evlist[i] & 0x3FF].event_name);
    else if(perr!=PAPI_OK)
      printf("\n[ERROR! ERROR! ERROR! %s:%d] Naming event failed. %s [i=%d  event=%d]\n",__FILE__,__LINE__,
	     PAPI_strerror(perr),i,evlist[i]);

  printf("The following %d events were named:\n",nev2);
  for(i=0;i<nev2;i++) printf("  %12s",evname2[i]);
  printf("\n");
  
  /************************************************************************/

  if((perr=MPI_Init(&argc,&argv)))
     printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Init failed. err=%d\n",__FILE__,__LINE__,perr);

  if((perr=MPI_Comm_rank(MPI_COMM_WORLD,&rank)))
     printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Comm_rank failed. err=%d\n",__FILE__,__LINE__,perr);
  
  printf("Cache warm up/flush: Writing %d chars into array.\n",N);
  for(i=0;i<N;i++) {
    v[i]= rank;  //i % 256;
  }

  /************************************************************************/

  printf("Experiment 1: Using derived event for torus traffic\n");

  if((perr=PAPI_start(ev_set)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);

#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  if((perr=PAPI_stop(ev_set,counts)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_stop failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  printf("Counts registered: \n");
  for(i=0;i<nev;i++) printf("  %12s",evname[i]);
  printf("\n");
  for(i=0;i<nev;i++) printf("  %12llu",counts[i]);
  printf("\n");
  
  /************************************************************************/
  
  printf("Experiment 2: Using native events for torus traffic\n");

  if((perr=PAPI_start(ev_set2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif

  if((perr=PAPI_stop(ev_set2,counts2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_stop failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  printf("Counts registered: \n");
  for(i=0;i<nev2;i++) printf("  %12s",evname2[i]);
  printf("\n");
  for(i=0;i<nev2;i++) printf("  %12llu",counts2[i]);
  printf("\n");
  
  /************************************************************************/
  
  printf("Experiment : Using both native events and derived event for torus traffic\n");
  printf("Experiment 3: Two broadcasts, with native events reset inbetween.\n");

  if((perr=PAPI_start(ev_set)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_start(ev_set2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start 2 failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }

  if((perr=PAPI_read(ev_set2,counts2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_read failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  printf("Counts skipped by reset: \n");
  for(i=0;i<nev2;i++) printf("  %12s",evname2[i]);
  printf("\n");
  for(i=0;i<nev2;i++) printf("  %12llu",counts2[i]);
  printf("\n");
#endif
  
  if((perr=PAPI_reset(ev_set2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))  
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  if((perr=PAPI_stop(ev_set2,counts2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_read failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_stop(ev_set,counts)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_read failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  printf("Counts registered in second broadcast: \n");
  for(i=0;i<nev2;i++) printf("  %12s",evname2[i]);
  printf("\n");
  for(i=0;i<nev2;i++) printf("  %12llu",counts2[i]);
  printf("\n");


  printf("Counts for both broadcasts registered: \n");
  for(i=0;i<nev;i++) printf("  %12s",evname[i]);
  printf("\n");
  for(i=0;i<nev;i++) printf("  %12llu",counts[i]);
  printf("\n");
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif

  /************************************************************************/

  if((perr=PAPI_cleanup_eventset(&ev_set)))
    printf("\nPAPI_cleanup_eventset failed. %s\n",PAPI_strerror(perr));
  if((perr=PAPI_destroy_eventset(&ev_set)))
    printf("\nPAPI_destroy_eventset failed. %s\n",PAPI_strerror(perr));
  if((perr=PAPI_cleanup_eventset(&ev_set2)))
    printf("\nPAPI_cleanup_eventset failed. %s\n",PAPI_strerror(perr));
  if((perr=PAPI_destroy_eventset(&ev_set2)))
    printf("\nPAPI_destroy_eventset 2 failed. %s\n",PAPI_strerror(perr));

  if((perr=PAPI_create_eventset(&ev_set)))
    printf("\nPAPI_create_eventset failed. %s\n",PAPI_strerror(perr));
  if((perr=PAPI_create_eventset(&ev_set2)))
    printf("\nPAPI_create_eventset 2 failed. %s\n",PAPI_strerror(perr));

  if((perr=PAPI_add_event(ev_set,PAPI_BGL_TR_DPKT)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TR_SNDR_2_VC1_DPKTS_SENT", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TR_SNDR_0_VC0_DPKTS_SENT", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TR_SNDR_0_VC1_DPKTS_SENT", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TR_SNDR_1_VC0_DPKTS_SENT", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TR_SNDR_1_VC1_DPKTS_SENT", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  retval = PAPI_event_name_to_code("BGL_UPC_TR_SNDR_2_VC0_DPKTS_SENT", &EventCode);
  if (retval != PAPI_OK)
     printf("%s:%d  PAPI_event_name_to_code  %d\n", __FILE__,__LINE__, retval);
  if((perr=PAPI_add_event(ev_set2, EventCode )))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_add_event failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  nev=NCOUNTS;
  if((perr=PAPI_list_events(ev_set,evlist,&nev)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_list_events failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  for(i=0;i<nev;i++)
    if((perr=PAPI_event_code_to_name(evlist[i],evname[i])) == PAPI_ENOTPRESET)
      sprintf(evname[i],"%s",BGL_PERFCTR_event_table[evlist[i] & 0x3FF].event_name);
    else if(perr!=PAPI_OK)
      printf("\n[ERROR! ERROR! ERROR! %s:%d] Naming event failed. %s [i=%d  event=%d]\n",__FILE__,__LINE__,
	     PAPI_strerror(perr),i,evlist[i]);

  nev2=NCOUNTS;
  if((perr=PAPI_list_events(ev_set2,evlist,&nev2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_list_events failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  for(i=0;i<nev2;i++)
    if((perr=PAPI_event_code_to_name(evlist[i],evname2[i])) == PAPI_ENOTPRESET)
      sprintf(evname2[i],"%s",BGL_PERFCTR_event_table[evlist[i] & 0x3FF].event_name);
    else if(perr!=PAPI_OK)
      printf("\n[ERROR! ERROR! ERROR! %s:%d] Naming event failed. %s [i=%d  event=%d]\n",__FILE__,__LINE__,
	     PAPI_strerror(perr),i,evlist[i]);

  /************************************************************************/

  printf("Experiment 1: Using derived event for tree traffic\n");

  if((perr=PAPI_start(ev_set)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);

#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  if((perr=PAPI_stop(ev_set,counts)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_stop failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  printf("Counts registered: \n");
  for(i=0;i<nev;i++) printf("  %12s",evname[i]);
  printf("\n");
  for(i=0;i<nev;i++) printf("  %12llu",counts[i]);
  printf("\n");
  
  /************************************************************************/
  
  printf("Experiment 2: Using native events for tree traffic\n");

  if((perr=PAPI_start(ev_set2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

 /* if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif

  if((perr=PAPI_stop(ev_set2,counts2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_stop failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  printf("Counts registered: \n");
  for(i=0;i<nev2;i++) printf("  %12s",evname2[i]);
  printf("\n");
  for(i=0;i<nev2;i++) printf("  %12llu",counts2[i]);
  printf("\n");
  
  /************************************************************************/
  
  printf("Experiment : Using both native events and derived event for tree traffic\n");
  printf("Experiment 3: Two broadcasts, with native events reset inbetween.\n");

  if((perr=PAPI_start(ev_set)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_start(ev_set2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start 2 failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  /*if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))*/
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }

  if((perr=PAPI_read(ev_set2,counts2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_read failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  printf("Counts skipped by reset: \n");
  for(i=0;i<nev2;i++) printf("  %12s",evname2[i]);
  printf("\n");
  for(i=0;i<nev2;i++) printf("  %12llu",counts2[i]);
  printf("\n");
#endif
  
  if((perr=PAPI_reset(ev_set2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_start failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  /*if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))*/
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  if((perr=PAPI_stop(ev_set2,counts2)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_stop failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));
  if((perr=PAPI_stop(ev_set,counts)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] PAPI_stop failed. %s\n",__FILE__,__LINE__,PAPI_strerror(perr));

  printf("Counts registered in second broadcast: \n");
  for(i=0;i<nev2;i++) printf("  %12s",evname2[i]);
  printf("\n");
  for(i=0;i<nev2;i++) printf("  %12llu",counts2[i]);
  printf("\n");


  printf("Counts for both broadcasts registered: \n");
  for(i=0;i<nev;i++) printf("  %12s",evname[i]);
  printf("\n");
  for(i=0;i<nev;i++) printf("  %12llu",counts[i]);
  printf("\n");
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif

  /************************************************************************/

  if((perr=MPI_Finalize()))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Finalize failed. err=%d\n",__FILE__,__LINE__,perr);

  PAPI_shutdown();

  return 0;
}
