#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "mpi.h"
#include "papiStdEventDefs.h"
#include <bgl_perfctr.h>

#define N (1024*1024)
#define ROOT 0
#define NCOUNTS 6

char *evname[] = {"PAPI_BGL_TS_32B", NULL};
char *evname2[] = {"BGL_UPC_TS_XM_32B_CHUNKS", "BGL_UPC_TS_XP_32B_CHUNKS", "BGL_UPC_TS_YM_32B_CHUNKS","BGL_UPC_TS_YP_32B_CHUNKS",
            "BGL_UPC_TS_ZM_32B_CHUNKS", "BGL_UPC_TS_ZP_32B_CHUNKS", NULL};
char *evname3[] = {"PAPI_BGL_TR_DPKT", NULL};
char *evname4[] = {"BGL_UPC_TR_SNDR_2_VC1_DPKTS_SENT", "BGL_UPC_TR_SNDR_0_VC0_DPKTS_SENT", "BGL_UPC_TR_SNDR_0_VC1_DPKTS_SENT", "BGL_UPC_TR_SNDR_1_VC0_DPKTS_SENT",
            "BGL_UPC_TR_SNDR_1_VC1_DPKTS_SENT", "BGL_UPC_TR_SNDR_2_VC0_DPKTS_SENT", NULL};

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
  int ev_set = PAPI_NULL, ev_set2 = PAPI_NULL, ev_set3 = PAPI_NULL, ev_set4 = PAPI_NULL;
  int nev,nev2;
  long long counts[NCOUNTS], counts2[NCOUNTS];
  int EventCode, retval, perr;
  
  if(PAPI_VER_CURRENT!=(retval=PAPI_library_init(PAPI_VER_CURRENT)))
    printf("\nPAPI_library_init failed. %s\n",PAPI_strerror(retval));

  if((retval=PAPI_create_eventset(&ev_set)))
    test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  if((retval=PAPI_create_eventset(&ev_set2)))
    test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  nev=0;
  while(evname[nev] != NULL){
    retval = PAPI_event_name_to_code(evname[nev], &EventCode);
    if (retval != PAPI_OK)
       test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);

    if((retval=PAPI_add_event(ev_set, EventCode)))
       test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
    nev++;

  }
  
  nev2=0;
  while(evname2[nev2] != NULL){
    retval = PAPI_event_name_to_code(evname2[nev2], &EventCode);
    if (retval != PAPI_OK)
       test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);

    if((retval=PAPI_add_event(ev_set2, EventCode)))
       test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
    nev2++;

  }
  
  printf("The following %d preset events were added:\n",nev);
  for(i=0;i<nev;i++) printf("  %12s",evname[i]);
  printf("\n");
  
  printf("The following %d native events were added:\n",nev2);
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

  if((retval=PAPI_start(ev_set)))
    test_fail(__FILE__, __LINE__, "PAPI_start", retval);

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);

#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  if((retval=PAPI_stop(ev_set,counts)))
    test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

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

  if((retval=PAPI_start(ev_set2)))
    test_fail(__FILE__, __LINE__, "PAPI_start", retval);

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif

  if((retval=PAPI_stop(ev_set2,counts2)))
    test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  printf("Counts registered: \n");
  for(i=0;i<nev2;i++) printf("  %12s",evname2[i]);
  printf("\n");
  for(i=0;i<nev2;i++) printf("  %12llu",counts2[i]);
  printf("\n");
  
  /************************************************************************/

  if((retval=PAPI_cleanup_eventset(ev_set)))
    test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
  if((perr=PAPI_destroy_eventset(&ev_set)))
    test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);

  if((retval=PAPI_cleanup_eventset(ev_set2)))
    test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
  if((retval=PAPI_destroy_eventset(&ev_set2)))
    test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);

  /************************************************************************/

  if((retval=PAPI_create_eventset(&ev_set3)))
    test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  if((retval=PAPI_create_eventset(&ev_set4)))
    test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  nev=0;
  while(evname3[nev] != NULL){
    retval = PAPI_event_name_to_code(evname3[nev], &EventCode);
    if (retval != PAPI_OK)
       test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);

    if((retval=PAPI_add_event(ev_set3, EventCode)))
       test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
    nev++;

  }
  
  nev2=0;
  while(evname4[nev2] != NULL){
    retval = PAPI_event_name_to_code(evname4[nev2], &EventCode);
    if (retval != PAPI_OK)
       test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);
    else
       printf("rank(%d): %d %s %p\n", rank, nev2, evname4[nev2], EventCode);

    if((retval=PAPI_add_event(ev_set4, EventCode)))
       test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
    nev2++;

  }
  
  printf("The following %d preset events were added:\n",nev);
  for(i=0;i<nev;i++) printf("  %12s",evname3[i]);
  printf("\n");
  
  printf("The following %d native events were added:\n",nev2);
  for(i=0;i<nev2;i++) printf("  %12s",evname4[i]);
  printf("\n");

  /************************************************************************/

  printf("Experiment 1: Using derived event for torus traffic\n");

  if((retval=PAPI_start(ev_set3)))
    test_fail(__FILE__, __LINE__, "PAPI_start", retval);

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);

#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  if((retval=PAPI_stop(ev_set3,counts)))
    test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif
  
  printf("Counts registered: \n");
  for(i=0;i<nev;i++) printf("  %12s",evname3[i]);
  printf("\n");
  for(i=0;i<nev;i++) printf("  %12llu",counts[i]);
  printf("\n");
  
  /************************************************************************/
  
  printf("Experiment 2: Using native events for torus traffic\n");

  if((retval=PAPI_start(ev_set4)))
    test_fail(__FILE__, __LINE__, "PAPI_start", retval);

/*  if((perr=MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD))) */
  if((perr=Pt2Pt_MPI_Bcast(v,N,MPI_CHAR,ROOT,MPI_COMM_WORLD)))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Bcast failed. err=%d\n",__FILE__,__LINE__,perr);
  
#ifdef DEBUG
  if(rank==7) {
    bgl_perfctr_update();
    bgl_perfctr_dump_state(stdout);
  }
#endif

  if((retval=PAPI_stop(ev_set4,counts2)))
    test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  printf("Counts registered: \n");
  for(i=0;i<nev2;i++) printf("  %12s",evname4[i]);
  printf("\n");
  for(i=0;i<nev2;i++) printf("  %12llu",counts2[i]);
  printf("\n");
  
  /************************************************************************/

  if((retval=PAPI_cleanup_eventset(ev_set3)))
    test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
  if((perr=PAPI_destroy_eventset(&ev_set3)))
    test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);

  if((retval=PAPI_cleanup_eventset(ev_set4)))
    test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
  if((retval=PAPI_destroy_eventset(&ev_set4)))
    test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);

  /************************************************************************/

  if((perr=MPI_Finalize()))
    printf("\n[ERROR! ERROR! ERROR! %s:%d] MPI_Finalize failed. err=%d\n",__FILE__,__LINE__,perr);

  PAPI_shutdown();

  return 0;
}
