#include <stdio.h>
#include <stdlib.h>
#include "bgl_perfctr.h"
#include "bgl_perfctr_events.h"
//#define EV1 BGL_UPC_L3_CACHE_HIT
//#define EV2 BGL_UPC_L3_CACHE_MISS_DATA_WILL_BE_REQED_DDR
#define EV1 BGL_UPC_L3_CACHE_MISS_DATA_ALRDY_WAY_DDR
#define EV2 BGL_UPC_PU0_DCURD_WAIT_L3

int main() {

  bgl_perfctr_control_t *hwctrs;
  BGL_PERFCTR_event_t ev;
  int i,n,err,rank;
  int *memarea;

#include "bglpersonality.h"
#include "rts.h"
  {
    BGLPersonality me;
    rts_get_personality(&me,sizeof(me));
    if(me.xCoord != 0 ) goto fine;
    if(me.yCoord != 0 ) goto fine;
    if(me.zCoord != 0 ) goto fine;
  }
  
  
  if(BGL_PERFCTR_MODE_SYNC != bgl_perfctr_init_synch(BGL_PERFCTR_MODE_SYNC))
    abort();

  bgl_perfctr_dump_state(stdout);

  ev.edge=BGL_PERFCTR_UPC_EDGE_RISE;
  ev.num=EV1;
  err=bgl_perfctr_add_event(ev);
  if(err) {
    printf("Add event line %d failed.\n",__LINE__-2);
    exit(1);
  } else printf("One event added. %s\n",BGL_PERFCTR_event_table[EV1].event_name);

  bgl_perfctr_dump_state(stdout);

  ev.num=EV2;
  ev.edge=BGL_PERFCTR_UPC_EDGE_HI;
  err=bgl_perfctr_add_event(ev);
  if(err) {
    printf("Add event line %d failed.\n",__LINE__-2);
    exit(1);
  } else printf("One more event added. %s\n",BGL_PERFCTR_event_table[EV2].event_name);

  bgl_perfctr_dump_state(stdout);

  err=bgl_perfctr_commit();
  if(err) {
    printf("Commit %d failed.\n",__LINE__-2);
    exit(1);
  } else printf("Commit successful.\n");

  bgl_perfctr_dump_state(stdout);

  ev.num=EV1;
  ev.edge=BGL_PERFCTR_UPC_EDGE_RISE;
  err=bgl_perfctr_remove_event(ev);
  if(err) {
    printf("Remove %d failed.\n",__LINE__-2);
    exit(1);
  } else printf("Remove successful.\n");

  bgl_perfctr_dump_state(stdout);

  err=bgl_perfctr_revoke();
  if(err) {
    printf("Revoke %d failed.\n",__LINE__-2);
    exit(1);
  } else printf("Revoke successful.\n");

  bgl_perfctr_dump_state(stdout);

  printf("\n\n----------------------\n\n");

  printf("\n bgl_perfctr_read_counters \n");
  bgl_perfctr_update();
  bgl_perfctr_dump_state(stdout);

  n=1024*1024;
  memarea=(int *) malloc(1024*1024*sizeof(int));
  for(i=0;i<n;i++)
    memarea[i]=n-1;

  printf("\n bgl_perfctr_read_counters again after loop\n");
  bgl_perfctr_update();
  bgl_perfctr_dump_state(stdout);

  for(i=0;i<n;i++)
    memarea[i]-=1;

  printf("\n bgl_perfctr_read_counters again after loop\n");
  bgl_perfctr_update();
  bgl_perfctr_dump_state(stdout);

  if(bgl_perfctr_shutdown())
    abort();

fine:

  return 0;
}
