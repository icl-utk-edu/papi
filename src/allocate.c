/* this file contains utility functions for counter allocation 
   created by Haihang You < you@cs.utk.edu >
*/

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

#include "allocate.h"

extern hwd_preset_t _papi_hwd_preset_map[PAPI_MAX_PRESET_EVENTS];
extern pm_info_t pminfo;

/* this function try to find out whether native events contained by this preset have already been mapped. If it is, mapping is done */
int _papi_hwd_event_precheck(hwd_control_state_t *tmp_state, unsigned int EventCode, EventInfo_t *out, void *v)
{
	int metric, i, j, found_native=0, hwd_idx=0;
	hwd_preset_t *this_preset;
	hwd_native_t *this_native;
	int counter_mapping[MAX_COUNTERS];
	unsigned char selector;
	  
	/* to find first empty slot */
  hwd_idx=out->counter_index;
		
	/* preset event */
	if(EventCode & PRESET_MASK){
		this_preset=(hwd_preset_t *)v;
		for( metric=0; metric<this_preset->metric_count; metric++){
			for (j=0; j<MAX_COUNTERS; j++) {
				if (tmp_state->master_selector & (1<<j) && this_preset->selector[metric] & (1<<j)) {
					if (tmp_state->counter_cmd.events[j] == this_preset->counter_cmd[metric][j])
						break;
			  	}
			}
		
			if(j<MAX_COUNTERS){ /* found mapping */ 
				counter_mapping[metric]=j;
				for(i=0;i<tmp_state->native_idx;i++)
					if(j==tmp_state->native[i].position){
						tmp_state->native[i].link++;
						break;
					}
			}
			else{
				return 0;
			}		
		}
		
		/* successfully found mapping. Write to EventInfo_t *out, return 1 */
		tmp_state->allevent[hwd_idx]=EventCode;
		selector=0;
		for(j=0;j<metric;j++){
			tmp_state->emap[hwd_idx][j]=counter_mapping[j];
			selector|=1<<counter_mapping[j];
		}
	
		/* update EventInfo_t *out */
		out->event_code = EventCode;
		out->bits.selector = selector;
		out->derived = this_preset->derived;
		out->counter_index = tmp_state->emap[hwd_idx][0];
		tmp_state->hwd_idx_a++;
	
		return 1;
	}
	else{
		this_native=(hwd_native_t *)v;

		/* to find the native event from the native events list */
		for(i=0; i<tmp_state->native_idx;i++){
			if(strcmp(this_native->name, tmp_state->native[i].name)==0){
				found_native=1; printf("found native name: %s\n", this_native->name);
				break;
			}
		}
		if(found_native){
			tmp_state->allevent[hwd_idx]=EventCode;
			tmp_state->emap[hwd_idx][0]=tmp_state->native[i].position;
			tmp_state->native[i].link++;
			/* update EventInfo_t *out */
			out->event_code = EventCode;
			out->bits.selector |= 1<<tmp_state->native[i].position;
			out->derived = NOT_DERIVED;
			out->counter_index = tmp_state->emap[hwd_idx][0];
			tmp_state->hwd_idx_a++;
			return 1;
		}
		else{
			return 0;
		}
	}
}	  



/* this function is called after mapping is done */
int _papi_hwd_event_mapafter(hwd_control_state_t *tmp_state, int index, EventInfo_t *out)
{
	int metric, j;
	hwd_preset_t *this_preset;
	int counter_mapping[MAX_COUNTERS];
	unsigned char selector;
	unsigned int EventCode;
	  
  	EventCode=tmp_state->allevent[index];
	/* preset */
	if(EventCode & PRESET_MASK){
		this_preset = &(_papi_hwd_preset_map[EventCode & PRESET_AND_MASK]);
		for( metric=0; metric<this_preset->metric_count; metric++){
			for (j=0; j<MAX_COUNTERS; j++) {
				if (tmp_state->master_selector & (1<<j) && this_preset->selector[metric] & (1<<j)) {
					if (tmp_state->counter_cmd.events[j] == this_preset->counter_cmd[metric][j])
						break;
			  	}
			}
		
			if(j<MAX_COUNTERS){ /* found mapping */
				counter_mapping[metric]=j;
			}
			else{
				return 0;
			}		
		}
	
		/* successfully found mapping. Write to EventInfo_t *out, return 1 */
		selector=0;
		for(j=0;j<metric;j++){
			tmp_state->emap[index][j]=counter_mapping[j];
			selector|=1<<counter_mapping[j];
		}
	
		out->bits.selector = selector;
		out->counter_index = tmp_state->emap[index][0];
		return 1;
	}
	else{
		pm_events_t *pe;
		int found_native=0, hwcntr_num, i;
		unsigned int event_code;
		char name[PAPI_MAX_STR_LEN];
		
		/* to get pm event name */ 
		event_code=EventCode>>8;
		hwcntr_num = EventCode & 0xff;
		pe=pminfo.list_events[hwcntr_num];
		for(i=0;i<pminfo.maxevents[hwcntr_num];i++, pe++){
			if(pe->event_id==event_code){
				strcpy(name, pe->short_name); /* will be found */
				break;
			}
		}
		
		/* to find the native event from the native events list */
		for(i=0; i<MAX_COUNTERS;i++){
			if(strcmp(name, tmp_state->native[i].name)==0){
				found_native=1;
				break;
			}
		}
		if(found_native){
			tmp_state->emap[index][0]=tmp_state->native[i].position;
	
			/* update EventInfo_t *out */
			out->bits.selector |= 1<<tmp_state->native[i].position;
			out->counter_index = tmp_state->emap[index][0];
			return 1;
		}
		else{
			return 0;
		}
	}
}	  

int do_counter_mapping(hwd_native_t *event_list, int size)
{
	int i,j;
	hwd_native_t *queue[MAX_COUNTERS];
	int head, tail;
	
	/* if the event competes 1 counter only, it has priority, map it */
	head=0;
	tail=0;
	for(i=0;i<size;i++){ /* push rank=1 into queue */
		event_list[i].mod=-1;
		if(event_list[i].rank==1){
			queue[tail]=&event_list[i];
			event_list[i].mod=i;
			tail++;
		}
	}
	
	while(head<tail){
		for(i=0;i<size;i++){
			if(i!=(*queue[head]).mod){
				if(event_list[i].selector & (*queue[head]).selector){
					if(event_list[i].rank==1){
						return 0; /* mapping fail, 2 events compete 1 counter only */
					}
					else{
						event_list[i].selector ^= (*queue[head]).selector;
						event_list[i].rank--;
						if(event_list[i].rank==1){
							queue[tail]=&event_list[i];
							event_list[i].mod=i;
							tail++;
						}
					}
				}
			}
		}
		head++;
	}
	if(tail==size){
		return 1; /* successfully mapped */
	}
	else{
		hwd_native_t rest_event_list[MAX_COUNTERS];
		hwd_native_t copy_rest_event_list[MAX_COUNTERS];
		
		j=0;
		for(i=0;i<size;i++){
			if(event_list[i].mod<0){
				memcpy(copy_rest_event_list+j, event_list+i, sizeof(hwd_native_t));
				copy_rest_event_list[j].mod=i;
				j++;
			}
		}
		
		memcpy(rest_event_list, copy_rest_event_list, sizeof(hwd_native_t)*(size-tail));
		
		for(i=0;i<MAX_COUNTERS;i++){
			if(rest_event_list[0].selector & (1<<i)){ /* pick first event on the list, set 1 to 0, to see whether there is an answer */
				for(j=0;j<size-tail;j++){
					if(j==0){
						rest_event_list[j].selector = 1<<i;
						rest_event_list[j].rank = 1;
					}
					else{
						if(rest_event_list[j].selector & (1<<i)){
							rest_event_list[j].selector ^= 1<<i;
							rest_event_list[j].rank--;
						}
					}
				}
				if(do_counter_mapping(rest_event_list, size-tail))
					break;
				
				memcpy(rest_event_list, copy_rest_event_list, sizeof(hwd_native_t)*(size-tail));
			}
		}
		if(i==MAX_COUNTERS){
			return 0; /* fail to find mapping */
		}
		for(i=0;i<size-tail;i++){
			event_list[copy_rest_event_list[i].mod].selector=rest_event_list[i].selector;
		}
		return 1;		
	}
}	
	

/* this function will be called when there are counters available, (void *) is the pointer to adding event structure
   (hwd_preset_t *)  or (hwd_native_t *)  
*/      
int _papi_hwd_counter_mapping(hwd_control_state_t *tmp_state, unsigned int EventCode, EventInfo_t *out, void *v)
{
  hwd_preset_t *this_preset;
  hwd_native_t *this_native;
  unsigned char selector;
  int metric, i, j, k, getname=1, hwd_idx=0, triger=0, natNum;
  pm_events_t *pe;
  int tr;
  EventInfo_t *zeroth;


  hwd_idx=out->counter_index;
  
  tmp_state->allevent[hwd_idx]=EventCode;
  selector=0;
  natNum=tmp_state->native_idx;
  
  if(EventCode & PRESET_MASK){
	this_preset=(hwd_preset_t *)v;
	
	
	/* try to find unmapped native events, then put then on to native list */ 
	for( metric=0; metric<this_preset->metric_count; metric++){
		for (j=0; j<MAX_COUNTERS; j++) {
			if (tmp_state->master_selector & (1<<j) && this_preset->selector[metric] & (1<<j)) {
				if (tmp_state->counter_cmd.events[j] == this_preset->counter_cmd[metric][j]){
					selector |= 1<<j;
					tmp_state->emap[hwd_idx][metric]=j;
					if(triger){
						for(i=0;i<natNum;i++)
							if(j==tmp_state->native[i].position){
								tmp_state->native[i].link++;
								break;
							}
					}
					break;
				}
		  	}
		}
		if(j==MAX_COUNTERS){ /* not found mapping from existed mapped native events */
			if(tmp_state->native_idx==MAX_COUNTERS){ /* can not do mapping, no counter available */
				return 0;
			}
			triger=1;
			tmp_state->native[tmp_state->native_idx].selector=this_preset->selector[metric];
			tmp_state->native[tmp_state->native_idx].rank=this_preset->rank[metric];
			getname=1;
			
			for(i=0;i<MAX_COUNTERS;i++){
				tmp_state->native[tmp_state->native_idx].counter_cmd[i]=this_preset->counter_cmd[metric][i];
				if(getname && tmp_state->native[tmp_state->native_idx].counter_cmd[i]!=COUNT_NOTHING){
					/* get the native event's name */
					pe=pminfo.list_events[i];
					/*printf("tmp_state->native[%d].counter_cmd[%d]=%d\n", tmp_state->native_idx, i, tmp_state->native[tmp_state->native_idx].counter_cmd[i]  );*/
					for(k=0;k<pminfo.maxevents[i];k++, pe++){ 
						if(pe->event_id==tmp_state->native[tmp_state->native_idx].counter_cmd[i]){
							strcpy(tmp_state->native[tmp_state->native_idx].name, pe->short_name);
							tmp_state->native[tmp_state->native_idx].link++;
							getname=0;
							break;
						}
					}
				}
			}
			tmp_state->native_idx++;
		}
	}
  }
  else{
	
  	this_native=(hwd_native_t *)v;
	this_native->link++;
	memcpy(tmp_state->native+tmp_state->native_idx, this_native, sizeof(hwd_native_t));
	tmp_state->native_idx++;
  }

  { /* not successfully mapped, but have enough slots for events */
  	hwd_native_t event_list[MAX_COUNTERS];
	
	memcpy(event_list, tmp_state->native, sizeof(hwd_native_t)*(tmp_state->native_idx));
	
	if(do_counter_mapping(event_list, tmp_state->native_idx)){ /* successfully mapped */
		/* update tmp_state, reset... */
		tmp_state->master_selector=0;
		for (i = 0; i <MAX_COUNTERS; i++) {
		    tmp_state->counter_cmd.events[i] = COUNT_NOTHING;
		}
		
		for(i=0;i<tmp_state->native_idx;i++){
			tmp_state->master_selector |= event_list[i].selector;
			/* update tmp_state->native->position */
			tmp_state->native[i].position=get_avail_hwcntr_num(event_list[i].selector); 
			/* update tmp_state->counter_cmd */
			tmp_state->counter_cmd.events[tmp_state->native[i].position] = tmp_state->native[i].counter_cmd[tmp_state->native[i].position];
		}
		
		/* copy new value to out */
		zeroth = out->head;
		j=0;
		for(i=0;i<=tmp_state->hwd_idx_a;i++){
			while(tmp_state->allevent[j]==COUNT_NOTHING)
				j++;
			tr=_papi_hwd_event_mapafter(tmp_state, j, zeroth+j);
			if(!tr)
				printf("************************not possible!  j=%d\n", j);
			j++;
		}
		
		out->event_code = EventCode;
		if(EventCode & PRESET_MASK)
			out->derived = this_preset->derived;
		else
			out->derived = NOT_DERIVED;
		/*out->index=tmp_state->hwd_idx_a;*/
	
		tmp_state->hwd_idx++;
		tmp_state->hwd_idx_a++;
		return 1;
	}
	else{
		DBG((stderr,"--------fail 1: %x  %d \n",EventCode, tmp_state->hwd_idx_a));
		return 0;
	}
  }

}

int get_avail_hwcntr_num(int cntr_avail_bits)
{
  int tmp = 0, i = MAX_COUNTERS - 1;
 
  while (i)
    {
      tmp = (1 << i) & cntr_avail_bits;
      if (tmp)
	return(i);
      i--;
    }
  return(0);
}

void print_state(hwd_control_state_t *s)
{
  int i;
  
  fprintf(stderr,"\n\n-----------------------------------------\nmaster_selector 0x%x\n",s->master_selector);
  for(i=0;i<MAX_COUNTERS;i++){
  	if(s->master_selector & (1<<i)) fprintf(stderr, "  1  ");
	else fprintf(stderr, "  0  ");
  }
  fprintf(stderr,"\nnative_event_name       %12s %12s %12s %12s %12s %12s %12s %12s\n",s->native[0].name,s->native[1].name,
    s->native[2].name,s->native[3].name,s->native[4].name,s->native[5].name,s->native[6].name,s->native[7].name);
  fprintf(stderr,"native_event_selectors    %12d %12d %12d %12d %12d %12d %12d %12d\n",s->native[0].selector,s->native[1].selector,
    s->native[2].selector,s->native[3].selector,s->native[4].selector,s->native[5].selector,s->native[6].selector,s->native[7].selector);
  fprintf(stderr,"native_event_position     %12d %12d %12d %12d %12d %12d %12d %12d\n",s->native[0].position,s->native[1].position,
    s->native[2].position,s->native[3].position,s->native[4].position,s->native[5].position,s->native[6].position,s->native[7].position);
  fprintf(stderr,"counters                  %12d %12d %12d %12d %12d %12d %12d %12d\n",s->counter_cmd.events[0],
    s->counter_cmd.events[1],s->counter_cmd.events[2],s->counter_cmd.events[3],
    s->counter_cmd.events[4],s->counter_cmd.events[5],s->counter_cmd.events[6],
    s->counter_cmd.events[7]);
  fprintf(stderr,"native links              %12d %12d %12d %12d %12d %12d %12d %12d\n",s->native[0].link,s->native[1].link,
    s->native[2].link,s->native[3].link,s->native[4].link,s->native[5].link,s->native[6].link,s->native[7].link);
  for(i=0;i<s->hwd_idx_a;i++){
  	fprintf(stderr,"event_codes %x\n",s->allevent[i]);
  }
}
