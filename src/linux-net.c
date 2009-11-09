#include <inttypes.h>
#include "papi.h"
#include "papi_internal.h"
#include "linux-net.h"
#include "papi_memory.h"

#define NETITEMS 13

extern int get_cpu_info(PAPI_hw_info_t * hwinfo);
void net_init_mdi();
int net_init_presets();

enum native_name {
   PNE_LO_RX_PACKETS = 0x40000000,
   PNE_LO_RX_ERRORS,
   PNE_LO_RX_DROPPED,
   PNE_LO_RX_OVERRUNS,
   PNE_LO_RX_FRAME,
   PNE_LO_RX_BYTES,
   PNE_LO_TX_PACKETS,
   PNE_LO_TX_ERRORS,
   PNE_LO_TX_DROPPED,
   PNE_LO_TX_OVERRUNS,
   PNE_LO_TX_CARRIER,
   PNE_LO_TX_BYTES,
   PNE_LO_COLLISIONS,
   PNE_ETH0_RX_PACKETS,
   PNE_ETH0_RX_ERRORS,
   PNE_ETH0_RX_DROPPED,
   PNE_ETH0_RX_OVERRUNS,
   PNE_ETH0_RX_FRAME,
   PNE_ETH0_RX_BYTES,
   PNE_ETH0_TX_PACKETS,
   PNE_ETH0_TX_ERRORS,
   PNE_ETH0_TX_DROPPED,
   PNE_ETH0_TX_OVERRUNS,
   PNE_ETH0_TX_CARRIER,
   PNE_ETH0_TX_BYTES,
   PNE_ETH0_COLLISIONS,
   PNE_ETH1_RX_PACKETS,
   PNE_ETH1_RX_ERRORS,
   PNE_ETH1_RX_DROPPED,
   PNE_ETH1_RX_OVERRUNS,
   PNE_ETH1_RX_FRAME,
   PNE_ETH1_RX_BYTES,
   PNE_ETH1_TX_PACKETS,
   PNE_ETH1_TX_ERRORS,
   PNE_ETH1_TX_DROPPED,
   PNE_ETH1_TX_OVERRUNS,
   PNE_ETH1_TX_CARRIER,
   PNE_ETH1_TX_BYTES,
   PNE_ETH1_COLLISIONS,
   PNE_ETH2_RX_PACKETS,
   PNE_ETH2_RX_ERRORS,
   PNE_ETH2_RX_DROPPED,
   PNE_ETH2_RX_OVERRUNS,
   PNE_ETH2_RX_FRAME,
   PNE_ETH2_RX_BYTES,
   PNE_ETH2_TX_PACKETS,
   PNE_ETH2_TX_ERRORS,
   PNE_ETH2_TX_DROPPED,
   PNE_ETH2_TX_OVERRUNS,
   PNE_ETH2_TX_CARRIER,
   PNE_ETH2_TX_BYTES,
   PNE_ETH2_COLLISIONS,
   PNE_ETH3_RX_PACKETS,
   PNE_ETH3_RX_ERRORS,
   PNE_ETH3_RX_DROPPED,
   PNE_ETH3_RX_OVERRUNS,
   PNE_ETH3_RX_FRAME,
   PNE_ETH3_RX_BYTES,
   PNE_ETH3_TX_PACKETS,
   PNE_ETH3_TX_ERRORS,
   PNE_ETH3_TX_DROPPED,
   PNE_ETH3_TX_OVERRUNS,
   PNE_ETH3_TX_CARRIER,
   PNE_ETH3_TX_BYTES,
   PNE_ETH3_COLLISIONS,
   PNE_ETH4_RX_PACKETS,
   PNE_ETH4_RX_ERRORS,
   PNE_ETH4_RX_DROPPED,
   PNE_ETH4_RX_OVERRUNS,
   PNE_ETH4_RX_FRAME,
   PNE_ETH4_RX_BYTES,
   PNE_ETH4_TX_PACKETS,
   PNE_ETH4_TX_ERRORS,
   PNE_ETH4_TX_DROPPED,
   PNE_ETH4_TX_OVERRUNS,
   PNE_ETH4_TX_CARRIER,
   PNE_ETH4_TX_BYTES,
   PNE_ETH4_COLLISIONS
};

NET_native_event_entry_t net_native_table[] = {
	{{1, {NETPATH}},
   "LO_RX_PACKETS",
   "LO_RX_PACKETS",
   },
   {{2, {NETPATH}},
   "LO_RX_ERRORS",
   "LO_RX_ERRORS",
   },
   {{3, {NETPATH}},
   "LO_RX_DROPPED",
   "LO_RX_DROPPED",
   },
   {{4, {NETPATH}},
   "LO_RX_OVERRUNS",
   "LO_RX_OVERRUNS",
   },
   {{5, {NETPATH}},
   "LO_RX_FRAME",
   "LO_RX_FRAME",
   },
   {{6, {NETPATH}},
   "LO_RX_BYTES",
   "LO_RX_BYTES",
   },
   {{7, {NETPATH}},
   "LO_TX_PACKETS",
   "LO_TX_PACKETS",
   },
   {{8, {NETPATH}},
   "LO_TX_ERRORS",
   "LO_TX_ERRORS",
   },
   {{9, {NETPATH}},
   "LO_TX_DROPPED",
   "LO_TX_DROPPED",
   },
   {{10, {NETPATH}},
   "LO_TX_OVERRUNS",
   "LO_TX_OVERRUNS",
   },
   {{11, {NETPATH}},
   "LO_TX_CARRIER",
   "LO_TX_CARRIER",
   },
   {{12, {NETPATH}},
   "LO_TX_BYTES",
   "LO_TX_BYTES",
   },
   {{13, {NETPATH}},
   "LO_COLLISIONS",
   "LO_COLLISIONS",
   },
   {{14, {NETPATH}},
   "ETH0_RX_PACKETS",
   "ETH0_RX_PACKETS",
   },
   {{15, {NETPATH}},
   "ETH0_RX_ERRORS",
   "ETH0_RX_ERRORS",
   },
   {{16, {NETPATH}},
   "ETH0_RX_DROPPED",
   "ETH0_RX_DROPPED",
   },
   {{17, {NETPATH}},
   "ETH0_RX_OVERRUNS",
   "ETH0_RX_OVERRUNS",
   },
   {{18, {NETPATH}},
   "ETH0_RX_FRAME",
   "ETH0_RX_FRAME",
   },
   {{19, {NETPATH}},
   "ETH0_RX_BYTES",
   "ETH0_RX_BYTES",
   },
   {{20, {NETPATH}},
   "ETH0_TX_PACKETS",
   "ETH0_TX_PACKETS",
   },
   {{21, {NETPATH}},
   "ETH0_TX_ERRORS",
   "ETH0_TX_ERRORS",
   },
   {{22, {NETPATH}},
   "ETH0_TX_DROPPED",
   "ETH0_TX_DROPPED",
   },
   {{23, {NETPATH}},
   "ETH0_TX_OVERRUNS",
   "ETH0_TX_OVERRUNS",
   },
   {{24, {NETPATH}},
   "ETH0_TX_CARRIER",
   "ETH0_TX_CARRIER",
   },
   {{25, {NETPATH}},
   "ETH0_TX_BYTES",
   "ETH0_TX_BYTES",
   },
   {{26, {NETPATH}},
   "ETH0_COLLISIONS",
   "ETH0_COLLISIONS",
   },
   {{27, {NETPATH}},
   "ETH1_RX_PACKETS",
   "ETH1_RX_PACKETS",
   },
   {{28, {NETPATH}},
   "ETH1_RX_ERRORS",
   "ETH1_RX_ERRORS",
   },
   {{29, {NETPATH}},
   "ETH1_RX_DROPPED",
   "ETH1_RX_DROPPED",
   },
   {{30, {NETPATH}},
   "ETH1_RX_OVERRUNS",
   "ETH1_RX_OVERRUNS",
   },
   {{31, {NETPATH}},
   "ETH1_RX_FRAME",
   "ETH1_RX_FRAME",
   },
   {{32, {NETPATH}},
   "ETH1_RX_BYTES",
   "ETH1_RX_BYTES",
   },
   {{33, {NETPATH}},
   "ETH1_TX_PACKETS",
   "ETH1_TX_PACKETS",
   },
   {{34, {NETPATH}},
   "ETH1_TX_ERRORS",
   "ETH1_TX_ERRORS",
   },
   {{35, {NETPATH}},
   "ETH1_TX_DROPPED",
   "ETH1_TX_DROPPED",
   },
   {{36, {NETPATH}},
   "ETH1_TX_OVERRUNS",
   "ETH1_TX_OVERRUNS",
   },
   {{37, {NETPATH}},
   "ETH1_TX_CARRIER",
   "ETH1_TX_CARRIER",
   },
   {{38, {NETPATH}},
   "ETH1_TX_BYTES",
   "ETH1_TX_BYTES",
   },
   {{39, {NETPATH}},
   "ETH1_COLLISIONS",
   "ETH1_COLLISIONS",
   },
   {{40, {NETPATH}},
   "ETH2_RX_PACKETS",
   "ETH2_RX_PACKETS",
   },
   {{41, {NETPATH}},
   "ETH2_RX_ERRORS",
   "ETH2_RX_ERRORS",
   },
   {{42, {NETPATH}},
   "ETH2_RX_DROPPED",
   "ETH2_RX_DROPPED",
   },
   {{43, {NETPATH}},
   "ETH2_RX_OVERRUNS",
   "ETH2_RX_OVERRUNS",
   },
   {{44, {NETPATH}},
   "ETH2_RX_FRAME",
   "ETH2_RX_FRAME",
   },
   {{45, {NETPATH}},
   "ETH2_RX_BYTES",
   "ETH2_RX_BYTES",
   },
   {{46, {NETPATH}},
   "ETH2_TX_PACKETS",
   "ETH2_TX_PACKETS",
   },
   {{47, {NETPATH}},
   "ETH2_TX_ERRORS",
   "ETH2_TX_ERRORS",
   },
   {{48, {NETPATH}},
   "ETH2_TX_DROPPED",
   "ETH2_TX_DROPPED",
   },
   {{49, {NETPATH}},
   "ETH2_TX_OVERRUNS",
   "ETH2_TX_OVERRUNS",
   },
   {{50, {NETPATH}},
   "ETH2_TX_CARRIER",
   "ETH2_TX_CARRIER",
   },
   {{51, {NETPATH}},
   "ETH2_TX_BYTES",
   "ETH2_TX_BYTES",
   },
   {{52, {NETPATH}},
   "ETH2_COLLISIONS",
   "ETH2_COLLISIONS",
   },
   {{53, {NETPATH}},
   "ETH3_RX_PACKETS",
   "ETH3_RX_PACKETS",
   },
   {{54, {NETPATH}},
   "ETH3_RX_ERRORS",
   "ETH3_RX_ERRORS",
   },
   {{55, {NETPATH}},
   "ETH3_RX_DROPPED",
   "ETH3_RX_DROPPED",
   },
   {{56, {NETPATH}},
   "ETH3_RX_OVERRUNS",
   "ETH3_RX_OVERRUNS",
   },
   {{57, {NETPATH}},
   "ETH3_RX_FRAME",
   "ETH3_RX_FRAME",
   },
   {{58, {NETPATH}},
   "ETH3_RX_BYTES",
   "ETH3_RX_BYTES",
   },
   {{59, {NETPATH}},
   "ETH3_TX_PACKETS",
   "ETH3_TX_PACKETS",
   },
   {{60, {NETPATH}},
   "ETH3_TX_ERRORS",
   "ETH3_TX_ERRORS",
   },
   {{61, {NETPATH}},
   "ETH3_TX_DROPPED",
   "ETH3_TX_DROPPED",
   },
   {{62, {NETPATH}},
   "ETH3_TX_OVERRUNS",
   "ETH3_TX_OVERRUNS",
   },
   {{63, {NETPATH}},
   "ETH3_TX_CARRIER",
   "ETH3_TX_CARRIER",
   },
   {{64, {NETPATH}},
   "ETH3_TX_BYTES",
   "ETH3_TX_BYTES",
   },
   {{65, {NETPATH}},
   "ETH3_COLLISIONS",
   "ETH3_COLLISIONS",
   },
   {{66, {NETPATH}},
   "ETH4_RX_PACKETS",
   "ETH4_RX_PACKETS",
   },
   {{67, {NETPATH}},
   "ETH4_RX_ERRORS",
   "ETH4_RX_ERRORS",
   },
   {{68, {NETPATH}},
   "ETH4_RX_DROPPED",
   "ETH4_RX_DROPPED",
   },
   {{69, {NETPATH}},
   "ETH4_RX_OVERRUNS",
   "ETH4_RX_OVERRUNS",
   },
   {{70, {NETPATH}},
   "ETH4_RX_FRAME",
   "ETH4_RX_FRAME",
   },
   {{71, {NETPATH}},
   "ETH4_RX_BYTES",
   "ETH4_RX_BYTES",
   },
   {{72, {NETPATH}},
   "ETH4_TX_PACKETS",
   "ETH4_TX_PACKETS",
   },
   {{73, {NETPATH}},
   "ETH4_TX_ERRORS",
   "ETH4_TX_ERRORS",
   },
   {{74, {NETPATH}},
   "ETH4_TX_DROPPED",
   "ETH4_TX_DROPPED",
   },
   {{75, {NETPATH}},
   "ETH4_TX_OVERRUNS",
   "ETH4_TX_OVERRUNS",
   },
   {{76, {NETPATH}},
   "ETH4_TX_CARRIER",
   "ETH4_TX_CARRIER",
   },
   {{77, {NETPATH}},
   "ETH4_TX_BYTES",
   "ETH4_TX_BYTES",
   },
   {{78, {NETPATH}},
   "ETH4_COLLISIONS",
   "ETH4_COLLISIONS",
   },
	{{0, {0}}, "", ""}
};

long long _papi_hwd_net_register_start[NET_MAX_COUNTERS];
long long _papi_hwd_net_register[NET_MAX_COUNTERS];

/*
 * Substrate setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int NET_init_substrate()
{
   int retval=PAPI_OK, i;

   for(i=0;i<NET_MAX_COUNTERS;i++){
     _papi_hwd_net_register_start[i] = -1;
     _papi_hwd_net_register[i] = -1;
   }
   /* Internal function, doesn't necessarily need to be a function */
   net_init_mdi();

   /* Internal function, doesn't necessarily need to be a function */
   net_init_presets();

   return(retval);
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * for the presets are setup here.
 */
hwi_search_t net_preset_map[] = {
   {0, {0, {PAPI_NULL, PAPI_NULL}, {0,}}}
};


int net_init_presets(){
  return (_papi_hwi_setup_all_presets(net_preset_map, NULL));
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * is setup in _papi_hwd_init_substrate.  Below is some, but not
 * all of the values that will need to be setup.  For a complete
 * list check out papi_mdi_t, though some of the values are setup
 * and used above the substrate level.
 */
void net_init_mdi(){
/* 
   get_cpu_info(&_papi_hwi_system_info.hw_info);
   _papi_hwi_system_info.num_cntrs = MAX_COUNTERS;
   _papi_hwi_system_info.supports_program = 0;
   _papi_hwi_system_info.supports_write = 0;
   _papi_hwi_system_info.supports_hw_overflow = 0;
   _papi_hwi_system_info.supports_hw_profile = 0;
   _papi_hwi_system_info.supports_multiple_threads = 0;
   _papi_hwi_system_info.supports_64bit_counters = 0;
   _papi_hwi_system_info.supports_attach = 0;
   _papi_hwi_system_info.supports_real_usec = 0;
   _papi_hwi_system_info.supports_real_cyc = 0;
   _papi_hwi_system_info.supports_virt_usec = 0;
   _papi_hwi_system_info.supports_virt_cyc = 0;
   _papi_hwi_system_info.size_machdep = sizeof(hwd_control_state_t);
*/}


/*
 * This is called whenever a thread is initialized
 */
int NET_init(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

int NET_shutdown(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int NET_init_control_state(hwd_control_state_t *ptr){
   return PAPI_OK;
}

int NET_update_control_state(hwd_control_state_t *ptr, NativeInfo_t *native, int count, hwd_context_t *ctx){
   int i, index;

   for (i = 0; i < count; i++) {
     index = native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
     native[i].ni_position = net_native_table[index].resources.selector-1;
   }
   return(PAPI_OK);
}

int checkif(char *name)
{
  if(strcmp(name, "lo")==0)
    return 0;
  if(strcmp(name, "eth0")==0)
    return 1;
  if(strcmp(name, "eth1")==0)
    return 2;
  if(strcmp(name, "eth2")==0)
    return 3;
  if(strcmp(name, "eth3")==0)
    return 4;
  if(strcmp(name, "eth4")==0)
    return 5;
  return -1;
}

int checkstate(char *name)
{
  if(strcmp(name, "RX")==0)
    return 0;
  if(strcmp(name, "TX")==0)
    return 6;
  return -1;
}

void update_counters(long long *counters, int curstate, int curstart, char *input)
{
  char *number = NULL;

  number = strstr(input, ":"); 
  if(number){
    number[0] = '\0';
    number = number+1;
  }

  if(number){
    if(strcmp(input, "packets") == 0){
       counters[curstart*NETITEMS + curstate] = atoll(number);
    }
    else if(strcmp(input, "errors") == 0){
       counters[curstart*NETITEMS + curstate + 1] = atoll(number);
    }
    else if(strcmp(input, "dropped") == 0){
       counters[curstart*NETITEMS + curstate + 2] = atoll(number);
    }
    else if(strcmp(input, "overruns") == 0){
       counters[curstart*NETITEMS + curstate + 3] = atoll(number);
    }
    else if(strcmp(input, "frame") == 0 || strcmp(input, "carrier") == 0){
       counters[curstart*NETITEMS + curstate + 4] = atoll(number);
    }
    else if(strcmp(input, "bytes") == 0){
       counters[curstart*NETITEMS + curstate + 5] = atoll(number);
    }
    else if(strcmp(input, "collisions") == 0){
       counters[curstart*NETITEMS + 12] = atoll(number);
    }
    else
      return;
    /*printf("  count(%lld, %s, %s)\n",  atoll(number), number, input);*/
  }


}

int read_net_counters(long long *counters)
{
   FILE *fp;
   char line[NETLINELEN], lastchar;
   char *fields=NULL;
   int i, tf, lno, curstart = -1, cname = 0, state, curstate = -1;

   fp=popen(NETPATH, "r");
   if(!fp){
      perror("popen");
      return(PAPI_ESBSTR);
   }

   lno=0;
   while(fgets(line, NETLINELEN, fp)){
      lastchar = ' ';
      tf=0;
      for(i=0; line[i] != '\0' && i < NETLINELEN-1; i++){
         if(isspace(line[i])){
            if(lastchar != ' '){
               lastchar = line[i];
               line[i] = '\0';
               if(!cname){
                  curstart = checkif(fields);
                  if(curstart >= 0){
                     cname++;
                     curstate = -1;
                  }
               }
               else{
                  state = checkstate(fields);
                  if(state >= 0)
                     curstate = state;
                  update_counters(counters, curstate, curstart, fields);
               }
            }
         }
         else{
            if(isspace(lastchar)){
               fields = line+i;
               tf++;
            }
            lastchar = line[i];
         }
      }
      if(!tf){
         cname = 0;
      }
   }
   
   fclose(fp);

   /*for(i=0;i<128;i++)
      printf("%d  %d  %lld\n", i/13, i%13, counters[i]);   */
   return(PAPI_OK);
}

int NET_start(hwd_context_t *ctx, hwd_control_state_t *ctrl){
   read_net_counters(_papi_hwd_net_register_start);
   memcpy(_papi_hwd_net_register, _papi_hwd_net_register_start, NET_MAX_COUNTERS*sizeof(long long));

   return(PAPI_OK);
}


int NET_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long **events, int flags)
{
    int i;

    read_net_counters(_papi_hwd_net_register); 
    for(i = 0; i < NET_MAX_COUNTERS; i++){
       ((NET_control_state_t *)ctrl)->counts[i] = _papi_hwd_net_register[i] - _papi_hwd_net_register_start[i];
      /*printf("%d  %lld\n", i, ctrl->counts[i]);*/
    }
    *events=((NET_control_state_t *)ctrl)->counts;
    return(PAPI_OK);
}

int NET_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   int i;

   read_net_counters(_papi_hwd_net_register);
   for(i = 0; i < NET_MAX_COUNTERS; i++){
      ((NET_control_state_t *)ctrl)->counts[i] = _papi_hwd_net_register[i] - _papi_hwd_net_register_start[i];
/*      printf("%d  %lld\n", i, ctrl->counts[i]);*/
   }

   return(PAPI_OK);
}

int NET_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   NET_start(ctx, ctrl);

   return(PAPI_OK);
}

int NET_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *from)
{
   return(PAPI_OK);
}

/*
 * Functions for setting up various options
 */

/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int NET_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
  return(PAPI_OK);
}

/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
int NET_set_domain(hwd_control_state_t *cntrl, int domain) 
{
  int found = 0;
  if ( PAPI_DOM_USER & domain ){
        found = 1;
  }
  if ( PAPI_DOM_KERNEL & domain ){
        found = 1;
  }
  if ( PAPI_DOM_OTHER & domain ){
        found = 1;
  }
  if ( !found )
        return(PAPI_EINVAL);
   return(PAPI_OK);
}

/* 
 * Timing Routines
 * These functions should return the highest resolution timers available.
 */
/*long long _papi_hwd_get_real_usec(void)
{
   return(1);
}

long long _papi_hwd_get_real_cycles(void)
{
   return(1);
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   return(1);
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
   return(1);
}
*/

int NET_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
   strncpy(name, net_native_table[EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK].name, len);
   return(PAPI_OK);
}

int NET_ntv_code_to_descr(unsigned int EventCode, char *name, int len)
{
   strncpy(name, net_native_table[EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK].description, len);
   return(PAPI_OK);
}

int NET_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   memcpy(( NET_register_t *) bits, &(net_native_table[EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK].resources), sizeof(NET_register_t)); /* it is not right, different type */
   return (PAPI_OK);
}

int NET_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count)
{
  return(1);
}


int NET_ntv_enum_events(unsigned int *EventCode, int modifier)
{
  int cidx = PAPI_COMPONENT_INDEX(*EventCode);

  if (modifier == PAPI_ENUM_FIRST) {
    /* assumes first native event is always 0x4000000 */
    *EventCode = PAPI_NATIVE_MASK|PAPI_COMPONENT_MASK(cidx); 
    return (PAPI_OK);
  }

   if (modifier == PAPI_ENUM_EVENTS) {
      int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

      if (net_native_table[index + 1].resources.selector) {
         *EventCode = *EventCode + 1;
         return (PAPI_OK);
      } else
         return (PAPI_ENOEVNT);
   } 
   else
      return (PAPI_EINVAL);
}

/*
 * Shared Library Information and other Information Functions
 */
/*int _papi_hwd_update_shlib_info(void){
  return(PAPI_OK);
}
*/
papi_vector_t _net_vector = {
    .cmp_info = {
      /* default component information (unspecified values are initialized to 0) */
      .name = "$Id$",
      .version = "$Revision$",
      .num_mpx_cntrs =	PAPI_MPX_DEF_DEG,
      .num_cntrs =	NET_MAX_COUNTERS,
      .default_domain =	PAPI_DOM_USER,
      .available_domains =	PAPI_DOM_USER,
      .default_granularity =	PAPI_GRN_THR,
      .available_granularities = PAPI_GRN_THR,
      .hardware_intr_sig =    PAPI_INT_SIGNAL,

      /* component specific cmp_info initializations */
      .fast_real_timer =	0,
      .fast_virtual_timer =	0,
      .attach =		0,
      .attach_must_ptrace =	0,
      .available_domains =	PAPI_DOM_USER|PAPI_DOM_KERNEL,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
	.context =		sizeof(NET_context_t),
	.control_state =	sizeof(NET_control_state_t),
	.reg_value =		sizeof(NET_register_t),
	.reg_alloc =		sizeof(NET_reg_alloc_t),
    },
    /* function pointers in this component */
    .init =	NET_init,
    .init_substrate =	NET_init_substrate,
    .init_control_state =	NET_init_control_state,
    .start =			NET_start,
    .stop =			NET_stop,
    .read =			NET_read,
    .shutdown =			NET_shutdown,
    .ctl =			NET_ctl,
    .update_control_state =	NET_update_control_state,
    .set_domain =		NET_set_domain,
    .reset =			NET_reset,
/*    .set_overflow =		_p3_set_overflow,
    .stop_profiling =		_p3_stop_profiling,*/
    .ntv_enum_events =		NET_ntv_enum_events,
    .ntv_code_to_name =		NET_ntv_code_to_name,
    .ntv_code_to_descr =	NET_ntv_code_to_descr,
    .ntv_code_to_bits =		NET_ntv_code_to_bits,
    .ntv_bits_to_info =		NET_ntv_bits_to_info
};


