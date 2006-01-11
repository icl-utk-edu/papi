#define IN_SUBSTRATE

#include "papi.h"
#include "papi_internal.h"
#include "linux-mx.h"
#include "papi_protos.h"
#include "papi_vector.h"

static void init_mdi();
static int init_presets();

int sidx;

enum native_name {
   PNE_LANAI_UPTIME = 0x40000000,
   PNE_COUNTERS_UPTIME,
   PNE_BAD_CRC8,
   PNE_BAD_CRC32,
   PNE_UNSTRIPPED_ROUTE,
   PNE_PKT_DESC_INVALID,
   PNE_RECV_PKT_ERRORS,
   PNE_PKT_MISROUTED,
   PNE_DATA_SRC_UNKNOWN,
   PNE_DATA_BAD_ENDPT,
   PNE_DATA_ENDPT_CLOSED,
   PNE_DATA_BAD_SESSION,
   PNE_PUSH_BAD_WINDOW,
   PNE_PUSH_DUPLICATE,
   PNE_PUSH_OBSOLETE,
   PNE_PUSH_RACE_DRIVER,
   PNE_PUSH_BAD_SEND_HANDLE_MAGIC,
   PNE_PUSH_BAD_SRC_MAGIC,
   PNE_PULL_OBSOLETE,
   PNE_PULL_NOTIFY_OBSOLETE,
   PNE_PULL_RACE_DRIVER,
   PNE_ACK_BAD_TYPE,
   PNE_ACK_BAD_MAGIC,
   PNE_ACK_RESEND_RACE,
   PNE_LATE_ACK,
   PNE_ACK_NACK_FRAMES_IN_PIPE,
   PNE_NACK_BAD_ENDPT,
   PNE_NACK_ENDPT_CLOSED,
   PNE_NACK_BAD_SESSION,
   PNE_NACK_BAD_RDMAWIN,
   PNE_NACK_EVENTQ_FULL,
   PNE_SEND_BAD_RDMAWIN,
   PNE_CONNECT_TIMEOUT,
   PNE_CONNECT_SRC_UNKNOWN,
   PNE_QUERY_BAD_MAGIC,
   PNE_QUERY_TIMED_OUT,
   PNE_QUERY_SRC_UNKNOWN,
   PNE_RAW_SENDS,
   PNE_RAW_RECEIVES,
   PNE_RAW_OVERSIZED_PACKETS,
   PNE_RAW_RECV_OVERRUN,
   PNE_RAW_DISABLED,
   PNE_CONNECT_SEND,
   PNE_CONNECT_RECV,
   PNE_ACK_SEND,
   PNE_ACK_RECV,
   PNE_PUSH_SEND,
   PNE_PUSH_RECV,
   PNE_QUERY_SEND,
   PNE_QUERY_RECV,
   PNE_REPLY_SEND,
   PNE_REPLY_RECV,
   PNE_QUERY_UNKNOWN,
   PNE_DATA_SEND_NULL,
   PNE_DATA_SEND_SMALL,
   PNE_DATA_SEND_MEDIUM,
   PNE_DATA_SEND_RNDV,
   PNE_DATA_SEND_PULL,
   PNE_DATA_RECV_NULL,
   PNE_DATA_RECV_SMALL_INLINE,
   PNE_DATA_RECV_SMALL_COPY,
   PNE_DATA_RECV_MEDIUM,
   PNE_DATA_RECV_RNDV,
   PNE_DATA_RECV_PULL,
   PNE_ETHER_SEND_UNICAST_CNT,
   PNE_ETHER_SEND_MULTICAST_CNT,
   PNE_ETHER_RECV_SMALL_CNT,
   PNE_ETHER_RECV_BIG_CNT,
   PNE_ETHER_OVERRUN,
   PNE_ETHER_OVERSIZED,
   PNE_DATA_RECV_NO_CREDITS,
   PNE_PACKETS_RESENT,
   PNE_PACKETS_DROPPED,
   PNE_MAPPER_ROUTES_UPDATE,
   PNE_ROUTE_DISPERSION,
   PNE_OUT_OF_SEND_HANDLES,
   PNE_OUT_OF_PULL_HANDLES,
   PNE_OUT_OF_PUSH_HANDLES,
   PNE_MEDIUM_CONT_RACE,
   PNE_CMD_TYPE_UNKNOWN,
   PNE_UREQ_TYPE_UNKNOWN,
   PNE_INTERRUPTS_OVERRUN,
   PNE_WAITING_FOR_INTERRUPT_DMA,
   PNE_WAITING_FOR_INTERRUPT_ACK,
   PNE_WAITING_FOR_INTERRUPT_TIMER,
   PNE_SLABS_RECYCLING,
   PNE_SLABS_PRESSURE,
   PNE_SLABS_STARVATION,
   PNE_OUT_OF_RDMA_HANDLES,
   PNE_EVENTQ_FULL,
   PNE_BUFFER_DROP,
   PNE_MEMORY_DROP,
   PNE_HARDWARE_FLOW_CONTROL,
   PNE_SIMULATED_PACKETS_LOST,
   PNE_LOGGING_FRAMES_DUMPED,
   PNE_WAKE_INTERRUPTS,
   PNE_AVERTED_WAKEUP_RACE,
   PNE_DMA_METADATA_RACE
};

#define MXPATH "/opt/mx/bin/mx_counters"

static native_event_entry_t native_table[] = {
   {{ 1, MXPATH},
   "LANAI_UPTIME",
   "Lanai uptime (seconds)",
   },
   {{ 2, MXPATH},
   "COUNTERS_UPTIME",
   "Counters uptime (seconds)",
   },
   {{ 3, MXPATH},
   "BAD_CRC8",
   "Bad CRC8 (Port 0)",
   },
   {{ 4, MXPATH},
   "BAD_CRC32",
   "Bad CRC32 (Port 0)",
   },
   {{ 5, MXPATH},
   "UNSTRIPPED_ROUTE",
   "Unstripped route (Port 0)",
   },
   {{ 6, MXPATH},
   "PKT_DESC_INVALID",
   "pkt_desc_invalid (Port 0)",
   },
   {{ 7, MXPATH},
   "RECV_PKT_ERRORS",
   "recv_pkt_errors (Port 0)",
   },
   {{ 8, MXPATH},
   "PKT_MISROUTED",
   "pkt_misrouted (Port 0)",
   },
   {{ 9, MXPATH},
   "DATA_SRC_UNKNOWN",
   "data_src_unknown",
   },
   {{ 10, MXPATH},
   "DATA_BAD_ENDPT",
   "data_bad_endpt",
   },
   {{ 11, MXPATH},
   "DATA_ENDPT_CLOSED",
   "data_endpt_closed",
   },
   {{ 12, MXPATH},
   "DATA_BAD_SESSION",
   "data_bad_session",
   },
   {{ 13, MXPATH},
   "PUSH_BAD_WINDOW",
   "push_bad_window",
   },
   {{ 14, MXPATH},
   "PUSH_DUPLICATE",
   "push_duplicate",
   },
   {{ 15, MXPATH},
   "PUSH_OBSOLETE",
   "push_obsolete",
   },
   {{ 16, MXPATH},
   "PUSH_RACE_DRIVER",
   "push_race_driver",
   },
   {{ 17, MXPATH},
   "PUSH_BAD_SEND_HANDLE_MAGIC",
   "push_bad_send_handle_magic",
   },
   {{ 18, MXPATH},
   "PUSH_BAD_SRC_MAGIC",
   "push_bad_src_magic",
   },
   {{ 19, MXPATH},
   "PULL_OBSOLETE",
   "pull_obsolete",
   },
   {{ 20, MXPATH},
   "PULL_NOTIFY_OBSOLETE",
   "pull_notify_obsolete",
   },
   {{ 21, MXPATH},
   "PULL_RACE_DRIVER",
   "pull_race_driver",
   },
   {{ 22, MXPATH},
   "ACK_BAD_TYPE",
   "ack_bad_type",
   },
   {{ 23, MXPATH},
   "ACK_BAD_MAGIC",
   "ack_bad_magic",
   },
   {{ 24, MXPATH},
   "ACK_RESEND_RACE",
   "ack_resend_race",
   },
   {{ 25, MXPATH},
   "LATE_ACK",
   "Late ack",
   },
   {{ 26, MXPATH},
   "ACK_NACK_FRAMES_IN_PIPE",
   "ack_nack_frames_in_pipe",
   },
   {{ 27, MXPATH},
   "NACK_BAD_ENDPT",
   "nack_bad_endpt",
   },
   {{ 28, MXPATH},
   "NACK_ENDPT_CLOSED",
   "nack_endpt_closed",
   },
   {{ 29, MXPATH},
   "NACK_BAD_SESSION",
   "nack_bad_session",
   },
   {{ 30, MXPATH},
   "NACK_BAD_RDMAWIN",
   "nack_bad_rdmawin",
   },
   {{ 31, MXPATH},
   "NACK_EVENTQ_FULL",
   "nack_eventq_full",
   },
   {{ 32, MXPATH},
   "SEND_BAD_RDMAWIN",
   "send_bad_rdmawin",
   },
   {{ 33, MXPATH},
   "CONNECT_TIMEOUT",
   "connect_timeout",
   },
   {{ 34, MXPATH},
   "CONNECT_SRC_UNKNOWN",
   "connect_src_unknown",
   },
   {{ 35, MXPATH},
   "QUERY_BAD_MAGIC",
   "query_bad_magic",
   },
   {{ 36, MXPATH},
   "QUERY_TIMED_OUT",
   "query_timed_out",
   },
   {{ 37, MXPATH},
   "QUERY_SRC_UNKNOWN",
   "query_src_unknown",
   },
   {{ 38, MXPATH},
   "RAW_SENDS",
   "Raw sends (Port 0)",
   },
   {{ 39, MXPATH},
   "RAW_RECEIVES",
   "Raw receives (Port 0)",
   },
   {{ 40, MXPATH},
   "RAW_OVERSIZED_PACKETS",
   "Raw oversized packets (Port 0)",
   },
   {{ 41, MXPATH},
   "RAW_RECV_OVERRUN",
   "raw_recv_overrun",
   },
   {{ 42, MXPATH},
   "RAW_DISABLED",
   "raw_disabled",
   },
   {{ 43, MXPATH},
   "CONNECT_SEND",
   "connect_send",
   },
   {{ 44, MXPATH},
   "CONNECT_RECV",
   "connect_recv",
   },
   {{ 45, MXPATH},
   "ACK_SEND",
   "ack_send (Port 0)",
   },
   {{ 46, MXPATH},
   "ACK_RECV",
   "ack_recv (Port 0)",
   },
   {{ 47, MXPATH},
   "PUSH_SEND",
   "push_send (Port 0)",
   },
   {{ 48, MXPATH},
   "PUSH_RECV",
   "push_recv (Port 0)",
   },
   {{ 49, MXPATH},
   "QUERY_SEND",
   "query_send (Port 0)",
   },
   {{ 50, MXPATH},
   "QUERY_RECV",
   "query_recv (Port 0)",
   },
   {{ 51, MXPATH},
   "REPLY_SEND",
   "reply_send (Port 0)",
   },
   {{ 52, MXPATH},
   "REPLY_RECV",
   "reply_recv (Port 0)",
   },
   {{ 53, MXPATH},
   "QUERY_UNKNOWN",
   "query_unknown (Port 0)",
   },
/*   {{ 54, MXPATH},
   "QUERY_UNKNOWN",
   "query_unknown (Port 0)",
   },*/
   {{ 55, MXPATH},
   "DATA_SEND_NULL",
   "data_send_null (Port 0)",
   },
   {{ 56, MXPATH},
   "DATA_SEND_SMALL",
   "data_send_small (Port 0)",
   },
   {{ 57, MXPATH},
   "DATA_SEND_MEDIUM",
   "data_send_medium (Port 0)",
   },
   {{ 58, MXPATH},
   "DATA_SEND_RNDV",
   "data_send_rndv (Port 0)",
   },
   {{ 59, MXPATH},
   "DATA_SEND_PULL",
   "data_send_pull (Port 0)",
   },
   {{ 60, MXPATH},
   "DATA_RECV_NULL",
   "data_recv_null (Port 0)",
   },
   {{ 61, MXPATH},
   "DATA_RECV_SMALL_INLINE",
   "data_recv_small_inline (Port 0)",
   },
   {{ 62, MXPATH},
   "DATA_RECV_SMALL_COPY",
   "data_recv_small_copy (Port 0)",
   },
   {{ 63, MXPATH},
   "DATA_RECV_MEDIUM",
   "data_recv_medium (Port 0)",
   },
   {{ 64, MXPATH},
   "DATA_RECV_RNDV",
   "data_recv_rndv (Port 0)",
   },
   {{ 65, MXPATH},
   "DATA_RECV_PULL",
   "data_recv_pull (Port 0)",
   },
   {{ 66, MXPATH},
   "ETHER_SEND_UNICAST_CNT",
   "ether_send_unicast_cnt (Port 0)",
   },
   {{ 67, MXPATH},
   "ETHER_SEND_MULTICAST_CNT",
   "ether_send_multicast_cnt (Port 0)",
   },
   {{ 68, MXPATH},
   "ETHER_RECV_SMALL_CNT",
   "ether_recv_small_cnt (Port 0)",
   },
   {{ 69, MXPATH},
   "ETHER_RECV_BIG_CNT",
   "ether_recv_big_cnt (Port 0)",
   },
   {{ 70, MXPATH},
   "ETHER_OVERRUN",
   "ether_overrun",
   },
   {{ 71, MXPATH},
   "ETHER_OVERSIZED",
   "ether_oversized",
   },
   {{ 72, MXPATH},
   "DATA_RECV_NO_CREDITS",
   "data_recv_no_credits",
   },
   {{ 73, MXPATH},
   "PACKETS_RECENT",
   "Packets resent",
   },
   {{ 74, MXPATH},
   "PACKETS_DROPPED",
   "Packets dropped (data send side)",
   },
   {{ 75, MXPATH},
   "MAPPER_ROUTES_UPDATE",
   "Mapper routes update",
   },
   {{ 76, MXPATH},
   "ROUTE_DISPERSION",
   "Route dispersion (Port 0)",
   },
   {{ 77, MXPATH},
   "OUT_OF_SEND_HANDLES",
   "out_of_send_handles",
   },
   {{ 78, MXPATH},
   "OUT_OF_PULL_HANDLES",
   "out_of_pull_handles",
   },
   {{ 79, MXPATH},
   "OUT_OF_PUSH_HANDLES",
   "out_of_push_handles",
   },
   {{ 80, MXPATH},
   "MEDIUM_CONT_RACE",
   "medium_cont_race",
   },
   {{ 81, MXPATH},
   "CMD_TYPE_UNKNOWN",
   "cmd_type_unknown",
   },
   {{ 82, MXPATH},
   "UREQ_TYPE_UNKNOWN",
   "ureq_type_unknown",
   },
   {{ 83, MXPATH},
   "INTERRUPTS_OVERRUN",
   "Interrupts overrun",
   },
   {{ 84, MXPATH},
   "WAITING_FOR_INTERRUPT_DMA",
   "Waiting for interrupt DMA",
   },
   {{ 85, MXPATH},
   "WAITING_FOR_INTERRUPT_ACK",
   "Waiting for interrupt Ack",
   },
   {{ 86, MXPATH},
   "WAITING_FOR_INTERRUPT_TIMER",
   "Waiting for interrupt Timer",
   },
   {{ 87, MXPATH},
   "SLABS_RECYCLING",
   "Slabs recycling",
   },
   {{ 88, MXPATH},
   "SLABS_PRESSURE",
   "Slabs pressure",
   },
   {{ 89, MXPATH},
   "SLABS_STARVATION",
   "Slabs starvation",
   },
   {{ 90, MXPATH},
   "OUT_OF_RDMA_HANDLES",
   "out_of_rdma handles",
   },
   {{ 91, MXPATH},
   "EVENTQ_FULL",
   "eventq_full",
   },
   {{ 92, MXPATH},
   "BUFFER_DROP",
   "buffer_drop (Port 0)",
   },
   {{ 93, MXPATH},
   "MEMORY_DROP",
   "memory_drop (Port 0)",
   },
   {{ 94, MXPATH},
   "HARDWARE_FLOW_CONTROL",
   "Hardware flow control (Port 0)",
   },
   {{ 95, MXPATH},
   "SIMULATED_PACKETS_LOST",
   "(Devel) Simulated packets lost (Port 0)",
   },
   {{ 96, MXPATH},
   "LOGGING_FRAMES_DUMPED",
   "(Logging) Logging frames dumped",
   },
   {{ 97, MXPATH},
   "WAKE_INTERRUPTS",
   "Wake interrupts",
   },
   {{ 98, MXPATH},
   "AVERTED_WAKEUP_RACE",
   "Averted wakeup race",
   },
   {{ 99, MXPATH},
   "DMA_METADATA_RACE",
   "Dma metadata race",
   },
   {{0, 0}, "", ""}
};


static long_long _papi_hwd_mx_register_start[MX_MAX_COUNTERS];
static long_long _papi_hwd_mx_register[MX_MAX_COUNTERS];

static papi_svector_t _mx_table[] = {
 {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
 {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {NULL, VEC_PAPI_END}
};


/*
 * Substrate setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int _papi_hwd_init_myrinet_mx_substrate(papi_vectors_t *vtable, int idx)
{
   int retval;

   retval = _papi_hwi_setup_vector_table( vtable, _mx_table);
   
   sidx = idx;

#ifdef DEBUG
   /* This prints out which functions are mapped to dummy routines
    * and this should be taken out once the substrate is completed.
    * The 0 argument will print out only dummy routines, change
    * it to a 1 to print out all routines.
    */
//   vector_print_table(vtable, 0);
#endif
   /* Internal function, doesn't necessarily need to be a function */
   init_mdi();

   /* Internal function, doesn't necessarily need to be a function */
   init_presets();

   return(retval);
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * for the presets are setup here.
 */
static hwi_search_t preset_map[] = {
   {0, {0, {PAPI_NULL, PAPI_NULL}, {0,}}}
};


static int init_presets(){
  return (_papi_hwi_setup_all_presets(preset_map, NULL, sidx));
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * is setup in _papi_hwd_init_substrate.  Below is some, but not
 * all of the values that will need to be setup.  For a complete
 * list check out papi_mdi_t, though some of the values are setup
 * and used above the substrate level.
 */

static void init_mdi(){
   strcpy(_papi_hwi_system_info.hw_info.vendor_string,"linux-myrinet_mx");
   strcpy(_papi_hwi_system_info.hw_info.model_string,"linux-myrinet_mx");
   _papi_hwi_system_info.hw_info.mhz = 100.0;
   _papi_hwi_system_info.hw_info.ncpu = 1;
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = 1;
   _papi_hwi_substrate_info[sidx].num_cntrs = MX_MAX_COUNTERS;
   _papi_hwi_substrate_info[sidx].supports_program = 0;
   _papi_hwi_substrate_info[sidx].supports_write = 0;
   _papi_hwi_substrate_info[sidx].supports_hw_overflow = 0;
   _papi_hwi_substrate_info[sidx].supports_hw_profile = 0;
   _papi_hwi_substrate_info[sidx].supports_64bit_counters = 0;
   _papi_hwi_substrate_info[sidx].supports_attach = 0;
   _papi_hwi_substrate_info[sidx].supports_read_reset = 0;
   _papi_hwi_substrate_info[sidx].context_size = sizeof(hwd_context_t);
   _papi_hwi_substrate_info[sidx].register_size = sizeof(hwd_register_t);
   _papi_hwi_substrate_info[sidx].reg_alloc_size = sizeof(hwd_reg_alloc_t);
   _papi_hwi_substrate_info[sidx].control_state_size = sizeof(hwd_control_state_t);
}


static int read_mx_counters(long_long *counters)
{
   FILE *fp;
   char line[LINELEN], lastchar;
   char *fields=NULL;
   int i, tf, lno;

   fp=popen(MXPATH, "r");
   if(!fp){
      perror("popen");
      return(PAPI_ESBSTR);
   }

   lno=0;
   while(fgets(line, LINELEN, fp)){
      lastchar = ' ';
      tf=0;
      for(i=0; line[i] != '\0' && i < LINELEN-1; i++){
         if(isspace(line[i])){
            if(lastchar != ' '){
               lastchar = line[i];
               line[i] = '\0';
            }
            if(tf && fields)
               break;
         }
         else{
            if(line[i]==':')
               tf++;
            if(isspace(lastchar) && tf){
               fields = line+i;
            }
            lastchar = line[i];
         }
      }
      if(tf){
         counters[lno] = atoll(fields);
         /*printf("--- %d:  %lld\n", lno, counters[lno]);*/
         lno++;
      }
   }
   
   fclose(fp);
 
   return(PAPI_OK);
}

VECTOR_STATIC
int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *ctrl){
   read_mx_counters(_papi_hwd_mx_register_start);
   memcpy(_papi_hwd_mx_register, _papi_hwd_mx_register_start, MX_MAX_COUNTERS*sizeof(long_long));

   return(PAPI_OK);
}


VECTOR_STATIC
int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long **events, int flags)
{
    int i;

    read_mx_counters(_papi_hwd_mx_register); 
    for(i = 0; i < MX_MAX_COUNTERS; i++){
       ctrl->counts[i] = _papi_hwd_mx_register[i] - _papi_hwd_mx_register_start[i];
      /*printf("%d  %lld\n", i, ctrl->counts[i]);*/
    }
    *events=ctrl->counts;
    return(PAPI_OK);
}

VECTOR_STATIC
int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   int i;

   read_mx_counters(_papi_hwd_mx_register);
   for(i = 0; i < MX_MAX_COUNTERS; i++){
      ctrl->counts[i] = _papi_hwd_mx_register[i] - _papi_hwd_mx_register_start[i];
/*      printf("%d  %lld\n", i, ctrl->counts[i]);*/
   }

   return(PAPI_OK);
}

VECTOR_STATIC
int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   _papi_hwd_start(ctx, ctrl);

   return(PAPI_OK);
}


/*
 * Native Event functions
 */

VECTOR_STATIC
int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   if (modifier == PAPI_ENUM_ALL) {
      int index = *EventCode & PAPI_NATIVE_AND_MASK;

      if (native_table[index + 1].resources.selector) {
         *EventCode = *EventCode + 1;
         return (PAPI_OK);
      } else
         return (PAPI_ENOEVNT);
   } 
   else
      return (PAPI_EINVAL);
}

VECTOR_STATIC
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_NATIVE_AND_MASK].name);
}

VECTOR_STATIC
char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_NATIVE_AND_MASK].description);
}

VECTOR_STATIC
int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   memcpy(bits, &(native_table[EventCode & PAPI_NATIVE_AND_MASK].resources), sizeof(hwd_register_t)); /* it is not right, different type */
   return (PAPI_OK);
}

VECTOR_STATIC
int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count)
{
  return(1);
}




