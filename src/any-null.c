#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

void init_mdi();
void init_presets();

papi_svector_t _any_null_table[] = {
 {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
 {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
 {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
 {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
 {(void (*)())_papi_hwd_shutdown_global, VEC_PAPI_HWD_SHUTDOWN_GLOBAL},
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_write, VEC_PAPI_HWD_WRITE},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 {(void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
 {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 {(void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_add_prog_event, VEC_PAPI_HWD_ADD_PROG_EVENT},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {(void (*)())_papi_hwd_bpt_map_set, VEC_PAPI_HWD_BPT_MAP_SET },
 {(void (*)())_papi_hwd_bpt_map_avail, VEC_PAPI_HWD_BPT_MAP_AVAIL },
 {(void (*)())_papi_hwd_bpt_map_exclusive, VEC_PAPI_HWD_BPT_MAP_EXCLUSIVE },
 {(void (*)())_papi_hwd_bpt_map_shared, VEC_PAPI_HWD_BPT_MAP_SHARED },
 {(void (*)())_papi_hwd_bpt_map_preempt, VEC_PAPI_HWD_BPT_MAP_PREEMPT },
 {(void (*)())_papi_hwd_bpt_map_update, VEC_PAPI_HWD_BPT_MAP_UPDATE },
 {(void (*)())_papi_hwd_allocate_registers, VEC_PAPI_HWD_ALLOCATE_REGISTERS },
 {NULL, VEC_PAPI_END}
};

/*
 * Substrate setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
   int retval;

   retval = _papi_hwi_setup_vector_table( vtable, _any_null_table);
   
#ifdef DEBUG
   /* This prints out which functions are mapped to dummy routines
    * and this should be taken out once the substrate is completed.
    * The 0 argument will print out only dummy routines, change
    * it to a 1 to print out all routines.
    */
   vector_print_table(vtable, 0);
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
const hwi_search_t preset_map[] = {
   {PAPI_L1_DCM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_ICM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_DCM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_ICM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_TCM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_CA_SNP, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_CA_SHR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_CA_CLN, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_CA_INV, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_CA_ITV, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_LDM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_STM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BRU_IDL, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FXU_IDL, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FPU_IDL, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_LSU_IDL, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_TLB_DM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_TLB_IM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_TLB_TL, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_LDM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_STM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_LDM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_STM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BTAC_M, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_PRF_DM, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_DCH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_TLB_SD, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_CSR_FAL, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_CSR_SUC, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_CSR_TOT, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_MEM_SCY, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_MEM_RCY, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_MEM_WCY, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_STL_ICY, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FUL_ICY, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_STL_CCY, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FUL_CCY, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_HW_INT, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BR_UCN, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BR_CN, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BR_TKN, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BR_NTK, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BR_MSP, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BR_PRC, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FMA_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_TOT_IIS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_TOT_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_INT_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FP_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_LD_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_SR_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_BR_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_VEC_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_RES_STL, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FP_STAL, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_TOT_CYC, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_LST_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_SYC_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCA, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCA, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_DCA, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_DCR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_DCW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_ICH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_ICH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_ICR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_ICR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_ICW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_ICW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_TCH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_TCH, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_TCA, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCA, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_TCA, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_TCR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_TCR, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L1_TCW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_L3_TCW, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FML_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FAD_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FDV_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FSQ_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FNV_INS, {0, {0x1, PAPI_NULL}, {0,}}},
   {PAPI_FP_OPS, {0, {0x1, PAPI_NULL}, {0,}}},
   {0, {0, {PAPI_NULL, PAPI_NULL}, {0,}}}
};


void init_presets(){
  return (_papi_hwi_setup_all_presets(preset_map, NULL));
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * is setup in _papi_hwd_init_substrate.  Below is some, but not
 * all of the values that will need to be setup.  For a complete
 * list check out papi_mdi_t, though some of the values are setup
 * and used above the substrate level.
 */
void init_mdi(){
   strcpy(_papi_hwi_system_info.hw_info.vendor_string,"any-null");
   strcpy(_papi_hwi_system_info.hw_info.model_string,"any-null");
   _papi_hwi_system_info.hw_info.mhz = 100.0;
   _papi_hwi_system_info.hw_info.ncpu = 1;
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = 1;
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
}


/*
 * This is called whenever a thread is initialized
 */
int _papi_hwd_init(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
   return(PAPI_OK);
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int _papi_hwd_init_control_state(hwd_control_state_t *ptr){
   return PAPI_OK;
}

int _papi_hwd_update_control_state(hwd_control_state_t *ptr, NativeInfo_t *native, int count, hwd_context_t *ctx){
   return(PAPI_OK);
}

int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *ctrl){
   return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long **events, int flags)
{
   return(PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   return(PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   return(PAPI_OK);
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long *from)
{
   return(PAPI_OK);
}

/*
 * Overflow and profile functions 
 */
void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, void *context)
{
  /* Real function would call the function below with the proper args
   * _papi_hwi_dispatch_overflow_signal(...);
   */
  return;
}

int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  return(PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, int EventIndex, int threashold)
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
int _papi_hwd_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
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
int _papi_hwd_set_domain(hwd_control_state_t *cntrl, int domain) 
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
long_long _papi_hwd_get_real_usec(void)
{
   return(1);
}

long_long _papi_hwd_get_real_cycles(void)
{
   return(1);
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   return(1);
}

long_long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
   return(1);
}

/*
 * Native Event functions
 */
int _papi_hwd_add_prog_event(hwd_control_state_t * ctrl, unsigned int EventCode, void *inout, EventInfo_t * EventInfoArray){
  return(PAPI_OK);
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
  return(PAPI_OK);
}

char *  _papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
  return("PAPI_ANY_NULL");
}

char * _papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
  return("Event doesn't exist, is an example for a skeleton substrate");
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
   return(PAPI_OK);
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count)
{
  const char str[]="Counter: 0  Event: 0";
  if ( count == 0 ) return(0);

  if ( strlen(str) > name_len ) return(0);

  strcpy(names, "Counter: 0  Event: 0");
  return(1);
}

/* 
 * Counter Allocation Functions, only need to implement if
 *    the substrate needs smart counter allocation.
 */

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) 
{
  return(1);
}

/* Forces the event to be mapped to only counter ctr. */
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
}

/* This function examines the event to determine if it can be mapped 
 * to counter ctr.  Returns true if it can, false if it can't. 
 */
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
   return(1);
} 

/* This function examines the event to determine if it has a single 
 * exclusive mapping.  Returns true if exlusive, false if 
 * non-exclusive.  
 */
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
   return(1);
}

/* This function compares the dst and src events to determine if any 
 * resources are shared. Typically the src event is exclusive, so 
 * this detects a conflict if true. Returns true if conflict, false 
 * if no conflict.  
 */
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
  return(0);
}

/* This function removes shared resources available to the src event
 *  from the resources available to the dst event,
 *  and reduces the rank of the dst event accordingly. Typically,
 *  the src event will be exclusive, but the code shouldn't assume it.
 *  Returns nothing.  
 */
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
  return;
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
  return;
}

/*
 * Shared Library Information and other Information Functions
 */
int _papi_hwd_update_shlib_info(void){
  return(PAPI_OK);
}
