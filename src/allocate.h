/*  This header file contains definition of counter allocation  functions
    created by Haihang You < you@cs.utk.edu >
 */

int _papi_hwd_event_precheck(hwd_control_state_t *tmp_state, unsigned int EventCode, EventInfo_t *out, void *v);
int _papi_hwd_event_mapafter(hwd_control_state_t *tmp_state, int index, EventInfo_t *out);
int do_counter_mapping(hwd_native_t *event_list, int size);
int _papi_hwd_counter_mapping(hwd_control_state_t *tmp_state, unsigned int EventCode, EventInfo_t *out, void *v);
int get_avail_hwcntr_num(int cntr_avail_bits);
void print_state(hwd_control_state_t *s);
