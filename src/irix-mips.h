#ifndef _IRIX_MIPS_H
#define _IRIX_MIPS_H

#define MAX_COUNTERS HWPERF_EVENTMAX
#define MAX_NATIVE_EVENT 32
#define PAPI_MAX_NATIVE_EVENTS MAX_NATIVE_EVENT

typedef int hwd_register_t;
typedef int hwd_reg_alloc_t;

typedef struct hwd_control_state {
   /* Generation number of the counters */
   int generation;
   /* Native encoding of the default counting domain */
   int selector;
   /* Buffer to pass to the kernel to control the counters */
   hwperf_profevctrarg_t counter_cmd;
   /* Number on each hwcounter */
   unsigned num_on_counter[2];
   int overflow_event_count;
   /* Buffer for reading counters */
   hwperf_cntr_t cntrs_read;
   /* Buffer for generating overflow vector */
   hwperf_cntr_t cntrs_last_read;
} hwd_control_state_t;

typedef struct _Context {
   /* File descriptor controlling the counters; */
   int fd;
} hwd_context_t;


typedef struct {
   unsigned int ri_fill:16, ri_imp:8,   /* implementation id */
    ri_majrev:4,                /* major revision */
    ri_minrev:4;                /* minor revision */
} papi_rev_id_t;


typedef siginfo_t hwd_siginfo_t;
typedef struct sigcontext hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  (caddr_t)(((hwd_ucontext_t *)ctx.ucontext)->sc_pc)

#endif
