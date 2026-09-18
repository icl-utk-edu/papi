#ifndef CUDA_RANGE_PROFILING_H
#define CUDA_RANGE_PROFILING_H

// Internal headers.
#include <papi.h>

// The lock for the cuda_range component.
extern unsigned int _cuda_range_lock;

/**
 * \brief Params for cuda_range_ctx_t.
 */
typedef struct cuda_range_ctx
{
    /// [in] The current state assigned to the event set (i.e. CUDA_RANGE_EVENTS_STOPPED and CUDA_RANGE_EVENTS_RUNNING).
    int state;
    /// [in] The user added cuda_range native event codes.
    unsigned int *event_codes;
    /// [in] The counter values for the user added cuda_range native events.
    long long *counters;
    /// [in] The number of user added cuda_range native events.
    int num_events;
} *cuda_range_ctx_t;

// The cuda_range PAPI instrumentation.
#ifdef __cplusplus
extern "C" {
#endif
int initialize_cuda_range_component(void);
int get_the_maximum_number_of_hardware_metrics_per_device(int *maximum_number_of_counters);
int cuda_range_event_enum(unsigned int *event_code, int modifier);
int cuda_range_native_event_name_to_native_event_code(const char *name, uint32_t *code);
int cuda_range_native_event_code_to_native_event_name(unsigned int code, char *name, int len);
int cuda_range_event_code_to_info(unsigned int event_code, PAPI_event_info_t *info);
int cuda_range_store_added_native_events(uint32_t *event_codes, int number_of_events);
int cuda_range_start_profiling(void);
int cuda_range_decode_and_evaluate_counter_data(cuda_range_ctx_t ctx, long long **counterValues);
int cuda_range_stop_profiling(void);
int cuda_range_reset_counters(cuda_range_ctx_t ctx);
int cuda_range_unload_function_pointers_and_shutdown(void);
const char *cuda_range_get_last_err_msg(void);
#ifdef __cplusplus
}
#endif

#endif // CUDA_RANGE_PROFILING_H
