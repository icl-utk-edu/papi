//-----------------------------------------------------------------------------
// @file    amdsmi.h
//
// @brief Header for AMD SMI PAPI component. Defines structures and function
//        prototypes used by linux-amd-smi.c and amdsmi.c.
//-----------------------------------------------------------------------------

#ifndef PAPI_AMDSMI_H
#define PAPI_AMDSMI_H

#include <stdint.h>

// AMD SMI handle types (opaque handles to sockets and processors)
typedef uint32_t amdsmi_socket_handle;
typedef uint32_t amdsmi_processor_handle;

// AMD SMI status code
typedef int32_t amdsmi_status_t;
#define AMDSMI_STATUS_SUCCESS 0

// Initialization flags
#define AMDSMI_INIT_AMD_GPUS (1 << 1)
#define AMDSMI_INIT_AMD_CPUS (1 << 0)  // not used in this component

// Temperature sensor types (see AMD SMI API)
typedef enum {
    AMDSMI_TEMPERATURE_TYPE_FIRST = 0,
    AMDSMI_TEMPERATURE_TYPE_EDGE = 0,
    AMDSMI_TEMPERATURE_TYPE_HOTSPOT = 1,
    AMDSMI_TEMPERATURE_TYPE_JUNCTION = 1,  // alias
    AMDSMI_TEMPERATURE_TYPE_VRAM = 2,
    AMDSMI_TEMPERATURE_TYPE_HBM_0 = 3,
    AMDSMI_TEMPERATURE_TYPE_HBM_1 = 4,
    AMDSMI_TEMPERATURE_TYPE_HBM_2 = 5,
    AMDSMI_TEMPERATURE_TYPE_HBM_3 = 6,
    AMDSMI_TEMPERATURE_TYPE_PLX = 7,
    AMDSMI_TEMPERATURE_TYPE__MAX = AMDSMI_TEMPERATURE_TYPE_PLX
} amdsmi_temperature_type_t;

// Temperature metric types
typedef enum {
    AMDSMI_TEMP_CURRENT = 0,
    AMDSMI_TEMP_FIRST = AMDSMI_TEMP_CURRENT,
    AMDSMI_TEMP_MAX,
    AMDSMI_TEMP_MIN,
    AMDSMI_TEMP_MAX_HYST,
    AMDSMI_TEMP_MIN_HYST,
    AMDSMI_TEMP_CRITICAL,
    AMDSMI_TEMP_CRITICAL_HYST,
    AMDSMI_TEMP_EMERGENCY,
    AMDSMI_TEMP_EMERGENCY_HYST,
    AMDSMI_TEMP_CRIT_MIN,
    AMDSMI_TEMP_CRIT_MIN_HYST,
    AMDSMI_TEMP_OFFSET,
    AMDSMI_TEMP_LOWEST,
    AMDSMI_TEMP_HIGHEST,
    AMDSMI_TEMP_SHUTDOWN,
    AMDSMI_TEMP_LAST = AMDSMI_TEMP_SHUTDOWN
} amdsmi_temperature_metric_t;

// Voltage sensor types (for GPU, only VDDGFX is applicable)
typedef enum {
    AMDSMI_VOLT_TYPE_FIRST = 0,
    AMDSMI_VOLT_TYPE_VDDGFX = 0,
    AMDSMI_VOLT_TYPE_LAST = AMDSMI_VOLT_TYPE_VDDGFX
} amdsmi_voltage_type_t;

// Voltage metric types
typedef enum {
    AMDSMI_VOLT_CURRENT = 0,
    AMDSMI_VOLT_FIRST = AMDSMI_VOLT_CURRENT,
    AMDSMI_VOLT_MAX,
    AMDSMI_VOLT_MIN_CRIT,
    AMDSMI_VOLT_MIN,
    AMDSMI_VOLT_MAX_CRIT,
    AMDSMI_VOLT_AVERAGE,
    AMDSMI_VOLT_LOWEST,
    AMDSMI_VOLT_HIGHEST,
    AMDSMI_VOLT_LAST = AMDSMI_VOLT_HIGHEST
} amdsmi_voltage_metric_t;

// Memory types for GPU memory queries
typedef enum {
    AMDSMI_MEM_TYPE_VRAM = 0,
    AMDSMI_MEM_TYPE_VIS_VRAM = 1,  // visible VRAM (aperture)
    AMDSMI_MEM_TYPE_GTT = 2       // GTT (system memory)
} amdsmi_memory_type_t;

// Clock types (for amdsmi_get_clk_freq)
typedef enum {
    AMDSMI_CLK_TYPE_SYS = 0,
    AMDSMI_CLK_TYPE_FIRST = AMDSMI_CLK_TYPE_SYS,
    AMDSMI_CLK_TYPE_GFX = AMDSMI_CLK_TYPE_SYS,  // alias
    AMDSMI_CLK_TYPE_DF,
    AMDSMI_CLK_TYPE_DCEF,
    AMDSMI_CLK_TYPE_SOC,
    AMDSMI_CLK_TYPE_MEM,
    AMDSMI_CLK_TYPE_PCIE,
    AMDSMI_CLK_TYPE_VCLK0,
    AMDSMI_CLK_TYPE_VCLK1,
    AMDSMI_CLK_TYPE_DCLK0,
    AMDSMI_CLK_TYPE_DCLK1,
    AMDSMI_CLK_TYPE__MAX = AMDSMI_CLK_TYPE_DCLK1
} amdsmi_clk_type_t;

// AMD SMI structures for power info
typedef struct {
    // On Linux bare-metal:
    uint32_t current_socket_power;   // Current power (W)
    uint32_t average_socket_power;   // Average power (W)
    uint32_t gfx_voltage;           // GFX voltage (mV)
    uint32_t power_limit;           // Power limit (W)
    uint32_t reserved[2];
} amdsmi_power_info_t;

// GPU clock frequencies structure
#define AMDSMI_MAX_NUM_FREQUENCIES 64
typedef struct {
    uint32_t num_supported;
    uint32_t current;           // index of current frequency or current frequency value (Hz)
    uint64_t frequency[AMDSMI_MAX_NUM_FREQUENCIES];
} amdsmi_frequencies_t;

// Performance counter event groups and types (for XGMI)
typedef enum {
    AMDSMI_EVNT_GRP_XGMI = 0,
    // For simplicity, use event group code directly as in AMD SMI
    AMDSMI_EVNT_GRP_XGMI_DATA_OUT = 10
} amdsmi_event_group_t;
typedef uint32_t amdsmi_event_type_t;
#define AMDSMI_EVNT_XGMI_DATA_OUT_0 ((amdsmi_event_type_t)0)  // assume continuous range for link events

// Performance counter handle and value
typedef uint64_t amdsmi_event_handle_t;
typedef enum {
    AMDSMI_CNTR_CMD_START = 0,
    AMDSMI_CNTR_CMD_STOP
} amdsmi_counter_command_t;
typedef struct {
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
} amdsmi_counter_value_t;

// Internal access mode
typedef enum {
    AMDSMI_MODE_READ,
    AMDSMI_MODE_WRITE
} amdsmi_access_mode_t;

// Internal context for an open EventSet (structure defined in amdsmi.c)
struct amdsmi_ctx {
    unsigned int *events_id;
    int num_events;
    long long *counters;
    int32_t device_mask;
    int state;
};
typedef struct amdsmi_ctx* amdsmi_ctx_t;

// Public API functions provided by amdsmi.c
int amdsmi_init(void);
int amdsmi_shutdown(void);
int amdsmi_err_get_last(const char **err_string);

// Context management
int amdsmi_ctx_open(unsigned int *events_id, int num_events, amdsmi_ctx_t *ctx);
int amdsmi_ctx_close(amdsmi_ctx_t ctx);
int amdsmi_ctx_start(amdsmi_ctx_t ctx);
int amdsmi_ctx_stop(amdsmi_ctx_t ctx);
int amdsmi_ctx_read(amdsmi_ctx_t ctx, long long **values);
int amdsmi_ctx_write(amdsmi_ctx_t ctx, long long *values);
int amdsmi_ctx_reset(amdsmi_ctx_t ctx);

// Native event enumeration and name/description mapping
int amdsmi_evt_enum(unsigned int *event_code, int modifier);
int amdsmi_evt_code_to_name(unsigned int event_code, char *name, int len);
int amdsmi_evt_code_to_descr(unsigned int event_code, char *descr, int len);
int amdsmi_evt_name_to_code(const char *name, unsigned int *event_code);

// Internal event table management
int init_event_table(void);
int shutdown_event_table(void);

// Utility to get last error (internal)
#endif // PAPI_AMDSMI_H
