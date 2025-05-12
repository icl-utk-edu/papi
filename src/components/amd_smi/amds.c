#include <string.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <amd_smi/amdsmi.h>
#include <inttypes.h>

#include "papi.h"
#include "papi_memory.h"
#include "amds.h"
#include "htable.h"

unsigned int _amd_smi_lock;

typedef enum {
    PAPI_MODE_READ = 1,
    PAPI_MODE_WRITE,
    PAPI_MODE_RDWR,
} rocs_access_mode_e;

/* Pointers to AMD SMI library functions (dynamically loaded) */
static amdsmi_status_t (*amdsmi_init_p)(uint64_t);
static amdsmi_status_t (*amdsmi_shut_down_p)(void);
static amdsmi_status_t (*amdsmi_get_socket_handles_p)(uint32_t *, amdsmi_socket_handle *);
static amdsmi_status_t (*amdsmi_get_processor_handles_by_type_p)(amdsmi_socket_handle,
                                                                 processor_type_t,
                                                                 amdsmi_processor_handle *, uint32_t *);
static amdsmi_status_t (*amdsmi_get_temp_metric_p)(amdsmi_processor_handle,
                                                   amdsmi_temperature_type_t,
                                                   amdsmi_temperature_metric_t, int64_t *);
static amdsmi_status_t (*amdsmi_get_gpu_fan_rpms_p)(amdsmi_processor_handle, uint32_t, int64_t *);
static amdsmi_status_t (*amdsmi_get_gpu_fan_speed_p)(amdsmi_processor_handle, uint32_t, int64_t *);
static amdsmi_status_t (*amdsmi_get_gpu_fan_speed_max_p)(amdsmi_processor_handle, uint32_t, int64_t *);

static amdsmi_status_t (*amdsmi_get_total_memory_p)(amdsmi_processor_handle, amdsmi_memory_type_t, uint64_t *);
static amdsmi_status_t (*amdsmi_get_memory_usage_p)(amdsmi_processor_handle, amdsmi_memory_type_t, uint64_t *);
static amdsmi_status_t (*amdsmi_get_gpu_activity_p)(amdsmi_processor_handle, amdsmi_engine_usage_t *);
static amdsmi_status_t (*amdsmi_get_power_cap_info_p)(amdsmi_processor_handle, uint32_t, amdsmi_power_cap_info_t *);  /* FIX: added sensor index param */
static amdsmi_status_t (*amdsmi_get_gpu_power_cap_set_p)(amdsmi_processor_handle, uint32_t, uint64_t);
static amdsmi_status_t (*amdsmi_get_gpu_power_ave_p)(amdsmi_processor_handle, uint32_t, uint64_t *);
static amdsmi_status_t (*amdsmi_get_power_info_p)(amdsmi_processor_handle, amdsmi_power_info_t *);
static amdsmi_status_t (*amdsmi_set_power_cap_p)(amdsmi_processor_handle, uint32_t, uint64_t);
static amdsmi_status_t (*amdsmi_get_gpu_pci_throughput_p)(amdsmi_processor_handle, uint64_t *, uint64_t *, uint64_t *);
static amdsmi_status_t (*amdsmi_get_gpu_pci_replay_counter_p)(amdsmi_processor_handle, uint64_t *);
static amdsmi_status_t (*amdsmi_get_clk_freq_p)(amdsmi_processor_handle, amdsmi_clk_type_t, amdsmi_frequencies_t *);
static amdsmi_status_t (*amdsmi_set_clk_freq_p)(amdsmi_processor_handle, amdsmi_clk_type_t, uint64_t);
static amdsmi_status_t (*amdsmi_get_gpu_metrics_info_p)(amdsmi_processor_handle, amdsmi_gpu_metrics_t *);

/* New AMD SMI function pointers */
static amdsmi_status_t (*amdsmi_get_gpu_id_p)(amdsmi_processor_handle, uint16_t *);
static amdsmi_status_t (*amdsmi_get_gpu_revision_p)(amdsmi_processor_handle, uint16_t *);
static amdsmi_status_t (*amdsmi_get_gpu_subsystem_id_p)(amdsmi_processor_handle, uint16_t *);
static amdsmi_status_t (*amdsmi_get_gpu_virtualization_mode_p)(amdsmi_processor_handle, amdsmi_virtualization_mode_t *);
static amdsmi_status_t (*amdsmi_get_gpu_pci_bandwidth_p)(amdsmi_processor_handle, amdsmi_pcie_bandwidth_t *);
static amdsmi_status_t (*amdsmi_get_gpu_bdf_id_p)(amdsmi_processor_handle, uint64_t *);
static amdsmi_status_t (*amdsmi_get_gpu_topo_numa_affinity_p)(amdsmi_processor_handle, int32_t *);
static amdsmi_status_t (*amdsmi_get_energy_count_p)(amdsmi_processor_handle, uint64_t *, float *, uint64_t *);
static amdsmi_status_t (*amdsmi_get_gpu_power_profile_presets_p)(amdsmi_processor_handle, uint32_t, amdsmi_power_profile_status_t *);

/* Global device list and count */
static int32_t device_count = 0;
static amdsmi_processor_handle *device_handles = NULL;
static int32_t device_mask = 0;

static void *amds_dlp = NULL;
static void *htable = NULL;
static char error_string[PAPI_MAX_STR_LEN+1];

/* forward declarations for internal helpers */
static int load_amdsmi_sym(void);
static int unload_amdsmi_sym(void);
static int init_device_table(void);
static int shutdown_device_table(void);
static int init_event_table(void);
static int shutdown_event_table(void);

/* Event descriptor structure for native events */
typedef struct native_event {
    unsigned int id;
    char *name;
    char *descr;
    int32_t device;                /* device index or -1 if not applicable */
    uint64_t value;               /* last read value or set value */
    uint32_t mode;                /* access mode (read/write) */
    uint32_t variant;             /* variant index (for metric type, etc.) */
    uint32_t subvariant;          /* subvariant index (for sensor index or sub-type) */
    int (*open_func)(struct native_event *);    /* optional open (reserve resources) */
    int (*close_func)(struct native_event *);   /* optional close (release resources) */
    int (*start_func)(struct native_event *);   /* optional start (begin counting) */
    int (*stop_func)(struct native_event *);    /* optional stop (stop counting) */
    int (*access_func)(int mode, void *arg);    /* read or write the event value */
} native_event_t;

/* Table of all native events */
typedef struct {
    native_event_t *events;
    int count;
} native_event_table_t;

static native_event_table_t ntv_table;
static native_event_table_t *ntv_table_p = NULL;

/* Locking device usage for contexts */
static int
acquire_devices(unsigned int *events_id, int num_events, int32_t *bitmask)
{
    int32_t mask_acq = 0;
    for (int i = 0; i < num_events; ++i) {
        int32_t dev_id = ntv_table_p->events[events_id[i]].device;
        if (dev_id < 0) continue;
        mask_acq |= (1 << dev_id);
    }
    if (mask_acq & device_mask) {
        return PAPI_ECNFLCT;  // conflict: device already in use
    }
    device_mask |= mask_acq;
    *bitmask = mask_acq;
    return PAPI_OK;
}

static int
release_devices(int32_t *bitmask)
{
    int32_t mask_rel = *bitmask;
    if ((mask_rel & device_mask) != mask_rel) {
        return PAPI_EMISC;
    }
    device_mask ^= mask_rel;
    *bitmask = 0;
    return PAPI_OK;
}

/* Prototypes for event access (read/write) functions */
static int access_amdsmi_temp_metric(int mode, void *arg);
static int access_amdsmi_fan_speed(int mode, void *arg);
static int access_amdsmi_fan_rpms(int mode, void *arg);
static int access_amdsmi_mem_total(int mode, void *arg);
static int access_amdsmi_mem_usage(int mode, void *arg);
static int access_amdsmi_power_cap(int mode, void *arg);
static int access_amdsmi_power_cap_range(int mode, void *arg);
static int access_amdsmi_power_average(int mode, void *arg);
static int access_amdsmi_pci_throughput(int mode, void *arg);
static int access_amdsmi_pci_replay_counter(int mode, void *arg);
static int access_amdsmi_clk_freq(int mode, void *arg);
static int access_amdsmi_gpu_metrics(int mode, void *arg);
static int access_amdsmi_gpu_info(int mode, void *arg);
static int access_amdsmi_gpu_activity(int mode, void *arg);
static int access_amdsmi_fan_speed_max(int mode, void *arg);
static int access_amdsmi_pci_bandwidth(int mode, void *arg);
static int access_amdsmi_energy_count(int mode, void *arg);
static int access_amdsmi_power_profile_status(int mode, void *arg);

/* Simple open/close/start/stop functions (no special handling needed for most events) */
static int open_simple(native_event_t *event)   { (void)event; return PAPI_OK; }

static int close_simple(native_event_t *event)  { (void)event; return PAPI_OK; }

static int start_simple(native_event_t *event)  { (void)event; return PAPI_OK; }

static int stop_simple(native_event_t *event)   { (void)event; return PAPI_OK; }

/* Dynamic load of AMD SMI library symbols */
static void *sym(const char *preferred, const char *fallback) {
    void *p = dlsym(amds_dlp, preferred);
    return p ? p : (fallback ? dlsym(amds_dlp, fallback) : NULL);
}

static int load_amdsmi_sym(void) {
    const char *root = getenv("PAPI_AMDSMI_ROOT");
    char so_path[PATH_MAX] = {0};
    if (!root) {
        snprintf(error_string, sizeof(error_string),
                 "PAPI_AMDSMI_ROOT not set; cannot find libamd_smi.so");
        return PAPI_ENOSUPP;
    }
    snprintf(so_path, sizeof(so_path), "%s/lib/libamd_smi.so", root);
    amds_dlp = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
    if (!amds_dlp) {
        snprintf(error_string, sizeof(error_string),
                 "dlopen(\"%s\"): %s", so_path, dlerror());
        return PAPI_ENOSUPP;
    }
    /* Resolve required function symbols */
    amdsmi_init_p          = sym("amdsmi_init", NULL);
    amdsmi_shut_down_p     = sym("amdsmi_shut_down", NULL);
    amdsmi_get_socket_handles_p = sym("amdsmi_get_socket_handles", NULL);
    amdsmi_get_processor_handles_by_type_p = sym("amdsmi_get_processor_handles_by_type", NULL);

    /* Sensors */
    amdsmi_get_temp_metric_p       = sym("amdsmi_get_temp_metric", NULL);
    amdsmi_get_gpu_fan_rpms_p      = sym("amdsmi_get_gpu_fan_rpms", NULL);
    amdsmi_get_gpu_fan_speed_p     = sym("amdsmi_get_gpu_fan_speed", NULL);
    amdsmi_get_gpu_fan_speed_max_p = sym("amdsmi_get_gpu_fan_speed_max", NULL);

    /* Memory */
    amdsmi_get_total_memory_p   = sym("amdsmi_get_gpu_memory_total", "amdsmi_get_total_memory");
    amdsmi_get_memory_usage_p   = sym("amdsmi_get_gpu_memory_usage", "amdsmi_get_memory_usage");

    /* Utilization / activity */
    amdsmi_get_gpu_activity_p   = sym("amdsmi_get_gpu_activity", "amdsmi_get_engine_usage"); /* fallback for older name */

    /* Power */
    amdsmi_get_power_info_p     = sym("amdsmi_get_power_info", NULL);
    amdsmi_get_power_cap_info_p = sym("amdsmi_get_power_cap_info", NULL);
    amdsmi_set_power_cap_p      = sym("amdsmi_set_power_cap", "amdsmi_dev_set_power_cap");

    /* PCIe */
    amdsmi_get_gpu_pci_throughput_p     = sym("amdsmi_get_gpu_pci_throughput", NULL);
    amdsmi_get_gpu_pci_replay_counter_p = sym("amdsmi_get_gpu_pci_replay_counter", NULL);

    /* Clocks */
    amdsmi_get_clk_freq_p       = sym("amdsmi_get_clk_freq", NULL);
    amdsmi_set_clk_freq_p       = sym("amdsmi_set_clk_freq", NULL);

    /* GPU metrics */
    amdsmi_get_gpu_metrics_info_p = sym("amdsmi_get_gpu_metrics_info", NULL);

    /* Identification and other queries */
    amdsmi_get_gpu_id_p                    = sym("amdsmi_get_gpu_id", NULL);
    amdsmi_get_gpu_revision_p             = sym("amdsmi_get_gpu_revision", NULL);
    amdsmi_get_gpu_subsystem_id_p         = sym("amdsmi_get_gpu_subsystem_id", NULL);
    amdsmi_get_gpu_virtualization_mode_p  = sym("amdsmi_get_gpu_virtualization_mode", NULL);
    amdsmi_get_gpu_pci_bandwidth_p        = sym("amdsmi_get_gpu_pci_bandwidth", NULL);
    amdsmi_get_gpu_bdf_id_p               = sym("amdsmi_get_gpu_bdf_id", NULL);
    amdsmi_get_gpu_topo_numa_affinity_p   = sym("amdsmi_get_gpu_topo_numa_affinity", NULL);
    amdsmi_get_energy_count_p             = sym("amdsmi_get_energy_count", NULL);
    amdsmi_get_gpu_power_profile_presets_p = sym("amdsmi_get_gpu_power_profile_presets", NULL);

    /* Verify that all required symbols are loaded */
    struct { const char *name; void *ptr; } required[] = {
        { "amdsmi_init",          amdsmi_init_p },
        { "amdsmi_shut_down",     amdsmi_shut_down_p },
        { "amdsmi_get_socket_handles", amdsmi_get_socket_handles_p },
        { "amdsmi_get_processor_handles_by_type", amdsmi_get_processor_handles_by_type_p },
        { "amdsmi_get_temp_metric",    amdsmi_get_temp_metric_p },
        { "amdsmi_get_gpu_memory_total", amdsmi_get_total_memory_p },
        { "amdsmi_get_gpu_memory_usage", amdsmi_get_memory_usage_p },
        { "amdsmi_get_gpu_activity",    amdsmi_get_gpu_activity_p },
        { "amdsmi_get_power_cap_info",  amdsmi_get_power_cap_info_p },
        { "amdsmi_set_power_cap",       amdsmi_set_power_cap_p },
        { "amdsmi_get_power_info",      amdsmi_get_power_info_p },
        { "amdsmi_get_gpu_pci_throughput",  amdsmi_get_gpu_pci_throughput_p },
        { "amdsmi_get_gpu_pci_replay_counter", amdsmi_get_gpu_pci_replay_counter_p },
        { "amdsmi_get_gpu_fan_rpms",    amdsmi_get_gpu_fan_rpms_p },
        { "amdsmi_get_gpu_fan_speed",   amdsmi_get_gpu_fan_speed_p },
        { "amdsmi_get_gpu_fan_speed_max", amdsmi_get_gpu_fan_speed_max_p },
        { "amdsmi_get_clk_freq",        amdsmi_get_clk_freq_p },
        { "amdsmi_set_clk_freq",        amdsmi_set_clk_freq_p },
        { "amdsmi_get_gpu_metrics_info", amdsmi_get_gpu_metrics_info_p },
        { "amdsmi_get_gpu_id",          amdsmi_get_gpu_id_p },
        { "amdsmi_get_gpu_revision",    amdsmi_get_gpu_revision_p },
        { "amdsmi_get_gpu_subsystem_id", amdsmi_get_gpu_subsystem_id_p },
        { "amdsmi_get_gpu_virtualization_mode", amdsmi_get_gpu_virtualization_mode_p },
        { "amdsmi_get_gpu_pci_bandwidth", amdsmi_get_gpu_pci_bandwidth_p },
        { "amdsmi_get_gpu_bdf_id",      amdsmi_get_gpu_bdf_id_p },
        { "amdsmi_get_gpu_topo_numa_affinity", amdsmi_get_gpu_topo_numa_affinity_p },
        { "amdsmi_get_energy_count",    amdsmi_get_energy_count_p },
        { "amdsmi_get_gpu_power_profile_presets", amdsmi_get_gpu_power_profile_presets_p }
    };
    int miss = 0;
    int pos = snprintf(error_string, sizeof(error_string), "Missing AMD SMI symbols:");
    for (size_t i = 0; i < sizeof(required)/sizeof(required[0]); ++i) {
        if (!required[i].ptr) {
            miss++;
            pos += snprintf(error_string + pos, sizeof(error_string) - pos, "\n  %s", required[i].name);
        }
    }
    if (miss) {
        dlclose(amds_dlp); amds_dlp = NULL;
        return PAPI_ENOSUPP;
    }
    return PAPI_OK;
}

static int unload_amdsmi_sym(void) {
    /* Reset all function pointers and close library */
    amdsmi_init_p = NULL;
    amdsmi_shut_down_p = NULL;
    amdsmi_get_socket_handles_p = NULL;
    amdsmi_get_processor_handles_by_type_p = NULL;
    amdsmi_get_temp_metric_p = NULL;
    amdsmi_get_gpu_fan_rpms_p = NULL;
    amdsmi_get_gpu_fan_speed_p = NULL;
    amdsmi_get_gpu_fan_speed_max_p = NULL;
    amdsmi_get_total_memory_p = NULL;
    amdsmi_get_memory_usage_p = NULL;
    amdsmi_get_gpu_activity_p = NULL;
    amdsmi_get_power_cap_info_p = NULL;
    amdsmi_set_power_cap_p = NULL;
    amdsmi_get_power_info_p = NULL;
    amdsmi_get_gpu_pci_throughput_p = NULL;
    amdsmi_get_gpu_pci_replay_counter_p = NULL;
    amdsmi_get_clk_freq_p = NULL;
    amdsmi_set_clk_freq_p = NULL;
    amdsmi_get_gpu_metrics_info_p = NULL;
    amdsmi_get_gpu_id_p = NULL;
    amdsmi_get_gpu_revision_p = NULL;
    amdsmi_get_gpu_subsystem_id_p = NULL;
    amdsmi_get_gpu_virtualization_mode_p = NULL;
    amdsmi_get_gpu_pci_bandwidth_p = NULL;
    amdsmi_get_gpu_bdf_id_p = NULL;
    amdsmi_get_gpu_topo_numa_affinity_p = NULL;
    amdsmi_get_energy_count_p = NULL;
    amdsmi_get_gpu_power_profile_presets_p = NULL;
    if (amds_dlp) {
        dlclose(amds_dlp);
        amds_dlp = NULL;
    }
    return PAPI_OK;
}

/* Initialize AMD SMI library and discover devices */
int amds_init(void) {
    int papi_errno = load_amdsmi_sym();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    amdsmi_status_t status = amdsmi_init_p(AMDSMI_INIT_AMD_GPUS);
    if (status != AMDSMI_STATUS_SUCCESS) {
        strcpy(error_string, "amdsmi_init failed");
        return PAPI_ENOSUPP;
    }
    htable_init(&htable);

    // Discover GPU devices (get socket handles, then GPU handles)
    uint32_t socket_count = 0;
    status = amdsmi_get_socket_handles_p(&socket_count, NULL);
    if (status != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
        sprintf(error_string, "Error discovering sockets or no AMD GPU socket found.");
        papi_errno = PAPI_ENOEVNT;
        goto fn_fail;
    }
    amdsmi_socket_handle *sockets = (amdsmi_socket_handle *) papi_calloc(socket_count, sizeof(amdsmi_socket_handle));
    if (!sockets) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }
    status = amdsmi_get_socket_handles_p(&socket_count, sockets);
    if (status != AMDSMI_STATUS_SUCCESS) {
        sprintf(error_string, "Error getting socket handles.");
        papi_free(sockets);
        papi_errno = PAPI_ENOSUPP;
        goto fn_fail;
    }

    // **FIX**: Handle multiple GPUs per socket by counting total GPUs first
    device_count = 0;
    uint32_t total_gpu_count = 0;
    for (uint32_t s = 0; s < socket_count; ++s) {
        uint32_t gpu_count = 0;
        processor_type_t proc_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
        amdsmi_status_t st = amdsmi_get_processor_handles_by_type_p(sockets[s], proc_type, NULL, &gpu_count);
        if (st == AMDSMI_STATUS_SUCCESS) {
            total_gpu_count += gpu_count;
        }
    }
    if (total_gpu_count == 0) {
        sprintf(error_string, "No AMD GPU devices found.");
        papi_errno = PAPI_ENOEVNT;
        papi_free(sockets);
        goto fn_fail;
    }
    device_handles = (amdsmi_processor_handle *) papi_calloc(total_gpu_count, sizeof(*device_handles));
    if (!device_handles) {
        papi_errno = PAPI_ENOMEM;
        sprintf(error_string, "Memory allocation error for device handles.");
        papi_free(sockets);
        goto fn_fail;
    }

    // Retrieve GPU processor handles for each socket
    for (uint32_t s = 0; s < socket_count; ++s) {
        uint32_t gpu_count = 0;
        processor_type_t proc_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
        status = amdsmi_get_processor_handles_by_type_p(sockets[s], proc_type, NULL, &gpu_count);
        if (status != AMDSMI_STATUS_SUCCESS || gpu_count == 0) {
            continue;  // no GPU on this socket or error
        }
        amdsmi_processor_handle *gpu_handles = (amdsmi_processor_handle *) papi_calloc(gpu_count, sizeof(*gpu_handles));
        if (!gpu_handles) {
            papi_errno = PAPI_ENOMEM;
            sprintf(error_string, "Memory allocation error for GPU handles on socket %u.", s);
            papi_free(sockets);
            goto fn_fail;
        }
        status = amdsmi_get_processor_handles_by_type_p(sockets[s], proc_type, gpu_handles, &gpu_count);
        if (status == AMDSMI_STATUS_SUCCESS) {
            for (uint32_t g = 0; g < gpu_count; ++g) {
                device_handles[device_count++] = gpu_handles[g];
            }
        }
        papi_free(gpu_handles);
    }
    papi_free(sockets);
    if (device_count == 0) {
        sprintf(error_string, "No AMD GPU devices found.");
        papi_errno = PAPI_ENOEVNT;
        goto fn_fail;
    }

    // Initialize the native event table for all discovered metrics
    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while initializing the native event table.");
        goto fn_fail;
    }
    ntv_table_p = &ntv_table;
    return PAPI_OK;

fn_fail:
    htable_shutdown(htable);
    if (device_handles) { papi_free(device_handles); device_handles = NULL; device_count = 0; }
    if (sockets) papi_free(sockets);
    amdsmi_shut_down_p();
    unload_amdsmi_sym();
    return papi_errno;
}

int amds_shutdown(void) {
    shutdown_event_table();
    shutdown_device_table();
    htable_shutdown(htable);
    amdsmi_shut_down_p();  // shutdown AMD SMI library
    return unload_amdsmi_sym();
}

int amds_err_get_last(const char **err_string) {
    if (err_string) *err_string = error_string;
    return PAPI_OK;
}

/* Event enumeration: iterate over native events */
int
amds_evt_enum(unsigned int *EventCode, int modifier)
{
    if (modifier == PAPI_ENUM_FIRST) {
        if (ntv_table_p->count == 0) {
            return PAPI_ENOEVNT;
        }
        *EventCode = 0;
        return PAPI_OK;
    } else if (modifier == PAPI_ENUM_EVENTS) {
        if (*EventCode + 1 < (unsigned int) ntv_table_p->count) {
            *EventCode = *EventCode + 1;
            return PAPI_OK;
        } else {
            return PAPI_ENOEVNT;
        }
    }
    return PAPI_EINVAL;
}

int amds_evt_code_to_name(unsigned int EventCode, char *name, int len) {
    if (EventCode >= (unsigned int) ntv_table_p->count) {
        return PAPI_EINVAL;
    }
    strncpy(name, ntv_table_p->events[EventCode].name, len);
    return PAPI_OK;
}

int amds_evt_name_to_code(const char *name, unsigned int *EventCode) {
    native_event_t *event = NULL;
    int hret = htable_find(htable, name, (void **) &event);
    if (hret != HTABLE_SUCCESS) {
        return (hret == HTABLE_ENOVAL) ? PAPI_ENOEVNT : PAPI_ECMP;
    }
    *EventCode = event->id;
    return PAPI_OK;
}

int amds_evt_code_to_descr(unsigned int EventCode, char *descr, int len) {
    if (EventCode >= (unsigned int) ntv_table_p->count) {
        return PAPI_EINVAL;
    }
    strncpy(descr, ntv_table_p->events[EventCode].descr, len);
    return PAPI_OK;
}

/* Context management: open/close, start/stop, read/write, reset */
struct amds_ctx {
    int state;
    unsigned int *events_id;
    int num_events;
    long long *counters;
    int32_t device_mask;
};

int
amds_ctx_open(unsigned int *event_ids, int num_events, amds_ctx_t *ctx)
{
    amds_ctx_t new_ctx = (amds_ctx_t) papi_calloc(1, sizeof(struct amds_ctx));
    if (new_ctx == NULL) {
        return PAPI_ENOMEM;
    }
    new_ctx->events_id = event_ids;
    new_ctx->num_events = num_events;
    new_ctx->counters = (long long *) papi_calloc(num_events, sizeof(long long));
    if (new_ctx->counters == NULL) {
        papi_free(new_ctx);
        return PAPI_ENOMEM;
    }
    // Acquire devices needed by these events to avoid conflicts
    int papi_errno = acquire_devices(event_ids, num_events, &new_ctx->device_mask);
    if (papi_errno != PAPI_OK) {
        papi_free(new_ctx->counters);
        papi_free(new_ctx);
        return papi_errno;
    }
    *ctx = new_ctx;
    return PAPI_OK;
}

int
amds_ctx_close(amds_ctx_t ctx)
{
    if (!ctx) return PAPI_OK;
    // release device usage
    release_devices(&ctx->device_mask);
    papi_free(ctx->counters);
    papi_free(ctx);
    return PAPI_OK;
}

int
amds_ctx_start(amds_ctx_t ctx)
{
    (void) ctx;
    // No additional actions needed to start in this design (all reads are on-demand)
    ctx->state |= AMDS_EVENTS_RUNNING;
    return PAPI_OK;
}

int
amds_ctx_stop(amds_ctx_t ctx)
{
    if (!(ctx->state & AMDS_EVENTS_RUNNING)) {
        return PAPI_OK;
    }
    ctx->state &= ~AMDS_EVENTS_RUNNING;
    return PAPI_OK;
}

int
amds_ctx_read(amds_ctx_t ctx, long long **counts)
{
    int papi_errno = PAPI_OK;
    for (int i = 0; i < ctx->num_events; ++i) {
        unsigned int id = ctx->events_id[i];
        papi_errno = ntv_table_p->events[id].access_func(PAPI_MODE_READ, &ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
        ctx->counters[i] = (long long) ntv_table_p->events[id].value;
    }
    *counts = ctx->counters;
    return papi_errno;
}

int
amds_ctx_write(amds_ctx_t ctx, long long *counts)
{
    int papi_errno = PAPI_OK;
    for (int i = 0; i < ctx->num_events; ++i) {
        unsigned int id = ctx->events_id[i];
        ntv_table_p->events[id].value = counts[i];
        papi_errno = ntv_table_p->events[id].access_func(PAPI_MODE_WRITE, &ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }
    return papi_errno;
}

int
amds_ctx_reset(amds_ctx_t ctx)
{
    for (int i = 0; i < ctx->num_events; ++i) {
        unsigned int id = ctx->events_id[i];
        ntv_table_p->events[id].value = 0;
        ctx->counters[i] = 0;
    }
    return PAPI_OK;
}

/* Initialize native event table: enumerate all supported events */
static int init_event_table(void) {
    ntv_table.count = 0;
    int idx = 0;
    ntv_table.events = (native_event_t *) papi_calloc(512 * device_count, sizeof(native_event_t));
    if (!ntv_table.events) {
        return PAPI_ENOMEM;
    }
    char name_buf[PAPI_MAX_STR_LEN];
    char descr_buf[PAPI_MAX_STR_LEN];

    /* Temperature sensors (only include if sensor is present) */
    amdsmi_temperature_type_t temp_sensors[] = {
        AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMPERATURE_TYPE_JUNCTION, AMDSMI_TEMPERATURE_TYPE_VRAM, AMDSMI_TEMPERATURE_TYPE_HBM_0 , AMDSMI_TEMPERATURE_TYPE_HBM_1 , AMDSMI_TEMPERATURE_TYPE_HBM_2 ,
        AMDSMI_TEMPERATURE_TYPE_HBM_3, AMDSMI_TEMPERATURE_TYPE_PLX
    };
    const amdsmi_temperature_metric_t temp_metrics[] = {
        AMDSMI_TEMP_CURRENT, AMDSMI_TEMP_MAX, AMDSMI_TEMP_MIN, AMDSMI_TEMP_MAX_HYST,
        AMDSMI_TEMP_MIN_HYST, AMDSMI_TEMP_CRITICAL, AMDSMI_TEMP_CRITICAL_HYST,
        AMDSMI_TEMP_EMERGENCY, AMDSMI_TEMP_EMERGENCY_HYST, AMDSMI_TEMP_CRIT_MIN,
        AMDSMI_TEMP_CRIT_MIN_HYST, AMDSMI_TEMP_OFFSET, AMDSMI_TEMP_LOWEST, AMDSMI_TEMP_HIGHEST
    };
    const char *temp_metric_names[] = {
        "temp_current", "temp_max", "temp_min", "temp_max_hyst", "temp_min_hyst",
        "temp_critical", "temp_critical_hyst", "temp_emergency", "temp_emergency_hyst",
        "temp_crit_min", "temp_crit_min_hyst", "temp_offset", "temp_lowest", "temp_highest"
    };
    for (int d = 0; d < device_count; ++d) {
        for (size_t si = 0; si < sizeof(temp_sensors)/sizeof(temp_sensors[0]); ++si) {
            // Probe if sensor exists by reading current temperature
            int64_t dummy_val;
            if (amdsmi_get_temp_metric_p(device_handles[d], temp_sensors[si], AMDSMI_TEMP_CURRENT, &dummy_val)
                != AMDSMI_STATUS_SUCCESS) {
                continue;  // skip this sensor if not present
            }
            for (size_t mi = 0; mi < sizeof(temp_metrics)/sizeof(temp_metrics[0]); ++mi) {
                snprintf(name_buf, sizeof(name_buf), "%s:device=%d:sensor=%d",
                         temp_metric_names[mi], d, (int) temp_sensors[si]);
                snprintf(descr_buf, sizeof(descr_buf), "Device %d %s for sensor %d",
                         d, temp_metric_names[mi], (int) temp_sensors[si]);
                native_event_t *ev = &ntv_table.events[idx];
                ev->id = idx;
                ev->name = strdup(name_buf);
                ev->descr = strdup(descr_buf);
                ev->device = d;
                ev->value = 0;
                ev->mode = PAPI_MODE_READ;
                ev->variant = temp_metrics[mi];
                ev->subvariant = temp_sensors[si];
                ev->open_func = open_simple;
                ev->close_func = close_simple;
                ev->start_func = start_simple;
                ev->stop_func = stop_simple;
                ev->access_func = access_amdsmi_temp_metric;
                htable_insert(htable, ev->name, ev);
                idx++;
            }
        }
    }

    /* Fan metrics (assume one fan sensor index 0 per device) */
    for (int d = 0; d < device_count; ++d) {
        // Fan RPM (speed in RPM)
        snprintf(name_buf, sizeof(name_buf), "fan_rpms:device=%d:sensor=0", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d fan speed in RPM", d);
        native_event_t *ev_rpm = &ntv_table.events[idx];
        ev_rpm->id = idx;
        ev_rpm->name = strdup(name_buf);
        ev_rpm->descr = strdup(descr_buf);
        ev_rpm->device = d;
        ev_rpm->value = 0;
        ev_rpm->mode = PAPI_MODE_READ;
        ev_rpm->variant = 0;
        ev_rpm->subvariant = 0;
        ev_rpm->open_func = open_simple;
        ev_rpm->close_func = close_simple;
        ev_rpm->start_func = start_simple;
        ev_rpm->stop_func = stop_simple;
        ev_rpm->access_func = access_amdsmi_fan_rpms;
        htable_insert(htable, ev_rpm->name, ev_rpm);
        idx++;

        // Fan speed (relative value 0-255, read-only; write not implemented)
        snprintf(name_buf, sizeof(name_buf), "fan_speed:device=%d:sensor=0", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d fan speed (0-255 relative)", d);
        native_event_t *ev_fan = &ntv_table.events[idx];
        ev_fan->id = idx;
        ev_fan->name = strdup(name_buf);
        ev_fan->descr = strdup(descr_buf);
        ev_fan->device = d;
        ev_fan->value = 0;
        ev_fan->mode = PAPI_MODE_READ;  /* FIX: only read supported, write not implemented */
        ev_fan->variant = 0;
        ev_fan->subvariant = 0;
        ev_fan->open_func = open_simple;
        ev_fan->close_func = close_simple;
        ev_fan->start_func = start_simple;
        ev_fan->stop_func = stop_simple;
        ev_fan->access_func = access_amdsmi_fan_speed;
        htable_insert(htable, ev_fan->name, ev_fan);
        idx++;

        // Fan max RPM (maximum speed in RPM)
        snprintf(name_buf, sizeof(name_buf), "fan_rpms_max:device=%d:sensor=0", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d fan maximum speed in RPM", d);
        native_event_t *ev_fanmax = &ntv_table.events[idx];
        ev_fanmax->id = idx;
        ev_fanmax->name = strdup(name_buf);
        ev_fanmax->descr = strdup(descr_buf);
        ev_fanmax->device = d;
        ev_fanmax->value = 0;
        ev_fanmax->mode = PAPI_MODE_READ;
        ev_fanmax->variant = 0;
        ev_fanmax->subvariant = 0;
        ev_fanmax->open_func = open_simple;
        ev_fanmax->close_func = close_simple;
        ev_fanmax->start_func = start_simple;
        ev_fanmax->stop_func = stop_simple;
        ev_fanmax->access_func = access_amdsmi_fan_speed_max;
        htable_insert(htable, ev_fanmax->name, ev_fanmax);
        idx++;
    }

    /* VRAM memory metrics */
    for (int d = 0; d < device_count; ++d) {
        // Total VRAM
        snprintf(name_buf, sizeof(name_buf), "mem_total_VRAM:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d total VRAM memory (bytes)", d);
        native_event_t *ev_mem_tot = &ntv_table.events[idx];
        ev_mem_tot->id = idx;
        ev_mem_tot->name = strdup(name_buf);
        ev_mem_tot->descr = strdup(descr_buf);
        ev_mem_tot->device = d;
        ev_mem_tot->value = 0;
        ev_mem_tot->mode = PAPI_MODE_READ;
        ev_mem_tot->variant = AMDSMI_MEM_TYPE_VRAM;
        ev_mem_tot->subvariant = 0;
        ev_mem_tot->open_func = open_simple;
        ev_mem_tot->close_func = close_simple;
        ev_mem_tot->start_func = start_simple;
        ev_mem_tot->stop_func = stop_simple;
        ev_mem_tot->access_func = access_amdsmi_mem_total;
        htable_insert(htable, ev_mem_tot->name, ev_mem_tot);
        idx++;

        // Used VRAM
        snprintf(name_buf, sizeof(name_buf), "mem_usage_VRAM:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d VRAM memory usage (bytes)", d);
        native_event_t *ev_mem_use = &ntv_table.events[idx];
        ev_mem_use->id = idx;
        ev_mem_use->name = strdup(name_buf);
        ev_mem_use->descr = strdup(descr_buf);
        ev_mem_use->device = d;
        ev_mem_use->value = 0;
        ev_mem_use->mode = PAPI_MODE_READ;
        ev_mem_use->variant = AMDSMI_MEM_TYPE_VRAM;
        ev_mem_use->subvariant = 0;
        ev_mem_use->open_func = open_simple;
        ev_mem_use->close_func = close_simple;
        ev_mem_use->start_func = start_simple;
        ev_mem_use->stop_func = stop_simple;
        ev_mem_use->access_func = access_amdsmi_mem_usage;
        htable_insert(htable, ev_mem_use->name, ev_mem_use);
        idx++;
    }

    /* GPU power metrics: average power, power cap, and cap range */
    for (int d = 0; d < device_count; ++d) {
        // Check support for power metrics on this device
        amdsmi_power_info_t pinfo;
        amdsmi_power_cap_info_t cinfo;
        amdsmi_status_t stat_avg = amdsmi_get_power_info_p(device_handles[d], &pinfo);
        amdsmi_status_t stat_cap = amdsmi_get_power_cap_info_p(device_handles[d], 0, &cinfo);
        if (stat_avg != AMDSMI_STATUS_SUCCESS && stat_cap != AMDSMI_STATUS_SUCCESS) {
            // Device supports neither power reading nor capping
            continue;
        }
        if (stat_avg == AMDSMI_STATUS_SUCCESS) {
            // Average power consumption (in Watts or microWatts)
            snprintf(name_buf, sizeof(name_buf), "power_average:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d average power consumption (W)", d);
            native_event_t *ev_pwr_avg = &ntv_table.events[idx];
            ev_pwr_avg->id = idx;
            ev_pwr_avg->name = strdup(name_buf);
            ev_pwr_avg->descr = strdup(descr_buf);
            ev_pwr_avg->device = d;
            ev_pwr_avg->value = 0;
            ev_pwr_avg->mode = PAPI_MODE_READ;
            ev_pwr_avg->variant = 0;
            ev_pwr_avg->subvariant = 0;
            ev_pwr_avg->open_func = open_simple;
            ev_pwr_avg->close_func = close_simple;
            ev_pwr_avg->start_func = start_simple;
            ev_pwr_avg->stop_func = stop_simple;
            ev_pwr_avg->access_func = access_amdsmi_power_average;
            htable_insert(htable, ev_pwr_avg->name, ev_pwr_avg);
            idx++;
        }
        if (stat_cap == AMDSMI_STATUS_SUCCESS) {
            // Current power cap limit
            snprintf(name_buf, sizeof(name_buf), "power_cap:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d current power cap (W)", d);
            native_event_t *ev_pcap = &ntv_table.events[idx];
            ev_pcap->id = idx;
            ev_pcap->name = strdup(name_buf);
            ev_pcap->descr = strdup(descr_buf);
            ev_pcap->device = d;
            ev_pcap->value = 0;
            ev_pcap->mode = PAPI_MODE_READ | PAPI_MODE_WRITE;
            ev_pcap->variant = 0;
            ev_pcap->subvariant = 0;
            ev_pcap->open_func = open_simple;
            ev_pcap->close_func = close_simple;
            ev_pcap->start_func = start_simple;
            ev_pcap->stop_func = stop_simple;
            ev_pcap->access_func = access_amdsmi_power_cap;
            htable_insert(htable, ev_pcap->name, ev_pcap);
            idx++;

            // Minimum allowed power cap
            snprintf(name_buf, sizeof(name_buf), "power_cap_range_min:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d minimum allowed power cap (W)", d);
            native_event_t *ev_pcap_min = &ntv_table.events[idx];
            ev_pcap_min->id = idx;
            ev_pcap_min->name = strdup(name_buf);
            ev_pcap_min->descr = strdup(descr_buf);
            ev_pcap_min->device = d;
            ev_pcap_min->value = 0;
            ev_pcap_min->mode = PAPI_MODE_READ;
            ev_pcap_min->variant = 1;  // variant 1 => min
            ev_pcap_min->subvariant = 0;
            ev_pcap_min->open_func = open_simple;
            ev_pcap_min->close_func = close_simple;
            ev_pcap_min->start_func = start_simple;
            ev_pcap_min->stop_func = stop_simple;
            ev_pcap_min->access_func = access_amdsmi_power_cap_range;
            htable_insert(htable, ev_pcap_min->name, ev_pcap_min);
            idx++;

            // Maximum allowed power cap
            snprintf(name_buf, sizeof(name_buf), "power_cap_range_max:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d maximum allowed power cap (W)", d);
            native_event_t *ev_pcap_max = &ntv_table.events[idx];
            ev_pcap_max->id = idx;
            ev_pcap_max->name = strdup(name_buf);
            ev_pcap_max->descr = strdup(descr_buf);
            ev_pcap_max->device = d;
            ev_pcap_max->value = 0;
            ev_pcap_max->mode = PAPI_MODE_READ;
            ev_pcap_max->variant = 2;  // variant 2 => max
            ev_pcap_max->subvariant = 0;
            ev_pcap_max->open_func = open_simple;
            ev_pcap_max->close_func = close_simple;
            ev_pcap_max->start_func = start_simple;
            ev_pcap_max->stop_func = stop_simple;
            ev_pcap_max->access_func = access_amdsmi_power_cap_range;
            htable_insert(htable, ev_pcap_max->name, ev_pcap_max);
            idx++;
        }
    }

    /* PCIe throughput and replay counter metrics */
    for (int d = 0; d < device_count; ++d) {
        // PCIe bytes sent
        snprintf(name_buf, sizeof(name_buf), "pci_throughput_sent:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d PCIe bytes sent per second", d);
        native_event_t *ev_pci_tx = &ntv_table.events[idx];
        ev_pci_tx->id = idx;
        ev_pci_tx->name = strdup(name_buf);
        ev_pci_tx->descr = strdup(descr_buf);
        ev_pci_tx->device = d;
        ev_pci_tx->value = 0;
        ev_pci_tx->mode = PAPI_MODE_READ;
        ev_pci_tx->variant = 0;  // 0 for "sent"
        ev_pci_tx->subvariant = 0;
        ev_pci_tx->open_func = open_simple;
        ev_pci_tx->close_func = close_simple;
        ev_pci_tx->start_func = start_simple;
        ev_pci_tx->stop_func = stop_simple;
        ev_pci_tx->access_func = access_amdsmi_pci_throughput;
        htable_insert(htable, ev_pci_tx->name, ev_pci_tx);
        idx++;

        // PCIe bytes received
        snprintf(name_buf, sizeof(name_buf), "pci_throughput_received:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d PCIe bytes received per second", d);
        native_event_t *ev_pci_rx = &ntv_table.events[idx];
        ev_pci_rx->id = idx;
        ev_pci_rx->name = strdup(name_buf);
        ev_pci_rx->descr = strdup(descr_buf);
        ev_pci_rx->device = d;
        ev_pci_rx->value = 0;
        ev_pci_rx->mode = PAPI_MODE_READ;
        ev_pci_rx->variant = 1;  // 1 for "received"
        ev_pci_rx->subvariant = 0;
        ev_pci_rx->open_func = open_simple;
        ev_pci_rx->close_func = close_simple;
        ev_pci_rx->start_func = start_simple;
        ev_pci_rx->stop_func = stop_simple;
        ev_pci_rx->access_func = access_amdsmi_pci_throughput;
        htable_insert(htable, ev_pci_rx->name, ev_pci_rx);
        idx++;

        // PCIe max packet size
        snprintf(name_buf, sizeof(name_buf), "pci_throughput_max_packet:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d PCIe max packet size (bytes)", d);
        native_event_t *ev_pci_pkt = &ntv_table.events[idx];
        ev_pci_pkt->id = idx;
        ev_pci_pkt->name = strdup(name_buf);
        ev_pci_pkt->descr = strdup(descr_buf);
        ev_pci_pkt->device = d;
        ev_pci_pkt->value = 0;
        ev_pci_pkt->mode = PAPI_MODE_READ;
        ev_pci_pkt->variant = 2;  // 2 for "max_packet"
        ev_pci_pkt->subvariant = 0;
        ev_pci_pkt->open_func = open_simple;
        ev_pci_pkt->close_func = close_simple;
        ev_pci_pkt->start_func = start_simple;
        ev_pci_pkt->stop_func = stop_simple;
        ev_pci_pkt->access_func = access_amdsmi_pci_throughput;
        htable_insert(htable, ev_pci_pkt->name, ev_pci_pkt);
        idx++;

        // PCIe replay counter (NAK counter)
        snprintf(name_buf, sizeof(name_buf), "pci_replay_counter:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d PCIe replay (NAK) counter", d);
        native_event_t *ev_pci_replay = &ntv_table.events[idx];
        ev_pci_replay->id = idx;
        ev_pci_replay->name = strdup(name_buf);
        ev_pci_replay->descr = strdup(descr_buf);
        ev_pci_replay->device = d;
        ev_pci_replay->value = 0;
        ev_pci_replay->mode = PAPI_MODE_READ;
        ev_pci_replay->variant = 0;
        ev_pci_replay->subvariant = 0;
        ev_pci_replay->open_func = open_simple;
        ev_pci_replay->close_func = close_simple;
        ev_pci_replay->start_func = start_simple;
        ev_pci_replay->stop_func = stop_simple;
        ev_pci_replay->access_func = access_amdsmi_pci_replay_counter;
        htable_insert(htable, ev_pci_replay->name, ev_pci_replay);
        idx++;
    }

    /* Additional GPU metrics and system information */
    /* GPU engine utilization metrics (gfx, umc, mm) */
    for (int d = 0; d < device_count; ++d) {
        amdsmi_engine_usage_t usage;
        if (amdsmi_get_gpu_activity_p(device_handles[d], &usage) != AMDSMI_STATUS_SUCCESS) {
            continue;
        }
        snprintf(name_buf, sizeof(name_buf), "gfx_activity:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d GFX engine activity (%%)", d);
        native_event_t *ev_gfx = &ntv_table.events[idx];
        ev_gfx->id = idx;
        ev_gfx->name = strdup(name_buf);
        ev_gfx->descr = strdup(descr_buf);
        ev_gfx->device = d;
        ev_gfx->value = 0;
        ev_gfx->mode = PAPI_MODE_READ;
        ev_gfx->variant = 0;
        ev_gfx->subvariant = 0;
        ev_gfx->open_func = open_simple;
        ev_gfx->close_func = close_simple;
        ev_gfx->start_func = start_simple;
        ev_gfx->stop_func = stop_simple;
        ev_gfx->access_func = access_amdsmi_gpu_activity;
        htable_insert(htable, ev_gfx->name, ev_gfx);
        idx++;

        snprintf(name_buf, sizeof(name_buf), "umc_activity:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d UMC engine activity (%%)", d);
        native_event_t *ev_umc = &ntv_table.events[idx];
        ev_umc->id = idx;
        ev_umc->name = strdup(name_buf);
        ev_umc->descr = strdup(descr_buf);
        ev_umc->device = d;
        ev_umc->value = 0;
        ev_umc->mode = PAPI_MODE_READ;
        ev_umc->variant = 1;
        ev_umc->subvariant = 0;
        ev_umc->open_func = open_simple;
        ev_umc->close_func = close_simple;
        ev_umc->start_func = start_simple;
        ev_umc->stop_func = stop_simple;
        ev_umc->access_func = access_amdsmi_gpu_activity;
        htable_insert(htable, ev_umc->name, ev_umc);
        idx++;

        snprintf(name_buf, sizeof(name_buf), "mm_activity:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d MM engine activity (%%)", d);
        native_event_t *ev_mm = &ntv_table.events[idx];
        ev_mm->id = idx;
        ev_mm->name = strdup(name_buf);
        ev_mm->descr = strdup(descr_buf);
        ev_mm->device = d;
        ev_mm->value = 0;
        ev_mm->mode = PAPI_MODE_READ;
        ev_mm->variant = 2;
        ev_mm->subvariant = 0;
        ev_mm->open_func = open_simple;
        ev_mm->close_func = close_simple;
        ev_mm->start_func = start_simple;
        ev_mm->stop_func = stop_simple;
        ev_mm->access_func = access_amdsmi_gpu_activity;
        htable_insert(htable, ev_mm->name, ev_mm);
        idx++;
    }

    /* GPU clock frequency levels */
    for (int d = 0; d < device_count; ++d) {
        amdsmi_frequencies_t f;
        if (amdsmi_get_clk_freq_p(device_handles[d], AMDSMI_CLK_TYPE_SYS, &f) != AMDSMI_STATUS_SUCCESS || f.num_supported == 0) {
            continue;
        }
        // Number of supported frequencies
        snprintf(name_buf, sizeof(name_buf), "clk_freq_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d number of supported GPU clock frequencies", d);
        native_event_t *ev_clk_count = &ntv_table.events[idx];
        ev_clk_count->id = idx;
        ev_clk_count->name = strdup(name_buf);
        ev_clk_count->descr = strdup(descr_buf);
        ev_clk_count->device = d;
        ev_clk_count->value = 0;
        ev_clk_count->mode = PAPI_MODE_READ;
        ev_clk_count->variant = 0;
        ev_clk_count->subvariant = 0;
        ev_clk_count->open_func = open_simple;
        ev_clk_count->close_func = close_simple;
        ev_clk_count->start_func = start_simple;
        ev_clk_count->stop_func = stop_simple;
        ev_clk_count->access_func = access_amdsmi_clk_freq;
        htable_insert(htable, ev_clk_count->name, ev_clk_count);
        idx++;

        // Current clock frequency
        snprintf(name_buf, sizeof(name_buf), "clk_freq_current:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d current GPU clock frequency (MHz)", d);
        native_event_t *ev_clk_cur = &ntv_table.events[idx];
        ev_clk_cur->id = idx;
        ev_clk_cur->name = strdup(name_buf);
        ev_clk_cur->descr = strdup(descr_buf);
        ev_clk_cur->device = d;
        ev_clk_cur->value = 0;
        ev_clk_cur->mode = PAPI_MODE_READ;
        ev_clk_cur->variant = 0;
        ev_clk_cur->subvariant = 1;
        ev_clk_cur->open_func = open_simple;
        ev_clk_cur->close_func = close_simple;
        ev_clk_cur->start_func = start_simple;
        ev_clk_cur->stop_func = stop_simple;
        ev_clk_cur->access_func = access_amdsmi_clk_freq;
        htable_insert(htable, ev_clk_cur->name, ev_clk_cur);
        idx++;

        // Supported frequency levels
        for (uint32_t fi = 0; fi < f.num_supported; ++fi) {
            snprintf(name_buf, sizeof(name_buf), "clk_freq_level_%u:device=%d", fi, d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d supported clock frequency level %u (MHz)", d, fi);
            native_event_t *ev_clk_lvl = &ntv_table.events[idx];
            ev_clk_lvl->id = idx;
            ev_clk_lvl->name = strdup(name_buf);
            ev_clk_lvl->descr = strdup(descr_buf);
            ev_clk_lvl->device = d;
            ev_clk_lvl->value = 0;
            ev_clk_lvl->mode = PAPI_MODE_READ;
            ev_clk_lvl->variant = 0;
            ev_clk_lvl->subvariant = fi + 2;
            ev_clk_lvl->open_func = open_simple;
            ev_clk_lvl->close_func = close_simple;
            ev_clk_lvl->start_func = start_simple;
            ev_clk_lvl->stop_func = stop_simple;
            ev_clk_lvl->access_func = access_amdsmi_clk_freq;
            htable_insert(htable, ev_clk_lvl->name, ev_clk_lvl);
            idx++;
        }
    }

    /* GPU identification and topology metrics */
    for (int d = 0; d < device_count; ++d) {
        uint16_t id16;
        uint64_t id64;
        amdsmi_virtualization_mode_t vmode;
        int32_t numa;
        // GPU ID
        if (amdsmi_get_gpu_id_p(device_handles[d], &id16) == AMDSMI_STATUS_SUCCESS) {
            snprintf(name_buf, sizeof(name_buf), "gpu_id:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d GPU identifier (Device ID)", d);
            native_event_t *ev_id = &ntv_table.events[idx];
            ev_id->id = idx;
            ev_id->name = strdup(name_buf);
            ev_id->descr = strdup(descr_buf);
            ev_id->device = d;
            ev_id->value = 0;
            ev_id->mode = PAPI_MODE_READ;
            ev_id->variant = 0;
            ev_id->subvariant = 0;
            ev_id->open_func = open_simple;
            ev_id->close_func = close_simple;
            ev_id->start_func = start_simple;
            ev_id->stop_func = stop_simple;
            ev_id->access_func = access_amdsmi_gpu_info;
            htable_insert(htable, ev_id->name, ev_id);
            idx++;
        }
        // GPU Revision
        if (amdsmi_get_gpu_revision_p(device_handles[d], &id16) == AMDSMI_STATUS_SUCCESS) {
            snprintf(name_buf, sizeof(name_buf), "gpu_revision:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d GPU revision ID", d);
            native_event_t *ev_rev = &ntv_table.events[idx];
            ev_rev->id = idx;
            ev_rev->name = strdup(name_buf);
            ev_rev->descr = strdup(descr_buf);
            ev_rev->device = d;
            ev_rev->value = 0;
            ev_rev->mode = PAPI_MODE_READ;
            ev_rev->variant = 1;
            ev_rev->subvariant = 0;
            ev_rev->open_func = open_simple;
            ev_rev->close_func = close_simple;
            ev_rev->start_func = start_simple;
            ev_rev->stop_func = stop_simple;
            ev_rev->access_func = access_amdsmi_gpu_info;
            htable_insert(htable, ev_rev->name, ev_rev);
            idx++;
        }
        // GPU Subsystem ID
        if (amdsmi_get_gpu_subsystem_id_p(device_handles[d], &id16) == AMDSMI_STATUS_SUCCESS) {
            snprintf(name_buf, sizeof(name_buf), "gpu_subsystem_id:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d GPU subsystem ID", d);
            native_event_t *ev_subid = &ntv_table.events[idx];
            ev_subid->id = idx;
            ev_subid->name = strdup(name_buf);
            ev_subid->descr = strdup(descr_buf);
            ev_subid->device = d;
            ev_subid->value = 0;
            ev_subid->mode = PAPI_MODE_READ;
            ev_subid->variant = 2;
            ev_subid->subvariant = 0;
            ev_subid->open_func = open_simple;
            ev_subid->close_func = close_simple;
            ev_subid->start_func = start_simple;
            ev_subid->stop_func = stop_simple;
            ev_subid->access_func = access_amdsmi_gpu_info;
            htable_insert(htable, ev_subid->name, ev_subid);
            idx++;
        }
        // GPU BDF ID
        if (amdsmi_get_gpu_bdf_id_p(device_handles[d], &id64) == AMDSMI_STATUS_SUCCESS) {
            snprintf(name_buf, sizeof(name_buf), "gpu_bdfid:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d GPU PCI BDF identifier", d);
            native_event_t *ev_bdf = &ntv_table.events[idx];
            ev_bdf->id = idx;
            ev_bdf->name = strdup(name_buf);
            ev_bdf->descr = strdup(descr_buf);
            ev_bdf->device = d;
            ev_bdf->value = 0;
            ev_bdf->mode = PAPI_MODE_READ;
            ev_bdf->variant = 3;
            ev_bdf->subvariant = 0;
            ev_bdf->open_func = open_simple;
            ev_bdf->close_func = close_simple;
            ev_bdf->start_func = start_simple;
            ev_bdf->stop_func = stop_simple;
            ev_bdf->access_func = access_amdsmi_gpu_info;
            htable_insert(htable, ev_bdf->name, ev_bdf);
            idx++;
        }
        // GPU Virtualization Mode
        if (amdsmi_get_gpu_virtualization_mode_p(device_handles[d], &vmode) == AMDSMI_STATUS_SUCCESS) {
            snprintf(name_buf, sizeof(name_buf), "gpu_virtualization_mode:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d GPU virtualization mode", d);
            native_event_t *ev_vmode = &ntv_table.events[idx];
            ev_vmode->id = idx;
            ev_vmode->name = strdup(name_buf);
            ev_vmode->descr = strdup(descr_buf);
            ev_vmode->device = d;
            ev_vmode->value = 0;
            ev_vmode->mode = PAPI_MODE_READ;
            ev_vmode->variant = 4;
            ev_vmode->subvariant = 0;
            ev_vmode->open_func = open_simple;
            ev_vmode->close_func = close_simple;
            ev_vmode->start_func = start_simple;
            ev_vmode->stop_func = stop_simple;
            ev_vmode->access_func = access_amdsmi_gpu_info;
            htable_insert(htable, ev_vmode->name, ev_vmode);
            idx++;
        }
        // GPU NUMA Node
        if (amdsmi_get_gpu_topo_numa_affinity_p(device_handles[d], &numa) == AMDSMI_STATUS_SUCCESS) {
            snprintf(name_buf, sizeof(name_buf), "numa_node:device=%d", d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d NUMA node", d);
            native_event_t *ev_numa = &ntv_table.events[idx];
            ev_numa->id = idx;
            ev_numa->name = strdup(name_buf);
            ev_numa->descr = strdup(descr_buf);
            ev_numa->device = d;
            ev_numa->value = 0;
            ev_numa->mode = PAPI_MODE_READ;
            ev_numa->variant = 5;
            ev_numa->subvariant = 0;
            ev_numa->open_func = open_simple;
            ev_numa->close_func = close_simple;
            ev_numa->start_func = start_simple;
            ev_numa->stop_func = stop_simple;
            ev_numa->access_func = access_amdsmi_gpu_info;
            htable_insert(htable, ev_numa->name, ev_numa);
            idx++;
        }
    }

    /* Energy consumption counter */
    for (int d = 0; d < device_count; ++d) {
        uint64_t energy = 0;
        float resolution = 0.0;
        uint64_t timestamp = 0;
        if (amdsmi_get_energy_count_p(device_handles[d], &energy, &resolution, &timestamp) != AMDSMI_STATUS_SUCCESS) {
            continue;
        }
        snprintf(name_buf, sizeof(name_buf), "energy_consumed:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d energy consumed (microJoules)", d);
        native_event_t *ev_energy = &ntv_table.events[idx];
        ev_energy->id = idx;
        ev_energy->name = strdup(name_buf);
        ev_energy->descr = strdup(descr_buf);
        ev_energy->device = d;
        ev_energy->value = 0;
        ev_energy->mode = PAPI_MODE_READ;
        ev_energy->variant = 0;
        ev_energy->subvariant = 0;
        ev_energy->open_func = open_simple;
        ev_energy->close_func = close_simple;
        ev_energy->start_func = start_simple;
        ev_energy->stop_func = stop_simple;
        ev_energy->access_func = access_amdsmi_energy_count;
        htable_insert(htable, ev_energy->name, ev_energy);
        idx++;
    }

    /* GPU power profile information */
    for (int d = 0; d < device_count; ++d) {
        amdsmi_power_profile_status_t profile_status;
        if (amdsmi_get_gpu_power_profile_presets_p(device_handles[d], 0, &profile_status) != AMDSMI_STATUS_SUCCESS) {
            continue;
        }
        snprintf(name_buf, sizeof(name_buf), "power_profiles_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d number of supported power profiles", d);
        native_event_t *ev_prof_count = &ntv_table.events[idx];
        ev_prof_count->id = idx;
        ev_prof_count->name = strdup(name_buf);
        ev_prof_count->descr = strdup(descr_buf);
        ev_prof_count->device = d;
        ev_prof_count->value = 0;
        ev_prof_count->mode = PAPI_MODE_READ;
        ev_prof_count->variant = 0;
        ev_prof_count->subvariant = 0;
        ev_prof_count->open_func = open_simple;
        ev_prof_count->close_func = close_simple;
        ev_prof_count->start_func = start_simple;
        ev_prof_count->stop_func = stop_simple;
        ev_prof_count->access_func = access_amdsmi_power_profile_status;
        htable_insert(htable, ev_prof_count->name, ev_prof_count);
        idx++;

        snprintf(name_buf, sizeof(name_buf), "power_profile_current:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d current power profile mask", d);
        native_event_t *ev_prof_curr = &ntv_table.events[idx];
        ev_prof_curr->id = idx;
        ev_prof_curr->name = strdup(name_buf);
        ev_prof_curr->descr = strdup(descr_buf);
        ev_prof_curr->device = d;
        ev_prof_curr->value = 0;
        ev_prof_curr->mode = PAPI_MODE_READ;
        ev_prof_curr->variant = 1;
        ev_prof_curr->subvariant = 0;
        ev_prof_curr->open_func = open_simple;
        ev_prof_curr->close_func = close_simple;
        ev_prof_curr->start_func = start_simple;
        ev_prof_curr->stop_func = stop_simple;
        ev_prof_curr->access_func = access_amdsmi_power_profile_status;
        htable_insert(htable, ev_prof_curr->name, ev_prof_curr);
        idx++;
    }

    ntv_table.count = idx;
    return PAPI_OK;
}

static int shutdown_event_table(void) {
    // Remove all events from hash table and free their names/descr
    for (int i = 0; i < ntv_table.count; ++i) {
        htable_delete(htable, ntv_table.events[i].name);
        papi_free(ntv_table.events[i].name);
        papi_free(ntv_table.events[i].descr);
    }
    papi_free(ntv_table.events);
    ntv_table.events = NULL;
    ntv_table.count = 0;
    return PAPI_OK;
}

static int init_device_table(void) {
    // Nothing to do (device_handles and device_count already set in amds_init)
    return PAPI_OK;
}

static int shutdown_device_table(void) {
    if (device_handles) {
        papi_free(device_handles);
        device_handles = NULL;
    }
    device_count = 0;
    return PAPI_OK;
}

/* Access function implementations (read/write operations for each event) */

static int access_amdsmi_temp_metric(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;  /* ensure device handle is valid */
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;
    }
    amdsmi_status_t status = amdsmi_get_temp_metric_p(device_handles[event->device],
                        (amdsmi_temperature_type_t) event->subvariant,
                        (amdsmi_temperature_metric_t) event->variant,
                        (int64_t *)&event->value);
    return (status == AMDSMI_STATUS_SUCCESS ? PAPI_OK : PAPI_EMISC);
}

static int access_amdsmi_fan_rpms(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;
    }
    int64_t speed = 0;
    amdsmi_status_t status = amdsmi_get_gpu_fan_rpms_p(device_handles[event->device],
                                                      event->subvariant, &speed);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = speed;
    return PAPI_OK;
}

static int access_amdsmi_fan_speed(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;  // writing fan speed not supported
    }
    int64_t val = 0;
    amdsmi_status_t status = amdsmi_get_gpu_fan_speed_p(device_handles[event->device],
                                                       event->subvariant, &val);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = val;
    return PAPI_OK;
}

static int access_amdsmi_mem_total(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;
    uint64_t data = 0;
    amdsmi_status_t status = amdsmi_get_total_memory_p(device_handles[event->device],
                                                       (amdsmi_memory_type_t) event->variant, &data);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

static int access_amdsmi_mem_usage(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;
    uint64_t data = 0;
    amdsmi_status_t status = amdsmi_get_memory_usage_p(device_handles[event->device],
                                                       (amdsmi_memory_type_t) event->variant, &data);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

static int access_amdsmi_power_cap(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode == PAPI_MODE_READ) {
        // Read current power cap
        amdsmi_power_cap_info_t info;
        amdsmi_status_t status = amdsmi_get_power_cap_info_p(device_handles[event->device], 0, &info);  // sensor index 0
        if (status != AMDSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
        event->value = (int64_t) info.power_cap;
        return PAPI_OK;
    } else if (mode == PAPI_MODE_WRITE) {
        // Set new power cap (value expected in microWatts if API uses uW)
        uint64_t new_cap = (uint64_t) event->value;
        amdsmi_status_t status = amdsmi_set_power_cap_p(device_handles[event->device], 0, new_cap);
        return (status == AMDSMI_STATUS_SUCCESS ? PAPI_OK : PAPI_EMISC);
    }
    return PAPI_ENOSUPP;
}

static int access_amdsmi_power_cap_range(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;
    amdsmi_power_cap_info_t info;
    amdsmi_status_t status = amdsmi_get_power_cap_info_p(device_handles[event->device], 0, &info);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    if (event->variant == 1) {
        event->value = (int64_t) info.min_power_cap;
    } else if (event->variant == 2) {
        event->value = (int64_t) info.max_power_cap;
    } else {
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int access_amdsmi_power_average(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;
    amdsmi_power_info_t power;
    amdsmi_status_t status = amdsmi_get_power_info_p(device_handles[event->device], &power);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    // Remove debug prints to avoid unnecessary console output
    #if 0
    printf("=== power info dump ===\n");
    printf("current_socket_power   : %u W\n", power.current_socket_power);
    printf("average_socket_power   : %u W\n", power.average_socket_power);
    printf("gfx_voltage            : %u mV\n", power.gfx_voltage);
    printf("soc_voltage            : %u mV\n", power.soc_voltage);
    printf("mem_voltage            : %u mV\n", power.mem_voltage);
    printf("power_limit            : %u W\n", power.power_limit);
    printf("========================\n");
    #endif
    event->value = (int64_t) power.average_socket_power;
    return PAPI_OK;
}

static int access_amdsmi_pci_throughput(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;
    uint64_t sent = 0, received = 0, max_pkt = 0;
    amdsmi_status_t status = amdsmi_get_gpu_pci_throughput_p(device_handles[event->device],
                                                             &sent, &received, &max_pkt);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    switch (event->variant) {
        case 0: event->value = (int64_t) sent; break;
        case 1: event->value = (int64_t) received; break;
        case 2: event->value = (int64_t) max_pkt; break;
        default: return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int access_amdsmi_pci_replay_counter(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;
    uint64_t counter = 0;
    amdsmi_status_t status = amdsmi_get_gpu_pci_replay_counter_p(device_handles[event->device], &counter);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) counter;
    return PAPI_OK;
}

static int access_amdsmi_clk_freq(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;
    amdsmi_frequencies_t freq_info;
    amdsmi_status_t status = amdsmi_get_clk_freq_p(device_handles[event->device], AMDSMI_CLK_TYPE_SYS, &freq_info);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    // Simplified: variant 0 -> count, 1 -> current frequency, >=2 -> specific index
    if (event->subvariant == 0) {
        event->value = freq_info.num_supported;
    } else if (event->subvariant == 1) {
        if (freq_info.num_supported > 0) {
            event->value = freq_info.frequency[0];  // assume first is current
        } else {
            event->value = 0;
        }
    } else {
        int idx = event->subvariant - 2;
        if (idx >= 0 && idx < (int)freq_info.num_supported) {
            event->value = freq_info.frequency[idx];
        } else {
            return PAPI_EMISC;
        }
    }
    return PAPI_OK;
}

static int access_amdsmi_gpu_metrics(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;
    amdsmi_gpu_metrics_t metrics;
    amdsmi_status_t status = amdsmi_get_gpu_metrics_info_p(device_handles[event->device], &metrics);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    // Parsing of metrics is not fully implemented; just return OK.
    // (In a full implementation, event->variant or subvariant would select a specific field of 'metrics'.)
    return PAPI_OK;
}

static int access_amdsmi_gpu_info(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;
    }
    amdsmi_status_t status;
    switch (event->variant) {
        case 0: {
            uint16_t id = 0;
            status = amdsmi_get_gpu_id_p(device_handles[event->device], &id);
            if (status == AMDSMI_STATUS_SUCCESS) {
                event->value = id;
            }
            break;
        }
        case 1: {
            uint16_t rev = 0;
            status = amdsmi_get_gpu_revision_p(device_handles[event->device], &rev);
            if (status == AMDSMI_STATUS_SUCCESS) {
                event->value = rev;
            }
            break;
        }
        case 2: {
            uint16_t subid = 0;
            status = amdsmi_get_gpu_subsystem_id_p(device_handles[event->device], &subid);
            if (status == AMDSMI_STATUS_SUCCESS) {
                event->value = subid;
            }
            break;
        }
        case 3: {
            uint64_t bdfid = 0;
            status = amdsmi_get_gpu_bdf_id_p(device_handles[event->device], &bdfid);
            if (status == AMDSMI_STATUS_SUCCESS) {
                event->value = (int64_t) bdfid;
            }
            break;
        }
        case 4: {
            amdsmi_virtualization_mode_t mode_val;
            status = amdsmi_get_gpu_virtualization_mode_p(device_handles[event->device], &mode_val);
            if (status == AMDSMI_STATUS_SUCCESS) {
                event->value = mode_val;
            }
            break;
        }
        case 5: {
            int32_t numa_node = -1;
            status = amdsmi_get_gpu_topo_numa_affinity_p(device_handles[event->device], &numa_node);
            if (status == AMDSMI_STATUS_SUCCESS) {
                event->value = numa_node;
            }
            break;
        }
        default:
            return PAPI_EMISC;
    }
    return (status == AMDSMI_STATUS_SUCCESS ? PAPI_OK : PAPI_EMISC);
}

static int access_amdsmi_gpu_activity(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;
    }
    amdsmi_engine_usage_t usage;
    amdsmi_status_t status = amdsmi_get_gpu_activity_p(device_handles[event->device], &usage);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    switch (event->variant) {
        case 0: event->value = usage.gfx_activity; break;
        case 1: event->value = usage.umc_activity; break;
        case 2: event->value = usage.mm_activity; break;
        default: return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int access_amdsmi_fan_speed_max(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;
    }
    int64_t max_speed = 0;
    amdsmi_status_t status = amdsmi_get_gpu_fan_speed_max_p(device_handles[event->device], event->subvariant, &max_speed);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = max_speed;
    return PAPI_OK;
}

static int access_amdsmi_pci_bandwidth(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;
    }
    amdsmi_pcie_bandwidth_t bw;
    amdsmi_status_t status = amdsmi_get_gpu_pci_bandwidth_p(device_handles[event->device], &bw);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    uint32_t cur_index = bw.transfer_rate.current;
    if (cur_index >= bw.transfer_rate.num_supported) {
        return PAPI_EMISC;
    }
    switch (event->variant) {
        case 0:
            event->value = bw.transfer_rate.num_supported;
            break;
        case 1:
            event->value = (int64_t) bw.transfer_rate.frequency[cur_index];
            break;
        case 2:
            event->value = bw.lanes[cur_index];
            break;
        default:
            return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int access_amdsmi_energy_count(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;
    }
    uint64_t energy = 0;
    float resolution = 0.0;
    uint64_t timestamp = 0;
    amdsmi_status_t status = amdsmi_get_energy_count_p(device_handles[event->device], &energy, &resolution, &timestamp);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    // Convert energy count to microJoules
    event->value = (int64_t) (energy * resolution);
    return PAPI_OK;
}

static int access_amdsmi_power_profile_status(int mode, void *arg) {
    native_event_t *event = (native_event_t *) arg;
    if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
        return PAPI_EMISC;
    }
    if (mode != PAPI_MODE_READ) {
        return PAPI_ENOSUPP;
    }
    amdsmi_power_profile_status_t status_info;
    amdsmi_status_t status = amdsmi_get_gpu_power_profile_presets_p(device_handles[event->device], 0, &status_info);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    if (event->variant == 0) {
        event->value = status_info.num_profiles;
    } else if (event->variant == 1) {
        event->value = (int64_t) status_info.current;
    } else {
        return PAPI_EMISC;
    }
    return PAPI_OK;
}
