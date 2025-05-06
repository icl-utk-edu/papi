/**
 * @file    amds.c
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#include "amds.h"
#define AMDS_PRIV_IMPL
#include "amds_priv.h"
#include <amd_smi/amdsmi.h>
#include "htable.h"
#include "papi.h"
#include "papi_memory.h"
#include <stdio.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <ctype.h>
#define MAX_EVENTS_PER_DEVICE 1024

// Pointers to AMD SMI library functions (dynamically loaded)
#include "amds_funcs.h"
#define DEFINE_AMDSMI(name, ret, args) ret(*name) args;
AMD_SMI_GPU_FUNCTIONS(DEFINE_AMDSMI)
#ifndef AMDSMI_DISABLE_ESMI
AMD_SMI_CPU_FUNCTIONS(DEFINE_AMDSMI)
#endif
#undef DEFINE_AMDSMI
// Global device list and count
static int32_t device_count = 0;
static amdsmi_processor_handle *device_handles = NULL;
static int32_t gpu_count = 0;
static int32_t cpu_count = 0;
static amdsmi_processor_handle **cpu_core_handles = NULL;
static uint32_t *cores_per_socket = NULL;
static void *amds_dlp = NULL;
static void *htable = NULL;
static char error_string[PAPI_MAX_STR_LEN + 1];
static uint32_t amdsmi_lib_major = 0;
static uint32_t amdsmi_lib_minor = 0;
// Forward declarations for internal helpers
static int load_amdsmi_sym(void);
static int init_device_table(void);
static int shutdown_device_table(void);
static int init_event_table(void);
static int shutdown_event_table(void);
static native_event_table_t ntv_table;
static native_event_table_t *ntv_table_p = NULL;

/* Internal state accessors */
int32_t amds_get_device_count(void) { return device_count; }
amdsmi_processor_handle *amds_get_device_handles(void) { return device_handles; }
int32_t amds_get_gpu_count(void) { return gpu_count; }
int32_t amds_get_cpu_count(void) { return cpu_count; }
amdsmi_processor_handle **amds_get_cpu_core_handles(void) {
  return cpu_core_handles;
}
uint32_t *amds_get_cores_per_socket(void) { return cores_per_socket; }
native_event_table_t *amds_get_ntv_table(void) { return ntv_table_p; }
void *amds_get_htable(void) { return htable; }
uint32_t amds_get_lib_major(void) { return amdsmi_lib_major; }

#define CHECK_EVENT_IDX(i)                                                     \
  do {                                                                        \
    if ((i) >= MAX_EVENTS_PER_DEVICE * device_count) {                         \
      return PAPI_ENOSUPP;                                                     \
    }                                                                         \
  } while (0)
  
#define REQ(sym) do { \
  if (!(sym)) { \
    snprintf(error_string, sizeof(error_string), "Missing required symbol: %s", #sym); \
    return PAPI_ENOSUPP; \
  } \
} while (0)

// Temporarily redirects stderr to /dev/null; returns dup of original fd (or -1 on failure)
static int silence_stderr_begin(void) {
  int devnull = open("/dev/null", O_WRONLY);
  if (devnull < 0)
    return -1;
  int saved = dup(STDERR_FILENO);
  if (saved < 0) {
    close(devnull);
    return -1;
  }
  (void)dup2(devnull, STDERR_FILENO);
  close(devnull);
  return saved;
}

// Restores stderr using the fd returned by silence_stderr_begin()
static void silence_stderr_end(int saved_fd) {
  if (saved_fd >= 0) {
    (void)dup2(saved_fd, STDERR_FILENO);
    close(saved_fd);
  }
}
// Simple open/close/start/stop functions (no special handling needed for most events)
static int open_simple(native_event_t *event) {
  (void)event;
  return PAPI_OK;
}
static int close_simple(native_event_t *event) {
  (void)event;
  return PAPI_OK;
}
static int start_simple(native_event_t *event) {
  (void)event;
  return PAPI_OK;
}
static int stop_simple(native_event_t *event) {
  (void)event;
  return PAPI_OK;
}

typedef struct {
  amdsmi_event_handle_t handle;
  uint64_t accum;
} counter_priv_t;

static int open_counter(native_event_t *event) {
  if (!amdsmi_gpu_create_counter_p)
    return PAPI_ENOSUPP;
  counter_priv_t *priv = (counter_priv_t *)papi_calloc(1, sizeof(counter_priv_t));
  if (!priv)
    return PAPI_ENOMEM;
  amdsmi_status_t status = amdsmi_gpu_create_counter_p(
      device_handles[event->device], (amdsmi_event_type_t)event->variant,
      &priv->handle);
  if (status != AMDSMI_STATUS_SUCCESS) {
    papi_free(priv);
    return PAPI_ENOSUPP;
  }
  event->priv = priv;
  return PAPI_OK;
}

static int close_counter(native_event_t *event) {
  counter_priv_t *priv = (counter_priv_t *)event->priv;
  if (priv) {
    if (amdsmi_gpu_destroy_counter_p)
      amdsmi_gpu_destroy_counter_p(priv->handle);
    papi_free(priv);
    event->priv = NULL;
  }
  return PAPI_OK;
}

static int start_counter(native_event_t *event) {
  counter_priv_t *priv = (counter_priv_t *)event->priv;
  if (!priv || !amdsmi_gpu_control_counter_p)
    return PAPI_ENOSUPP;
  priv->accum = 0;
  amdsmi_status_t status = amdsmi_gpu_control_counter_p(
      priv->handle, AMDSMI_CNTR_CMD_START, NULL);
  return (status == AMDSMI_STATUS_SUCCESS) ? PAPI_OK : PAPI_ENOSUPP;
}

static int stop_counter(native_event_t *event) {
  counter_priv_t *priv = (counter_priv_t *)event->priv;
  if (!priv || !amdsmi_gpu_control_counter_p)
    return PAPI_ENOSUPP;
  amdsmi_status_t status =
      amdsmi_gpu_control_counter_p(priv->handle, AMDSMI_CNTR_CMD_STOP, NULL);
  return (status == AMDSMI_STATUS_SUCCESS) ? PAPI_OK : PAPI_ENOSUPP;
}

static int access_amdsmi_gpu_counter(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  counter_priv_t *priv = (counter_priv_t *)event->priv;
  if (!priv || !amdsmi_gpu_read_counter_p)
    return PAPI_ENOSUPP;
  amdsmi_counter_value_t val;
  if (amdsmi_gpu_read_counter_p(priv->handle, &val) != AMDSMI_STATUS_SUCCESS)
    return PAPI_ENOSUPP;
  priv->accum += val.value;
  event->value = priv->accum;
  return PAPI_OK;
}

// Replace any non-alphanumeric characters with '_' to build safe event names
static void sanitize_name(const char *src, char *dst, size_t len) {
  if (len == 0) return;
  size_t j = 0;
  for (size_t i = 0; src[i] && j < len - 1; ++i) {
    char c = src[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9'))
      dst[j++] = c;
    else
      dst[j++] = '_';
  }
  dst[j] = '\0';
}

static void sanitize_description_text(char *str) {
  if (!str)
    return;
  for (size_t i = 0; str[i]; ++i) {
    unsigned char c = (unsigned char)str[i];
    if (c == '\n' || c == '\r' || c == '\t')
      str[i] = ' ';
    else if (!isprint(c))
      str[i] = '?';
  }
}

static const char *display_or_empty(const char *str) {
  return (str && str[0]) ? str : "<empty>";
}

// Dynamic load of AMD SMI library symbols
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
    snprintf(error_string, sizeof(error_string), "dlopen(\"%s\"): %s", so_path,
             dlerror());
    return PAPI_ENOSUPP;
  }
  // Resolve required function symbols
  amdsmi_init_p = sym("amdsmi_init", NULL);
  amdsmi_shut_down_p = sym("amdsmi_shut_down", NULL);
  amdsmi_get_socket_handles_p = sym("amdsmi_get_socket_handles", NULL);
  amdsmi_get_processor_handles_by_type_p =
      sym("amdsmi_get_processor_handles_by_type", NULL);
  amdsmi_get_processor_handles_p =
      sym("amdsmi_get_processor_handles", NULL);
  amdsmi_get_processor_info_p =
      sym("amdsmi_get_processor_info", NULL);
  amdsmi_get_processor_type_p =
      sym("amdsmi_get_processor_type", NULL);
  amdsmi_get_socket_info_p = sym("amdsmi_get_socket_info", NULL);
  // Sensors
  amdsmi_get_temp_metric_p = sym("amdsmi_get_temp_metric", NULL);
  amdsmi_get_gpu_fan_rpms_p = sym("amdsmi_get_gpu_fan_rpms", NULL);
  amdsmi_get_gpu_fan_speed_p = sym("amdsmi_get_gpu_fan_speed", NULL);
  amdsmi_get_gpu_fan_speed_max_p = sym("amdsmi_get_gpu_fan_speed_max", NULL);
  // Memory
  amdsmi_get_total_memory_p =
      sym("amdsmi_get_gpu_memory_total", "amdsmi_get_total_memory");
  amdsmi_get_memory_usage_p =
      sym("amdsmi_get_gpu_memory_usage", "amdsmi_get_memory_usage");
  // Utilization / activity
  amdsmi_get_gpu_activity_p =
      sym("amdsmi_get_gpu_activity", "amdsmi_get_engine_usage");
  amdsmi_get_utilization_count_p =
      sym("amdsmi_get_utilization_count", NULL);
  // Power
  amdsmi_get_power_info_p = sym("amdsmi_get_power_info", NULL);
  amdsmi_get_power_cap_info_p = sym("amdsmi_get_power_cap_info", NULL);
  amdsmi_set_power_cap_p =
      sym("amdsmi_set_power_cap", "amdsmi_dev_set_power_cap");
  // PCIe
  amdsmi_get_gpu_pci_throughput_p = sym("amdsmi_get_gpu_pci_throughput", NULL);
  amdsmi_get_gpu_pci_replay_counter_p =
      sym("amdsmi_get_gpu_pci_replay_counter", NULL);
  // Clocks
  amdsmi_get_clk_freq_p = sym("amdsmi_get_clk_freq", NULL);
  amdsmi_get_clock_info_p = sym("amdsmi_get_clock_info", NULL);
  amdsmi_set_clk_freq_p = sym("amdsmi_set_clk_freq", NULL);
  // GPU metrics
  amdsmi_get_gpu_metrics_info_p = sym("amdsmi_get_gpu_metrics_info", NULL);
  // Identification and other queries
  amdsmi_get_gpu_id_p = sym("amdsmi_get_gpu_id", NULL);
  amdsmi_get_gpu_revision_p = sym("amdsmi_get_gpu_revision", NULL);
  amdsmi_get_gpu_subsystem_id_p = sym("amdsmi_get_gpu_subsystem_id", NULL);
#if AMDSMI_LIB_VERSION_MAJOR >= 25
  amdsmi_get_gpu_virtualization_mode_p =
      sym("amdsmi_get_gpu_virtualization_mode", NULL);
#endif
  amdsmi_get_gpu_process_isolation_p =
      sym("amdsmi_get_gpu_process_isolation", NULL);
  amdsmi_get_gpu_xcd_counter_p = sym("amdsmi_get_gpu_xcd_counter", NULL);
  amdsmi_get_gpu_pci_bandwidth_p = sym("amdsmi_get_gpu_pci_bandwidth", NULL);
  amdsmi_get_gpu_bdf_id_p = sym("amdsmi_get_gpu_bdf_id", NULL);
  amdsmi_get_gpu_topo_numa_affinity_p =
      sym("amdsmi_get_gpu_topo_numa_affinity", NULL);
  amdsmi_get_energy_count_p = sym("amdsmi_get_energy_count", NULL);
  amdsmi_get_gpu_power_profile_presets_p =
      sym("amdsmi_get_gpu_power_profile_presets", NULL);
  amdsmi_get_violation_status_p =
      sym("amdsmi_get_violation_status", NULL);
  // Additional read-only queries
  amdsmi_get_lib_version_p = sym("amdsmi_get_lib_version", NULL);
  amdsmi_get_gpu_driver_info_p = sym("amdsmi_get_gpu_driver_info", NULL);
  amdsmi_get_gpu_asic_info_p = sym("amdsmi_get_gpu_asic_info", NULL);
  amdsmi_get_gpu_board_info_p = sym("amdsmi_get_gpu_board_info", NULL);
  amdsmi_get_fw_info_p = sym("amdsmi_get_fw_info", NULL);
  amdsmi_get_gpu_vbios_info_p = sym("amdsmi_get_gpu_vbios_info", NULL);
  amdsmi_get_gpu_device_uuid_p = sym("amdsmi_get_gpu_device_uuid", NULL);
#if AMDSMI_LIB_VERSION_MAJOR >= 25
  amdsmi_get_gpu_enumeration_info_p =
      sym("amdsmi_get_gpu_enumeration_info", NULL);
#endif
  amdsmi_get_gpu_vendor_name_p = sym("amdsmi_get_gpu_vendor_name", NULL);
  amdsmi_get_gpu_vram_vendor_p = sym("amdsmi_get_gpu_vram_vendor", NULL);
  amdsmi_get_gpu_subsystem_name_p = sym("amdsmi_get_gpu_subsystem_name", NULL);
  amdsmi_get_link_metrics_p = sym("amdsmi_get_link_metrics", NULL);
  amdsmi_get_gpu_process_list_p = sym("amdsmi_get_gpu_process_list", NULL);
  amdsmi_topo_get_numa_node_number_p =
      sym("amdsmi_topo_get_numa_node_number", NULL);
  amdsmi_topo_get_link_weight_p = sym("amdsmi_topo_get_link_weight", NULL);
  amdsmi_topo_get_link_type_p = sym("amdsmi_topo_get_link_type", NULL);
  amdsmi_topo_get_p2p_status_p = sym("amdsmi_topo_get_p2p_status", NULL);
  amdsmi_is_P2P_accessible_p = sym("amdsmi_is_P2P_accessible", NULL);
  amdsmi_get_link_topology_nearest_p =
      sym("amdsmi_get_link_topology_nearest", NULL);
  amdsmi_get_gpu_device_bdf_p = sym("amdsmi_get_gpu_device_bdf", NULL);
  amdsmi_get_gpu_ecc_enabled_p = sym("amdsmi_get_gpu_ecc_enabled", NULL);
  amdsmi_get_gpu_total_ecc_count_p =
      sym("amdsmi_get_gpu_total_ecc_count", NULL);
  amdsmi_get_gpu_ecc_count_p = sym("amdsmi_get_gpu_ecc_count", NULL);
  amdsmi_get_gpu_ecc_status_p = sym("amdsmi_get_gpu_ecc_status", NULL);
  amdsmi_get_gpu_compute_partition_p =
      sym("amdsmi_get_gpu_compute_partition", NULL);
  amdsmi_get_gpu_memory_partition_p =
        sym("amdsmi_get_gpu_memory_partition", NULL);
#if AMDSMI_LIB_VERSION_MAJOR >= 25
    amdsmi_get_gpu_memory_partition_config_p =
        sym("amdsmi_get_gpu_memory_partition_config", NULL);
#endif
  amdsmi_is_gpu_memory_partition_supported_p =
      sym("amdsmi_is_gpu_memory_partition_supported", NULL);
  amdsmi_get_gpu_memory_reserved_pages_p =
      sym("amdsmi_get_gpu_memory_reserved_pages", NULL);
  amdsmi_get_gpu_kfd_info_p = sym("amdsmi_get_gpu_kfd_info", NULL);
  amdsmi_get_gpu_metrics_header_info_p =
        sym("amdsmi_get_gpu_metrics_header_info", NULL);
#if AMDSMI_LIB_VERSION_MAJOR >= 25
    amdsmi_get_gpu_xgmi_link_status_p =
        sym("amdsmi_get_gpu_xgmi_link_status", NULL);
#endif
  amdsmi_get_xgmi_info_p = sym("amdsmi_get_xgmi_info", NULL);
  amdsmi_gpu_xgmi_error_status_p =
      sym("amdsmi_gpu_xgmi_error_status", NULL);
  amdsmi_get_gpu_accelerator_partition_profile_p =
      sym("amdsmi_get_gpu_accelerator_partition_profile", NULL);
  amdsmi_get_gpu_cache_info_p = sym("amdsmi_get_gpu_cache_info", NULL);
  amdsmi_get_gpu_mem_overdrive_level_p =
      sym("amdsmi_get_gpu_mem_overdrive_level", NULL);
  amdsmi_get_gpu_od_volt_curve_regions_p =
      sym("amdsmi_get_gpu_od_volt_curve_regions", NULL);
  amdsmi_get_gpu_od_volt_info_p = sym("amdsmi_get_gpu_od_volt_info", NULL);
  amdsmi_get_gpu_overdrive_level_p =
      sym("amdsmi_get_gpu_overdrive_level", NULL);
  amdsmi_get_gpu_perf_level_p = sym("amdsmi_get_gpu_perf_level", NULL);
  amdsmi_get_gpu_pm_metrics_info_p =
      sym("amdsmi_get_gpu_pm_metrics_info", NULL);
  amdsmi_is_gpu_power_management_enabled_p =
      sym("amdsmi_is_gpu_power_management_enabled", NULL);
  amdsmi_get_gpu_ras_feature_info_p =
      sym("amdsmi_get_gpu_ras_feature_info", NULL);
  amdsmi_get_gpu_ras_block_features_enabled_p =
      sym("amdsmi_get_gpu_ras_block_features_enabled", NULL);
  amdsmi_gpu_validate_ras_eeprom_p =
      sym("amdsmi_gpu_validate_ras_eeprom", NULL);
  amdsmi_get_gpu_reg_table_info_p = sym("amdsmi_get_gpu_reg_table_info", NULL);
  amdsmi_get_gpu_volt_metric_p = sym("amdsmi_get_gpu_volt_metric", NULL);
  amdsmi_get_gpu_vram_info_p = sym("amdsmi_get_gpu_vram_info", NULL);
  amdsmi_get_gpu_vram_usage_p = sym("amdsmi_get_gpu_vram_usage", NULL);
  amdsmi_get_pcie_info_p = sym("amdsmi_get_pcie_info", NULL);
  amdsmi_get_processor_count_from_handles_p =
      sym("amdsmi_get_processor_count_from_handles", NULL);
  amdsmi_get_soc_pstate_p = sym("amdsmi_get_soc_pstate", NULL);
  amdsmi_get_xgmi_plpd_p = sym("amdsmi_get_xgmi_plpd", NULL);
  amdsmi_get_gpu_bad_page_info_p = sym("amdsmi_get_gpu_bad_page_info", NULL);
  amdsmi_get_gpu_bad_page_threshold_p =
      sym("amdsmi_get_gpu_bad_page_threshold", NULL);
  amdsmi_get_power_info_v2_p = sym("amdsmi_get_power_info_v2", NULL);
  amdsmi_init_gpu_event_notification_p =
      sym("amdsmi_init_gpu_event_notification", NULL);
  amdsmi_set_gpu_event_notification_mask_p =
      sym("amdsmi_set_gpu_event_notification_mask", NULL);
  amdsmi_get_gpu_event_notification_p =
      sym("amdsmi_get_gpu_event_notification", NULL);
  amdsmi_stop_gpu_event_notification_p =
      sym("amdsmi_stop_gpu_event_notification", NULL);
  amdsmi_gpu_counter_group_supported_p =
      sym("amdsmi_gpu_counter_group_supported", NULL);
  amdsmi_get_gpu_available_counters_p =
      sym("amdsmi_get_gpu_available_counters", NULL);
  amdsmi_gpu_create_counter_p =
      sym("amdsmi_gpu_create_counter", NULL);
  amdsmi_gpu_control_counter_p =
      sym("amdsmi_gpu_control_counter", NULL);
  amdsmi_gpu_read_counter_p = sym("amdsmi_gpu_read_counter", NULL);
  amdsmi_gpu_destroy_counter_p =
      sym("amdsmi_gpu_destroy_counter", NULL);
  amdsmi_get_minmax_bandwidth_between_processors_p =
      sym("amdsmi_get_minmax_bandwidth_between_processors", NULL);
#ifndef AMDSMI_DISABLE_ESMI
  /* CPU functions */
  amdsmi_get_cpu_handles_p = sym("amdsmi_get_cpu_handles", NULL);
  amdsmi_get_cpucore_handles_p = sym("amdsmi_get_cpucore_handles", NULL);
  amdsmi_get_cpu_socket_power_p = sym("amdsmi_get_cpu_socket_power", NULL);
  amdsmi_get_cpu_socket_power_cap_p =
      sym("amdsmi_get_cpu_socket_power_cap", NULL);
  amdsmi_get_cpu_socket_power_cap_max_p =
      sym("amdsmi_get_cpu_socket_power_cap_max", NULL);
  amdsmi_get_cpu_core_energy_p = sym("amdsmi_get_cpu_core_energy", NULL);
  amdsmi_get_cpu_socket_energy_p = sym("amdsmi_get_cpu_socket_energy", NULL);
  amdsmi_get_cpu_smu_fw_version_p = sym("amdsmi_get_cpu_smu_fw_version", NULL);
  amdsmi_get_threads_per_core_p = sym("amdsmi_get_threads_per_core", NULL);
  amdsmi_get_cpu_family_p = sym("amdsmi_get_cpu_family", NULL);
  amdsmi_get_cpu_model_p = sym("amdsmi_get_cpu_model", NULL);
  amdsmi_get_cpu_core_boostlimit_p =
      sym("amdsmi_get_cpu_core_boostlimit", NULL);
  amdsmi_get_cpu_socket_current_active_freq_limit_p =
      sym("amdsmi_get_cpu_socket_current_active_freq_limit", NULL);
  amdsmi_get_cpu_socket_freq_range_p =
      sym("amdsmi_get_cpu_socket_freq_range", NULL);
  amdsmi_get_cpu_core_current_freq_limit_p =
      sym("amdsmi_get_cpu_core_current_freq_limit", NULL);
  amdsmi_get_cpu_cclk_limit_p = sym("amdsmi_get_cpu_cclk_limit", NULL);
  amdsmi_get_cpu_current_io_bandwidth_p =
      sym("amdsmi_get_cpu_current_io_bandwidth", NULL);
  amdsmi_get_cpu_current_xgmi_bw_p =
      sym("amdsmi_get_cpu_current_xgmi_bw", NULL);
  amdsmi_get_cpu_ddr_bw_p = sym("amdsmi_get_cpu_ddr_bw", NULL);
  amdsmi_get_cpu_fclk_mclk_p = sym("amdsmi_get_cpu_fclk_mclk", NULL);
  amdsmi_get_cpu_hsmp_driver_version_p =
      sym("amdsmi_get_cpu_hsmp_driver_version", NULL);
  amdsmi_get_cpu_hsmp_proto_ver_p = sym("amdsmi_get_cpu_hsmp_proto_ver", NULL);
  amdsmi_get_cpu_prochot_status_p =
      sym("amdsmi_get_cpu_prochot_status", NULL);
  amdsmi_get_cpu_pwr_svi_telemetry_all_rails_p =
      sym("amdsmi_get_cpu_pwr_svi_telemetry_all_rails", NULL);
  amdsmi_get_cpu_dimm_temp_range_and_refresh_rate_p =
      sym("amdsmi_get_cpu_dimm_temp_range_and_refresh_rate", NULL);
  amdsmi_get_cpu_dimm_power_consumption_p =
      sym("amdsmi_get_cpu_dimm_power_consumption", NULL);
  amdsmi_get_cpu_dimm_thermal_sensor_p =
      sym("amdsmi_get_cpu_dimm_thermal_sensor", NULL);
#endif

// Validate required symbols we call unconditionally later 
REQ(amdsmi_init_p);
REQ(amdsmi_shut_down_p);
REQ(amdsmi_get_socket_handles_p);
REQ(amdsmi_get_processor_handles_by_type_p);
#ifndef AMDSMI_DISABLE_ESMI
REQ(amdsmi_get_cpu_handles_p);
#endif
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
  if (cpu_core_handles) {
    for (int s = 0; s < cpu_count; ++s) {
      if (cpu_core_handles[s])
        papi_free(cpu_core_handles[s]);
    }
    papi_free(cpu_core_handles);
    cpu_core_handles = NULL;
  }
  if (cores_per_socket) {
    papi_free(cores_per_socket);
    cores_per_socket = NULL;
  }
  device_count = 0;
  gpu_count = 0;
  cpu_count = 0;
  return PAPI_OK;
}

int amds_init(void) {
  // Check if already initialized to avoid expensive re-initialization
  if (device_handles != NULL && device_count > 0)
    return PAPI_OK; // Already initialized
  int papi_errno = load_amdsmi_sym();
  if (papi_errno != PAPI_OK)
    return papi_errno;
  // AMDSMI_INIT_AMD_CPUS
  amdsmi_status_t status = amdsmi_init_p(AMDSMI_INIT_AMD_GPUS);
  if (status != AMDSMI_STATUS_SUCCESS) {
    snprintf(error_string, sizeof(error_string), "amdsmi_init failed");
    return PAPI_ENOSUPP;
  }
  if (amdsmi_get_lib_version_p) {
    amdsmi_version_t vinfo;
    if (amdsmi_get_lib_version_p(&vinfo) == AMDSMI_STATUS_SUCCESS) {
      amdsmi_lib_major = vinfo.major;
      amdsmi_lib_minor = vinfo.minor;
    }
  }
  htable_init(&htable);
  // Discover GPU and CPU devices
  uint32_t socket_count = 0;
  status = amdsmi_get_socket_handles_p(&socket_count, NULL);
  if (status != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
    snprintf(error_string, sizeof(error_string),
             "Error discovering sockets or no AMD socket found.");
    papi_errno = PAPI_ENOEVNT;
    goto fn_fail;
  }
  amdsmi_socket_handle *sockets = (amdsmi_socket_handle *)papi_calloc(
      socket_count, sizeof(amdsmi_socket_handle));
  if (!sockets) {
    papi_errno = PAPI_ENOMEM;
    goto fn_fail;
  }
  status = amdsmi_get_socket_handles_p(&socket_count, sockets);
  if (status != AMDSMI_STATUS_SUCCESS) {
    snprintf(error_string, sizeof(error_string),
             "Error getting socket handles.");
    papi_free(sockets);
    papi_errno = PAPI_ENOSUPP;
    goto fn_fail;
  }
  device_count = 0;
  uint32_t total_gpu_count = 0;
  for (uint32_t s = 0; s < socket_count; ++s) {
    uint32_t gpu_count_local = 0;
    processor_type_t proc_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
    amdsmi_status_t st = amdsmi_get_processor_handles_by_type_p(
        sockets[s], proc_type, NULL, &gpu_count_local);
    if (st == AMDSMI_STATUS_SUCCESS)
      total_gpu_count += gpu_count_local;
  }
  uint32_t total_cpu_count = 0;
#ifndef AMDSMI_DISABLE_ESMI
  status = amdsmi_get_cpu_handles_p(&total_cpu_count, NULL);
  if (status != AMDSMI_STATUS_SUCCESS)
    total_cpu_count = 0;
#endif
  if (total_gpu_count == 0 && total_cpu_count == 0) {
    snprintf(error_string, sizeof(error_string),
             "No AMD GPU or CPU devices found.");
    papi_errno = PAPI_ENOEVNT;
    papi_free(sockets);
    goto fn_fail;
  }
  device_handles = (amdsmi_processor_handle *)papi_calloc(
      total_gpu_count + total_cpu_count, sizeof(*device_handles));
  if (!device_handles) {
    papi_errno = PAPI_ENOMEM;
    snprintf(error_string, sizeof(error_string),
             "Memory allocation error for device handles.");
    papi_free(sockets);
    goto fn_fail;
  }
  // Retrieve GPU processor handles for each socket - optimized to reduce
  // allocations
  for (uint32_t s = 0; s < socket_count; ++s) {
    uint32_t gpu_count_local = 0;
    processor_type_t proc_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
    status = amdsmi_get_processor_handles_by_type_p(sockets[s], proc_type, NULL,
                                                    &gpu_count_local);
    if (status != AMDSMI_STATUS_SUCCESS || gpu_count_local == 0)
      continue; // no GPU on this socket or error
    // Use the main device_handles array directly to avoid extra allocation
    amdsmi_processor_handle *gpu_handles = &device_handles[device_count];
    status = amdsmi_get_processor_handles_by_type_p(
        sockets[s], proc_type, gpu_handles, &gpu_count_local);
    if (status == AMDSMI_STATUS_SUCCESS)
      device_count += gpu_count_local;
  }
  papi_free(sockets);
  // Set gpu_count for use in event table initialization
  gpu_count = device_count; // All devices added so far are GPUs
#ifndef AMDSMI_DISABLE_ESMI
  // Retrieve CPU socket handles
  amdsmi_processor_handle *cpu_handles = NULL;
  if (total_cpu_count > 0) {
    cpu_handles = (amdsmi_processor_handle *)papi_calloc(
        total_cpu_count, sizeof(amdsmi_processor_handle));
    if (!cpu_handles) {
      papi_errno = PAPI_ENOMEM;
      snprintf(error_string, sizeof(error_string),
               "Memory allocation error for CPU handles.");
      goto fn_fail;
    }
    status = amdsmi_get_cpu_handles_p(&total_cpu_count, cpu_handles);
    if (status != AMDSMI_STATUS_SUCCESS) {
      papi_free(cpu_handles);
      cpu_handles = NULL;
      total_cpu_count = 0;
    }
  }
  if (cpu_handles) {
    for (uint32_t i = 0; i < total_cpu_count; ++i) {
      device_handles[device_count++] = cpu_handles[i];
    }
    papi_free(cpu_handles);
  }
#endif
  // Set global GPU/CPU counts
  gpu_count = total_gpu_count;
  cpu_count = total_cpu_count;
  // Retrieve CPU core handles for each CPU socket
  if (cpu_count > 0) {
    cpu_core_handles = (amdsmi_processor_handle **)papi_calloc(
        cpu_count, sizeof(amdsmi_processor_handle *));
    cores_per_socket = (uint32_t *)papi_calloc(cpu_count, sizeof(uint32_t));
    if (!cpu_core_handles || !cores_per_socket) {
      papi_errno = PAPI_ENOMEM;
      snprintf(error_string, sizeof(error_string),
               "Memory allocation error for CPU core handles.");
      if (cpu_core_handles)
        papi_free(cpu_core_handles);
      if (cores_per_socket)
        papi_free(cores_per_socket);
      goto fn_fail;
    }
    for (uint32_t s = 0; s < cpu_count; ++s) {
      uint32_t core_count = 0;
      amdsmi_status_t st = amdsmi_get_processor_handles_by_type_p(
          device_handles[gpu_count + s], AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE,
          NULL, &core_count);
      if (st != AMDSMI_STATUS_SUCCESS || core_count == 0) {
        cores_per_socket[s] = 0;
        cpu_core_handles[s] = NULL;
        continue;
      }
      cpu_core_handles[s] = (amdsmi_processor_handle *)papi_calloc(
          core_count, sizeof(amdsmi_processor_handle));
      if (!cpu_core_handles[s]) {
        papi_errno = PAPI_ENOMEM;
        snprintf(error_string, sizeof(error_string),
                 "Memory allocation error for CPU core handles on socket %u.",
                 s);
        for (uint32_t t = 0; t < s; ++t) {
          if (cpu_core_handles[t])
            papi_free(cpu_core_handles[t]);
        }
        papi_free(cpu_core_handles);
        papi_free(cores_per_socket);
        goto fn_fail;
      }
      st = amdsmi_get_processor_handles_by_type_p(
          device_handles[gpu_count + s], AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE,
          cpu_core_handles[s], &core_count);
      if (st != AMDSMI_STATUS_SUCCESS) {
        papi_free(cpu_core_handles[s]);
        cpu_core_handles[s] = NULL;
        cores_per_socket[s] = 0;
      } else {
        cores_per_socket[s] = core_count;
      }
    }
  }
  // Initialize the native event table for all discovered metrics
  papi_errno = init_event_table();
  if (papi_errno != PAPI_OK) {
    snprintf(error_string, sizeof(error_string),
             "Error while initializing the native event table.");
    goto fn_fail;
  }
  ntv_table_p = &ntv_table;
  return PAPI_OK;
fn_fail:
  htable_shutdown(htable);
  if (device_handles) {
    papi_free(device_handles);
    device_handles = NULL;
    device_count = 0;
  }
  // sockets already freed if allocated
  if (cpu_core_handles) {
    for (int s = 0; s < cpu_count; ++s) {
      if (cpu_core_handles[s])
        papi_free(cpu_core_handles[s]);
    }
    papi_free(cpu_core_handles);
    cpu_core_handles = NULL;
  }
  if (cores_per_socket) {
    papi_free(cores_per_socket);
    cores_per_socket = NULL;
  }
  amdsmi_shut_down_p();
  return papi_errno;
}

int amds_shutdown(void) {
  // Tear down our tables first
  shutdown_event_table();
  shutdown_device_table();
  htable_shutdown(htable);
  htable = NULL;

  // Tell AMD SMI to shut down if the symbol exists
  amdsmi_status_t st = AMDSMI_STATUS_SUCCESS;
  if (amdsmi_shut_down_p)
    st = amdsmi_shut_down_p();

  // Unload the shared library if we loaded it
  if (amds_dlp) {
    dlclose(amds_dlp);
    amds_dlp = NULL;
  }

  // Clear function pointers so a future init can't call stale symbols
  #define NULLIFY(name, ret, args) name = NULL;
  AMD_SMI_GPU_FUNCTIONS(NULLIFY)
  #ifndef AMDSMI_DISABLE_ESMI
  AMD_SMI_CPU_FUNCTIONS(NULLIFY)
  #endif
  #undef NULLIFY

  // Reset a few globals used by init paths
  device_count = 0;
  gpu_count = 0;
  cpu_count = 0;
  ntv_table_p = NULL;
  amdsmi_lib_major = 0;

  return (st == AMDSMI_STATUS_SUCCESS) ? PAPI_OK : PAPI_EMISC;
}


int amds_err_get_last(const char **err_string) {
  if (err_string)
    *err_string = error_string;
  return PAPI_OK;
}

// Helper to add a new event entry to ntv_table
static int add_event(int *idx_ptr, const char *name, const char *descr, int device,
                     uint32_t variant, uint32_t subvariant, int mode,
                     amds_accessor_t access_func) {
  native_event_t *ev = &ntv_table.events[*idx_ptr];
  ev->id = *idx_ptr;
  ev->name = strdup(name);
  ev->descr = strdup(descr);
  if (!ev->name || !ev->descr)
    return PAPI_ENOMEM;
  ev->device = device;
  ev->value = 0;
  ev->mode = mode;
  ev->variant = variant;
  ev->subvariant = subvariant;
  ev->priv = NULL;
  ev->open_func = open_simple;
  ev->close_func = close_simple;
  ev->start_func = start_simple;
  ev->stop_func = stop_simple;
  ev->access_func = access_func;
  htable_insert(htable, ev->name, ev);
  (*idx_ptr)++;
  return PAPI_OK;
}

static int add_counter_event(int *idx_ptr, const char *name, const char *descr,
                             int device, uint32_t variant, uint32_t subvariant) {
  int papi_errno = add_event(idx_ptr, name, descr, device, variant, subvariant,
                             PAPI_MODE_READ, access_amdsmi_gpu_counter);
  if (papi_errno != PAPI_OK)
    return papi_errno;
  native_event_t *ev = &ntv_table.events[*idx_ptr - 1];
  ev->open_func = open_counter;
  ev->close_func = close_counter;
  ev->start_func = start_counter;
  ev->stop_func = stop_counter;
  return PAPI_OK;
}

// Initialize native event table: enumerate all supported events
static int init_event_table(void) {
  // Check if event table is already initialized
  if (ntv_table.count > 0 && ntv_table.events != NULL)
    return PAPI_OK; // Already initialized, skip expensive rebuild
  ntv_table.count = 0;
  int idx = 0;
  // Safety check - if no devices, return early
  if (device_count <= 0) {
    ntv_table.events = NULL;
    return PAPI_OK;
  }
  // Keep original allocation approach
  ntv_table.events = (native_event_t *)papi_calloc(
      MAX_EVENTS_PER_DEVICE * device_count, sizeof(native_event_t));
  if (!ntv_table.events)
    return PAPI_ENOMEM;
  char name_buf[PAPI_MAX_STR_LEN];
  char descr_buf[PAPI_MAX_STR_LEN];
  // Define sensor arrays first
  amdsmi_temperature_type_t temp_sensors[] = {
      AMDSMI_TEMPERATURE_TYPE_EDGE,  AMDSMI_TEMPERATURE_TYPE_JUNCTION,
      AMDSMI_TEMPERATURE_TYPE_VRAM,  AMDSMI_TEMPERATURE_TYPE_HBM_0,
      AMDSMI_TEMPERATURE_TYPE_HBM_1, AMDSMI_TEMPERATURE_TYPE_HBM_2,
      AMDSMI_TEMPERATURE_TYPE_HBM_3, AMDSMI_TEMPERATURE_TYPE_PLX};
  const int num_temp_sensors =
      sizeof(temp_sensors) / sizeof(temp_sensors[0]);
  const amdsmi_temperature_metric_t temp_metrics[] = {
      AMDSMI_TEMP_CURRENT,        AMDSMI_TEMP_MAX,           AMDSMI_TEMP_MIN,
      AMDSMI_TEMP_MAX_HYST,       AMDSMI_TEMP_MIN_HYST,      AMDSMI_TEMP_CRITICAL,
      AMDSMI_TEMP_CRITICAL_HYST,  AMDSMI_TEMP_EMERGENCY,     AMDSMI_TEMP_EMERGENCY_HYST,
      AMDSMI_TEMP_CRIT_MIN,       AMDSMI_TEMP_CRIT_MIN_HYST, AMDSMI_TEMP_OFFSET,
      AMDSMI_TEMP_LOWEST,         AMDSMI_TEMP_HIGHEST};
  const char *temp_metric_names[] = {
      "temp_current",       "temp_max",           "temp_min",
      "temp_max_hyst",      "temp_min_hyst",      "temp_critical",
      "temp_critical_hyst", "temp_emergency",     "temp_emergency_hyst",
      "temp_crit_min",      "temp_crit_min_hyst", "temp_offset",
      "temp_lowest",        "temp_highest"};
  // Temperature sensors - device-level cache + individual testing
  for (int d = 0; d < gpu_count; ++d) {
    // Safety check for device handle
    if (!device_handles || !device_handles[d])
      continue;

    // GPU cache info events
    if (amdsmi_get_gpu_cache_info_p) {
      amdsmi_gpu_cache_info_t cache_info;
      if (amdsmi_get_gpu_cache_info_p(device_handles[d], &cache_info) ==
          AMDSMI_STATUS_SUCCESS) {
        for (uint32_t i = 0; i < cache_info.num_cache_types; ++i) {
          CHECK_EVENT_IDX(idx);
          uint32_t level = cache_info.cache[i].cache_level;
          uint32_t prop = cache_info.cache[i].cache_properties;
          char type_str[8] = "cache";
          if ((prop & AMDSMI_CACHE_PROPERTY_INST_CACHE) &&
              !(prop & AMDSMI_CACHE_PROPERTY_DATA_CACHE)) {
            strcpy(type_str, "icache");
          } else if ((prop & AMDSMI_CACHE_PROPERTY_DATA_CACHE) &&
                     !(prop & AMDSMI_CACHE_PROPERTY_INST_CACHE)) {
            strcpy(type_str, "dcache");
          } else {
            strcpy(type_str, "cache");
          }
          snprintf(name_buf, sizeof(name_buf), "L%u_%s_size:device=%d", level,
                   type_str, d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d L%u %s size (bytes)", d, level,
                   (strcmp(type_str, "cache") == 0 ? "cache"
                     : (strcmp(type_str, "icache") == 0 ? "instruction cache"
                                                        : "data cache")));
          if (add_event(&idx, name_buf, descr_buf, d, 0, i, PAPI_MODE_READ,
                        access_amdsmi_cache_stat) != PAPI_OK)
            return PAPI_ENOMEM;

          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "L%u_%s_cu_shared:device=%d",
                   level, type_str, d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d L%u %s max CUs sharing", d, level, type_str);
          if (add_event(&idx, name_buf, descr_buf, d, 1, i, PAPI_MODE_READ,
                        access_amdsmi_cache_stat) != PAPI_OK)
            return PAPI_ENOMEM;

          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "L%u_%s_instances:device=%d",
                   level, type_str, d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d L%u %s instances", d, level, type_str);
          if (add_event(&idx, name_buf, descr_buf, d, 2, i, PAPI_MODE_READ,
                        access_amdsmi_cache_stat) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    // GPU VRAM info events
    if (amdsmi_get_gpu_vram_info_p) {
      amdsmi_vram_info_t vram_info;
      if (amdsmi_get_gpu_vram_info_p(device_handles[d], &vram_info) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "vram_bus_width:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d VRAM bus width (bits)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_vram_width) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "vram_size_bytes:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d VRAM size (bytes)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_vram_size) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "vram_type:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d VRAM type id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_vram_type) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "vram_vendor_id:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d VRAM vendor id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_vram_vendor) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    // PCIe information events
    if (amdsmi_get_pcie_info_p) {
      amdsmi_pcie_info_t pcie_info;
      if (amdsmi_get_pcie_info_p(device_handles[d], &pcie_info) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_max_width:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d maximum PCIe link width (lanes)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_max_speed:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d maximum PCIe link speed (GT/s)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_interface_version:device=%d",
                 d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe interface version", d);
        if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_slot_type:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe slot type", d);
        if (add_event(&idx, name_buf, descr_buf, d, 3, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

#if AMDSMI_LIB_VERSION_MAJOR >= 25
        if (amdsmi_lib_major >= 25) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "pcie_max_interface_version:device=%d", d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d maximum PCIe interface version", d);
          if (add_event(&idx, name_buf, descr_buf, d, 4, 0, PAPI_MODE_READ,
                        access_amdsmi_pcie_info) != PAPI_OK)
            return PAPI_ENOMEM;
        }
#endif

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_width:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current PCIe link width (lanes)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 5, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_speed:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current PCIe link speed (MT/s)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 6, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_bandwidth:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d instantaneous PCIe bandwidth (Mb/s)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 7, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_replay_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d PCIe replay count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 8, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_l0_to_recovery_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe L0->recovery count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 9, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_replay_rollover_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe replay rollover count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 10, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_nak_sent_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe NAK sent count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 11, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_nak_received_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe NAK received count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 12, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_other_end_recovery_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe other-end recovery count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 13, 0, PAPI_MODE_READ,
                      access_amdsmi_pcie_info) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    // GPU Overdrive level events
    if (amdsmi_get_gpu_overdrive_level_p) {
      uint32_t od_val;
      if (amdsmi_get_gpu_overdrive_level_p(device_handles[d], &od_val) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "gpu_overdrive_percent:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d GPU core clock overdrive (%%)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_overdrive_level) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    if (amdsmi_get_gpu_mem_overdrive_level_p) {
      uint32_t od_val;
      if (amdsmi_get_gpu_mem_overdrive_level_p(device_handles[d], &od_val) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "gpu_mem_overdrive_percent:device=%d",
                 d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d GPU memory clock overdrive (%%)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_mem_overdrive_level) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    // GPU performance level event
    if (amdsmi_get_gpu_perf_level_p) {
      amdsmi_dev_perf_level_t perf;
      if (amdsmi_get_gpu_perf_level_p(device_handles[d], &perf) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "perf_level:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current performance level", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_perf_level) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    // GPU PM metrics count event (available in lib version 25+)
    if (amdsmi_lib_major >= 25 && amdsmi_get_gpu_pm_metrics_info_p) {
      amdsmi_name_value_t *metrics = NULL;
      uint32_t mcount = 0;

      int saved_stderr = silence_stderr_begin();
      amdsmi_status_t st = amdsmi_get_gpu_pm_metrics_info_p(device_handles[d],
                                                            &metrics, &mcount);
      silence_stderr_end(saved_stderr);

      if (st == AMDSMI_STATUS_SUCCESS && mcount > 0) {
        if (idx >= MAX_EVENTS_PER_DEVICE * device_count && metrics)
          free(metrics);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pm_metrics_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d number of PM metrics available", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_pm_metrics_count) != PAPI_OK) {
          if (metrics) free(metrics);
          return PAPI_ENOMEM;
        }

        for (uint32_t i = 0; i < mcount; ++i) {
          if (idx >= MAX_EVENTS_PER_DEVICE * device_count) {
            if (metrics) free(metrics);
            CHECK_EVENT_IDX(idx);
          }
          char metric_name[MAX_AMDSMI_NAME_LENGTH];
          sanitize_name(metrics[i].name, metric_name, sizeof(metric_name));
          snprintf(name_buf, sizeof(name_buf), "pm_%s:device=%d", metric_name, d);
          snprintf(descr_buf, sizeof(descr_buf), "Device %d PM metric %s", d,
                   metrics[i].name);
          if (add_event(&idx, name_buf, descr_buf, d, i, 0, PAPI_MODE_READ,
                        access_amdsmi_pm_metric_value) != PAPI_OK) {
            if (metrics) free(metrics);
            return PAPI_ENOMEM;
          }
        }
      }
      if (metrics)
        free(metrics);
    }
    if (amdsmi_is_gpu_power_management_enabled_p) {
      bool enabled = false;
      if (amdsmi_is_gpu_power_management_enabled_p(device_handles[d], &enabled) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pm_enabled:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d power management enabled", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_pm_enabled) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    // GPU RAS feature (ECC schema) event
    if (amdsmi_get_gpu_ras_feature_info_p) {
      amdsmi_ras_feature_t ras;
      if (amdsmi_get_gpu_ras_feature_info_p(device_handles[d], &ras) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "ecc_correction_mask:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d ECC correction features mask", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_ras_ecc_schema) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "ras_eeprom_version:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d RAS EEPROM version", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_ras_eeprom_version) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    if (amdsmi_gpu_validate_ras_eeprom_p) {
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "ras_eeprom_valid:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d RAS EEPROM validation status", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_ras_eeprom_validate) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    if (amdsmi_get_gpu_ras_block_features_enabled_p) {
      amdsmi_gpu_block_t blocks[] = {
          AMDSMI_GPU_BLOCK_UMC,   AMDSMI_GPU_BLOCK_SDMA,   AMDSMI_GPU_BLOCK_GFX,
          AMDSMI_GPU_BLOCK_MMHUB, AMDSMI_GPU_BLOCK_ATHUB, AMDSMI_GPU_BLOCK_PCIE_BIF,
          AMDSMI_GPU_BLOCK_HDP,   AMDSMI_GPU_BLOCK_XGMI_WAFL, AMDSMI_GPU_BLOCK_DF,
          AMDSMI_GPU_BLOCK_SMN,   AMDSMI_GPU_BLOCK_SEM,   AMDSMI_GPU_BLOCK_MP0,
          AMDSMI_GPU_BLOCK_MP1,   AMDSMI_GPU_BLOCK_FUSE,  AMDSMI_GPU_BLOCK_MCA,
          AMDSMI_GPU_BLOCK_VCN,   AMDSMI_GPU_BLOCK_JPEG,  AMDSMI_GPU_BLOCK_IH,
          AMDSMI_GPU_BLOCK_MPIO};
      const char *block_names[] = {
          "umc",       "sdma", "gfx",  "mmhub", "athub", "pcie_bif", "hdp",
          "xgmi_wafl", "df",   "smn",  "sem",   "mp0",   "mp1",      "fuse",
          "mca",       "vcn",  "jpeg", "ih",    "mpio"};
      size_t nb = sizeof(blocks) / sizeof(blocks[0]);
      for (size_t bi = 0; bi < nb; ++bi) {
        amdsmi_ras_err_state_t st;
        if (amdsmi_get_gpu_ras_block_features_enabled_p(
                device_handles[d], blocks[bi], &st) == AMDSMI_STATUS_SUCCESS) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "ras_block_%s_state:device=%d",
                   block_names[bi], d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d RAS state for %s block", d, block_names[bi]);
          if (add_event(&idx, name_buf, descr_buf, d, (uint32_t)blocks[bi], 0,
                        PAPI_MODE_READ, access_amdsmi_ras_block_state) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }

    /* ECC related events */
    if (amdsmi_get_gpu_total_ecc_count_p) {
      amdsmi_error_count_t ec;
      if (amdsmi_get_gpu_total_ecc_count_p(device_handles[d], &ec) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "ecc_total_correctable:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d total correctable ECC errors", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_ecc_total) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "ecc_total_uncorrectable:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d total uncorrectable ECC errors", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_ecc_total) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "ecc_total_deferred:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d total deferred ECC errors", d);
        if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_ecc_total) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_gpu_ecc_enabled_p) {
      uint64_t mask = 0;
      if (amdsmi_get_gpu_ecc_enabled_p(device_handles[d], &mask) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "ecc_enabled_mask:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d ECC enabled block mask", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_ecc_enabled_mask) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_gpu_ecc_count_p) {
      amdsmi_gpu_block_t eblocks[] = {
          AMDSMI_GPU_BLOCK_UMC,   AMDSMI_GPU_BLOCK_SDMA,   AMDSMI_GPU_BLOCK_GFX,
          AMDSMI_GPU_BLOCK_MMHUB, AMDSMI_GPU_BLOCK_ATHUB, AMDSMI_GPU_BLOCK_PCIE_BIF,
          AMDSMI_GPU_BLOCK_HDP,   AMDSMI_GPU_BLOCK_XGMI_WAFL, AMDSMI_GPU_BLOCK_DF,
          AMDSMI_GPU_BLOCK_SMN,   AMDSMI_GPU_BLOCK_SEM,   AMDSMI_GPU_BLOCK_MP0,
          AMDSMI_GPU_BLOCK_MP1,   AMDSMI_GPU_BLOCK_FUSE,  AMDSMI_GPU_BLOCK_MCA,
          AMDSMI_GPU_BLOCK_VCN,   AMDSMI_GPU_BLOCK_JPEG,  AMDSMI_GPU_BLOCK_IH,
          AMDSMI_GPU_BLOCK_MPIO};
      const char *eblock_names[] = {
          "umc",       "sdma", "gfx",  "mmhub", "athub", "pcie_bif", "hdp",
          "xgmi_wafl", "df",   "smn",  "sem",   "mp0",   "mp1",      "fuse",
          "mca",       "vcn",  "jpeg", "ih",    "mpio"};
      size_t nb = sizeof(eblocks) / sizeof(eblocks[0]);
      for (size_t bi = 0; bi < nb; ++bi) {
        amdsmi_error_count_t ec;
        if (amdsmi_get_gpu_ecc_count_p(device_handles[d], eblocks[bi], &ec) ==
            AMDSMI_STATUS_SUCCESS) {
          for (uint32_t v = 0; v < 3; ++v) {
            CHECK_EVENT_IDX(idx);
            const char *suf =
                (v == 0) ? "correctable" : (v == 1) ? "uncorrectable" : "deferred";
            snprintf(name_buf, sizeof(name_buf),
                     "ecc_%s_%s:device=%d", eblock_names[bi], suf, d);
            snprintf(descr_buf, sizeof(descr_buf),
                     "Device %d %s %s ECC errors", d, eblock_names[bi], suf);
            if (add_event(&idx, name_buf, descr_buf, d, v,
                          (uint32_t)eblocks[bi], PAPI_MODE_READ,
                          access_amdsmi_ecc_block) != PAPI_OK)
              return PAPI_ENOMEM;
          }
        }
      }
    }

    if (amdsmi_get_gpu_ecc_status_p) {
      amdsmi_gpu_block_t eblocks[] = {
          AMDSMI_GPU_BLOCK_UMC,   AMDSMI_GPU_BLOCK_SDMA,   AMDSMI_GPU_BLOCK_GFX,
          AMDSMI_GPU_BLOCK_MMHUB, AMDSMI_GPU_BLOCK_ATHUB, AMDSMI_GPU_BLOCK_PCIE_BIF,
          AMDSMI_GPU_BLOCK_HDP,   AMDSMI_GPU_BLOCK_XGMI_WAFL, AMDSMI_GPU_BLOCK_DF,
          AMDSMI_GPU_BLOCK_SMN,   AMDSMI_GPU_BLOCK_SEM,   AMDSMI_GPU_BLOCK_MP0,
          AMDSMI_GPU_BLOCK_MP1,   AMDSMI_GPU_BLOCK_FUSE,  AMDSMI_GPU_BLOCK_MCA,
          AMDSMI_GPU_BLOCK_VCN,   AMDSMI_GPU_BLOCK_JPEG,  AMDSMI_GPU_BLOCK_IH,
          AMDSMI_GPU_BLOCK_MPIO};
      const char *eblock_names[] = {
          "umc",       "sdma", "gfx",  "mmhub", "athub", "pcie_bif", "hdp",
          "xgmi_wafl", "df",   "smn",  "sem",   "mp0",   "mp1",      "fuse",
          "mca",       "vcn",  "jpeg", "ih",    "mpio"};
      size_t nb = sizeof(eblocks) / sizeof(eblocks[0]);
      for (size_t bi = 0; bi < nb; ++bi) {
        amdsmi_ras_err_state_t st;
        if (amdsmi_get_gpu_ecc_status_p(device_handles[d], eblocks[bi], &st) ==
            AMDSMI_STATUS_SUCCESS) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "ecc_%s_status:device=%d",
                   eblock_names[bi], d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d ECC status for %s block", d, eblock_names[bi]);
          if (add_event(&idx, name_buf, descr_buf, d, 0,
                        (uint32_t)eblocks[bi], PAPI_MODE_READ,
                        access_amdsmi_ecc_status) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }

    // GPU voltage metrics events
    if (amdsmi_get_gpu_volt_metric_p) {
      const char *sensor_names[] = {"vddgfx",  "vddmem", "vddsoc", "vddio",
                                    "vddmisc", "vdd",    "vdd2",   "vddboard"};
      const amdsmi_voltage_metric_t metrics[] = {
          AMDSMI_VOLT_CURRENT, AMDSMI_VOLT_MAX,      AMDSMI_VOLT_MIN_CRIT,
          AMDSMI_VOLT_MIN,     AMDSMI_VOLT_MAX_CRIT, AMDSMI_VOLT_AVERAGE,
          AMDSMI_VOLT_LOWEST,  AMDSMI_VOLT_HIGHEST};
      const char *metric_names[] = {"current", "max",      "min_crit",
                                    "min",     "max_crit", "average",
                                    "lowest",  "highest"};
      const uint32_t max_sensors = 8;
      for (uint32_t s = 0; s < max_sensors; ++s) {
        int64_t dummy = 0;
        amdsmi_status_t st = amdsmi_get_gpu_volt_metric_p(
            device_handles[d], (amdsmi_voltage_type_t)s, AMDSMI_VOLT_CURRENT,
            &dummy);
        if (st != AMDSMI_STATUS_SUCCESS)
          continue;
        for (uint32_t m = 0; m < sizeof(metrics) / sizeof(metrics[0]); ++m) {
          st = amdsmi_get_gpu_volt_metric_p(
              device_handles[d], (amdsmi_voltage_type_t)s, metrics[m], &dummy);
          if (st != AMDSMI_STATUS_SUCCESS)
            continue;
          CHECK_EVENT_IDX(idx);
          const char *sname =
              (s < sizeof(sensor_names) / sizeof(sensor_names[0]))
                  ? sensor_names[s]
                  : "sensor";
          char sensor_buf[32];
          if (strcmp(sname, "sensor") == 0) {
            snprintf(sensor_buf, sizeof(sensor_buf), "sensor%u", s);
            sname = sensor_buf;
          }
          snprintf(name_buf, sizeof(name_buf), "voltage_%s_%s:device=%d", sname,
                   metric_names[m], d);
          snprintf(descr_buf, sizeof(descr_buf), "Device %d %s %s voltage (mV)",
                   d, sname, metric_names[m]);
          if (add_event(&idx, name_buf, descr_buf, d, metrics[m], s, PAPI_MODE_READ,
                        access_amdsmi_voltage) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    // GPU OD voltage curve region events
    if (amdsmi_get_gpu_od_volt_curve_regions_p) {
      uint32_t num_regions = 0;
      amdsmi_status_t st = amdsmi_get_gpu_od_volt_curve_regions_p(
          device_handles[d], &num_regions, NULL);
      if (st == AMDSMI_STATUS_SUCCESS && num_regions > 0) {
        amdsmi_freq_volt_region_t *regs =
            (amdsmi_freq_volt_region_t *)papi_calloc(
                num_regions, sizeof(amdsmi_freq_volt_region_t));
        if (regs) {
          st = amdsmi_get_gpu_od_volt_curve_regions_p(device_handles[d],
                                                     &num_regions, regs);
          if (st == AMDSMI_STATUS_SUCCESS) {
            if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
              papi_free(regs);
            CHECK_EVENT_IDX(idx);
            snprintf(name_buf, sizeof(name_buf), "volt_curve_regions:device=%d",
                     d);
            snprintf(descr_buf, sizeof(descr_buf),
                     "Device %d number of voltage curve regions", d);
            if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                          access_amdsmi_od_volt_regions_count) != PAPI_OK) {
              papi_free(regs);
              return PAPI_ENOMEM;
            }

            for (uint32_t r = 0; r < num_regions; ++r) {
              if (idx + 4 > MAX_EVENTS_PER_DEVICE * device_count)
                papi_free(regs);
              CHECK_EVENT_IDX(idx + 4);

              snprintf(name_buf, sizeof(name_buf),
                       "volt_curve_freq_min:device=%d:region=%u", d, r);
              snprintf(descr_buf, sizeof(descr_buf),
                       "Device %d voltage curve region %u frequency lower bound",
                       d, r);
              if (add_event(&idx, name_buf, descr_buf, d, 0, r, PAPI_MODE_READ,
                            access_amdsmi_od_volt_curve_range) != PAPI_OK) {
                papi_free(regs);
                return PAPI_ENOMEM;
              }

              snprintf(name_buf, sizeof(name_buf),
                       "volt_curve_freq_max:device=%d:region=%u", d, r);
              snprintf(descr_buf, sizeof(descr_buf),
                       "Device %d voltage curve region %u frequency upper bound",
                       d, r);
              if (add_event(&idx, name_buf, descr_buf, d, 1, r, PAPI_MODE_READ,
                            access_amdsmi_od_volt_curve_range) != PAPI_OK) {
                papi_free(regs);
                return PAPI_ENOMEM;
              }

              snprintf(name_buf, sizeof(name_buf),
                       "volt_curve_volt_min:device=%d:region=%u", d, r);
              snprintf(descr_buf, sizeof(descr_buf),
                       "Device %d voltage curve region %u voltage lower bound",
                       d, r);
              if (add_event(&idx, name_buf, descr_buf, d, 2, r, PAPI_MODE_READ,
                            access_amdsmi_od_volt_curve_range) != PAPI_OK) {
                papi_free(regs);
                return PAPI_ENOMEM;
              }

              snprintf(name_buf, sizeof(name_buf),
                       "volt_curve_volt_max:device=%d:region=%u", d, r);
              snprintf(descr_buf, sizeof(descr_buf),
                       "Device %d voltage curve region %u voltage upper bound",
                       d, r);
              if (add_event(&idx, name_buf, descr_buf, d, 3, r, PAPI_MODE_READ,
                            access_amdsmi_od_volt_curve_range) != PAPI_OK) {
                papi_free(regs);
                return PAPI_ENOMEM;
              }
            }
          }
          papi_free(regs);
        }
      }
    }
    if (amdsmi_get_gpu_od_volt_info_p) {
      amdsmi_od_volt_freq_data_t info;
      if (amdsmi_get_gpu_od_volt_info_p(device_handles[d], &info) ==
          AMDSMI_STATUS_SUCCESS) {
        if (idx + 8 + 2 * AMDSMI_NUM_VOLTAGE_CURVE_POINTS >
            MAX_EVENTS_PER_DEVICE * device_count)
          CHECK_EVENT_IDX(idx + 8 + 2 * AMDSMI_NUM_VOLTAGE_CURVE_POINTS);
        snprintf(name_buf, sizeof(name_buf), "od_curr_sclk_min:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current SCLK frequency lower bound", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_od_volt_info) != PAPI_OK)
          return PAPI_ENOMEM;

        snprintf(name_buf, sizeof(name_buf), "od_curr_sclk_max:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current SCLK frequency upper bound", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_od_volt_info) != PAPI_OK)
          return PAPI_ENOMEM;

        snprintf(name_buf, sizeof(name_buf), "od_curr_mclk_min:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current MCLK frequency lower bound", d);
        if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_od_volt_info) != PAPI_OK)
          return PAPI_ENOMEM;

        snprintf(name_buf, sizeof(name_buf), "od_curr_mclk_max:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current MCLK frequency upper bound", d);
        if (add_event(&idx, name_buf, descr_buf, d, 3, 0, PAPI_MODE_READ,
                      access_amdsmi_od_volt_info) != PAPI_OK)
          return PAPI_ENOMEM;

        snprintf(name_buf, sizeof(name_buf), "od_sclk_limit_min:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d SCLK frequency limit lower bound", d);
        if (add_event(&idx, name_buf, descr_buf, d, 4, 0, PAPI_MODE_READ,
                      access_amdsmi_od_volt_info) != PAPI_OK)
          return PAPI_ENOMEM;

        snprintf(name_buf, sizeof(name_buf), "od_sclk_limit_max:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d SCLK frequency limit upper bound", d);
        if (add_event(&idx, name_buf, descr_buf, d, 5, 0, PAPI_MODE_READ,
                      access_amdsmi_od_volt_info) != PAPI_OK)
          return PAPI_ENOMEM;

        snprintf(name_buf, sizeof(name_buf), "od_mclk_limit_min:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d MCLK frequency limit lower bound", d);
        if (add_event(&idx, name_buf, descr_buf, d, 6, 0, PAPI_MODE_READ,
                      access_amdsmi_od_volt_info) != PAPI_OK)
          return PAPI_ENOMEM;

        snprintf(name_buf, sizeof(name_buf), "od_mclk_limit_max:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d MCLK frequency limit upper bound", d);
        if (add_event(&idx, name_buf, descr_buf, d, 7, 0, PAPI_MODE_READ,
                      access_amdsmi_od_volt_info) != PAPI_OK)
          return PAPI_ENOMEM;

        for (uint32_t p = 0; p < AMDSMI_NUM_VOLTAGE_CURVE_POINTS; ++p) {
          CHECK_EVENT_IDX(idx + 2);
          snprintf(name_buf, sizeof(name_buf),
                   "volt_curve_point_freq:device=%d:point=%u", d, p);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d voltage curve point %u frequency", d, p);
          if (add_event(&idx, name_buf, descr_buf, d, 8, p, PAPI_MODE_READ,
                        access_amdsmi_od_volt_info) != PAPI_OK)
            return PAPI_ENOMEM;

          snprintf(name_buf, sizeof(name_buf),
                   "volt_curve_point_volt:device=%d:point=%u", d, p);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d voltage curve point %u voltage", d, p);
          if (add_event(&idx, name_buf, descr_buf, d, 9, p, PAPI_MODE_READ,
                        access_amdsmi_od_volt_info) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    // GPU SoC P-state policy events
    if (amdsmi_get_soc_pstate_p) {
      amdsmi_dpm_policy_t policy;
      if (amdsmi_get_soc_pstate_p(device_handles[d], &policy) ==
              AMDSMI_STATUS_SUCCESS &&
          policy.num_supported > 0) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "soc_pstate_policy:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current SoC P-state policy id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_soc_pstate_id) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "soc_pstate_supported:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d supported SoC P-state count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_soc_pstate_supported) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    // GPU XGMI PLPD policy events
    if (amdsmi_get_xgmi_plpd_p) {
      amdsmi_dpm_policy_t policy;
      if (amdsmi_get_xgmi_plpd_p(device_handles[d], &policy) ==
              AMDSMI_STATUS_SUCCESS &&
          policy.num_supported > 0) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "xgmi_plpd:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current XGMI PLPD policy id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_xgmi_plpd_id) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "xgmi_plpd_supported:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d supported XGMI PLPD policy count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_xgmi_plpd_supported) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    // GPU register table metrics count events (available in lib version 25+)
    if (amdsmi_lib_major >= 25 && amdsmi_get_gpu_reg_table_info_p) {
      amdsmi_reg_type_t reg_types[] = {AMDSMI_REG_XGMI, AMDSMI_REG_WAFL,
                                       AMDSMI_REG_PCIE, AMDSMI_REG_USR,
                                       AMDSMI_REG_USR1};
      const char *reg_names[] = {"XGMI", "WAFL", "PCIE", "USR", "USR1"};
      for (int rt = 0; rt < 5; ++rt) {
        amdsmi_name_value_t *reg_metrics = NULL;
        uint32_t num_metrics = 0;

        int saved_stderr = silence_stderr_begin();
        amdsmi_status_t st = amdsmi_get_gpu_reg_table_info_p(
            device_handles[d], reg_types[rt], &reg_metrics, &num_metrics);
        silence_stderr_end(saved_stderr);

        if (st == AMDSMI_STATUS_SUCCESS && num_metrics > 0) {
          if (idx >= MAX_EVENTS_PER_DEVICE * device_count) {
            if (reg_metrics)
              free(reg_metrics);
            CHECK_EVENT_IDX(idx);
          }
          snprintf(name_buf, sizeof(name_buf), "reg_%s_count:device=%d",
                   reg_names[rt], d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d number of %s register metrics", d, reg_names[rt]);
          if (add_event(&idx, name_buf, descr_buf, d, (uint32_t)reg_types[rt], 0,
                        PAPI_MODE_READ, access_amdsmi_reg_count) != PAPI_OK) {
            if (reg_metrics) free(reg_metrics);
            return PAPI_ENOMEM;
          }

          for (uint32_t i = 0; i < num_metrics; ++i) {
            if (idx >= MAX_EVENTS_PER_DEVICE * device_count) {
              if (reg_metrics)
                free(reg_metrics);
              CHECK_EVENT_IDX(idx);
            }
            char reg_metric_name[MAX_AMDSMI_NAME_LENGTH];
            sanitize_name(reg_metrics[i].name, reg_metric_name,
                          sizeof(reg_metric_name));
            snprintf(name_buf, sizeof(name_buf), "reg_%s_%s:device=%d",
                     reg_names[rt], reg_metric_name, d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d %s register %s",
                     d, reg_names[rt], reg_metrics[i].name);
            if (add_event(&idx, name_buf, descr_buf, d, (uint32_t)reg_types[rt],
                          i, PAPI_MODE_READ, access_amdsmi_reg_value) != PAPI_OK) {
              if (reg_metrics) free(reg_metrics);
              return PAPI_ENOMEM;
            }
          }
        }
        if (reg_metrics)
          free(reg_metrics);
      }
    }

    for (int si = 0; si < num_temp_sensors && si < 8; ++si) {
      // Test each sensor individually first
      int64_t sensor_test_val = 0;  // <= init
      if (!amdsmi_get_temp_metric_p ||
          amdsmi_get_temp_metric_p(device_handles[d], temp_sensors[si],
                                   AMDSMI_TEMP_CURRENT,
                                   &sensor_test_val) != AMDSMI_STATUS_SUCCESS)
        continue; // Skip this specific sensor if it doesn't work
    
      // Register metrics for this working sensor, testing each metric individually
      for (size_t mi = 0; mi < sizeof(temp_metrics) / sizeof(temp_metrics[0]); ++mi) {
        // Bounds check to prevent buffer overflow
        if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
          return PAPI_ENOSUPP; // Too many events
    
        int64_t metric_val = 0;  // <= init
        if (amdsmi_get_temp_metric_p(device_handles[d], temp_sensors[si],
                                     temp_metrics[mi], &metric_val)
            != AMDSMI_STATUS_SUCCESS)
          continue; /* skip this specific metric if not supported */
    
        snprintf(name_buf, sizeof(name_buf), "%s:device=%d:sensor=%d",
                 temp_metric_names[mi], d, (int)temp_sensors[si]);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d %s for sensor %d", d,
                 temp_metric_names[mi], (int)temp_sensors[si]);
        if (add_event(&idx, name_buf, descr_buf, d, temp_metrics[mi],
                      temp_sensors[si], PAPI_MODE_READ,
                      access_amdsmi_temp_metric) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
  }
  /* Fan metrics - test each device individually */
  for (int d = 0; d < gpu_count; ++d) {
    // Safety check for device handle
    if (!device_handles || !device_handles[d])
      continue;
    /* Register Fan RPM if available */
    int64_t dummy_rpm;
    if (amdsmi_get_gpu_fan_rpms_p &&
        amdsmi_get_gpu_fan_rpms_p(device_handles[d], 0, &dummy_rpm) ==
            AMDSMI_STATUS_SUCCESS) {
      if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
        return PAPI_ENOSUPP;
      snprintf(name_buf, sizeof(name_buf), "fan_rpms:device=%d:sensor=0", d);
      snprintf(descr_buf, sizeof(descr_buf), "Device %d fan speed in RPM", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_fan_rpms) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    /* Register Fan SPEED if available */
    int64_t dummy_speed;
    if (amdsmi_get_gpu_fan_speed_p &&
        amdsmi_get_gpu_fan_speed_p(device_handles[d], 0, &dummy_speed) ==
            AMDSMI_STATUS_SUCCESS) {
      if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
        return PAPI_ENOSUPP;
      snprintf(name_buf, sizeof(name_buf), "fan_speed:device=%d:sensor=0", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d fan speed (0-255 relative)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_fan_speed) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    /* Register Fan Max Speed - always probe directly */
    int64_t dummy_max;
    if (amdsmi_get_gpu_fan_speed_max_p &&
        amdsmi_get_gpu_fan_speed_max_p(device_handles[d], 0, &dummy_max) ==
            AMDSMI_STATUS_SUCCESS) {
      if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
        return PAPI_ENOSUPP;
      snprintf(name_buf, sizeof(name_buf), "fan_rpms_max:device=%d:sensor=0", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d fan maximum speed in RPM", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_fan_speed_max) != PAPI_OK)
        return PAPI_ENOMEM;
    }
  }
  /* VRAM memory metrics - test each device individually */
  for (int d = 0; d < gpu_count; ++d) {
    // Safety check for device handle
    if (!device_handles || !device_handles[d])
      continue;
    /* total VRAM bytes - test directly */
    uint64_t dummy_total;
    if (amdsmi_get_total_memory_p &&
        amdsmi_get_total_memory_p(device_handles[d], AMDSMI_MEM_TYPE_VRAM,
                                  &dummy_total) == AMDSMI_STATUS_SUCCESS) {
      if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
        return PAPI_ENOSUPP;
      snprintf(name_buf, sizeof(name_buf), "mem_total_VRAM:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d total VRAM memory (bytes)", d);
      if (add_event(&idx, name_buf, descr_buf, d, AMDSMI_MEM_TYPE_VRAM, 0,
                    PAPI_MODE_READ, access_amdsmi_mem_total) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    /* used VRAM bytes - test directly */
    uint64_t dummy_usage;
    if (amdsmi_get_memory_usage_p &&
        amdsmi_get_memory_usage_p(device_handles[d], AMDSMI_MEM_TYPE_VRAM,
                                  &dummy_usage) == AMDSMI_STATUS_SUCCESS) {
      if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
        return PAPI_ENOSUPP;
      snprintf(name_buf, sizeof(name_buf), "mem_usage_VRAM:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d VRAM memory usage (bytes)", d);
      if (add_event(&idx, name_buf, descr_buf, d, AMDSMI_MEM_TYPE_VRAM, 0,
                    PAPI_MODE_READ, access_amdsmi_mem_usage) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    if (amdsmi_get_gpu_vram_usage_p) {
      amdsmi_vram_usage_t vu;
      if (amdsmi_get_gpu_vram_usage_p(device_handles[d], &vu) ==
          AMDSMI_STATUS_SUCCESS) {
        if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
          return PAPI_ENOSUPP;
        snprintf(name_buf, sizeof(name_buf), "vram_total_mb:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d total VRAM (MB)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_vram_usage) != PAPI_OK)
          return PAPI_ENOMEM;
        if (idx >= MAX_EVENTS_PER_DEVICE * device_count)
          return PAPI_ENOSUPP;
        snprintf(name_buf, sizeof(name_buf), "vram_used_mb:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d used VRAM (MB)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_vram_usage) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
  }
  /* GPU power metrics: average power, power cap, and cap range */
  for (int d = 0; d < gpu_count; ++d) {
    // Safety check for device handle
    if (!device_handles || !device_handles[d])
      continue;
    // Register power average event - test directly
    amdsmi_power_info_t dummy_power;
    if (amdsmi_get_power_info_p &&
        amdsmi_get_power_info_p(device_handles[d], &dummy_power) ==
            AMDSMI_STATUS_SUCCESS) {
      // Average power consumption (W)
      snprintf(name_buf, sizeof(name_buf), "power_average:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d average power consumption (W)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_power_average) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    // Register power cap events (if available) - test directly
    amdsmi_power_cap_info_t dummy_cap_info;
    if (amdsmi_get_power_cap_info_p &&
        amdsmi_get_power_cap_info_p(device_handles[d], 0, &dummy_cap_info) ==
            AMDSMI_STATUS_SUCCESS) {
      // Current power cap limit
      snprintf(name_buf, sizeof(name_buf), "power_cap:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d current power cap (W)", d);
      if (add_event(&idx, name_buf, descr_buf, d,
                    0, 0, PAPI_MODE_READ | PAPI_MODE_WRITE,
                    access_amdsmi_power_cap) != PAPI_OK)
        return PAPI_ENOMEM;
      // Minimum allowed power cap
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "power_cap_range_min:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d minimum allowed power cap (W)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                    access_amdsmi_power_cap_range) != PAPI_OK)
        return PAPI_ENOMEM;
      // Maximum allowed power cap
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "power_cap_range_max:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d maximum allowed power cap (W)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                    access_amdsmi_power_cap_range) != PAPI_OK)
        return PAPI_ENOMEM;
      // Default power cap
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "power_cap_default:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d default power cap (W)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 3, 0, PAPI_MODE_READ,
                    access_amdsmi_power_cap_range) != PAPI_OK)
        return PAPI_ENOMEM;
      // DPM power cap
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "power_cap_dpm:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d DPM power cap (W)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 4, 0, PAPI_MODE_READ,
                    access_amdsmi_power_cap_range) != PAPI_OK)
        return PAPI_ENOMEM;
    }
  }
  /* PCIe throughput and replay counter metrics */
  uint64_t tx = 0, rx = 0, pkt = 0;
  amdsmi_status_t st_thr =
      amdsmi_get_gpu_pci_throughput_p(device_handles[0], &tx, &rx, &pkt);

  for (int d = 0; d < gpu_count; ++d) {
    if (st_thr == AMDSMI_STATUS_SUCCESS) {
      /* bytes sent per second */
      snprintf(name_buf, sizeof(name_buf), "pci_throughput_sent:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d PCIe bytes sent per second", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_pci_throughput) != PAPI_OK)
        return PAPI_ENOMEM;
      /* bytes received per second */
      snprintf(name_buf, sizeof(name_buf), "pci_throughput_received:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d PCIe bytes received per second", d);
      if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                    access_amdsmi_pci_throughput) != PAPI_OK)
        return PAPI_ENOMEM;
      /* max packet size */
      snprintf(name_buf, sizeof(name_buf),
               "pci_throughput_max_packet:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d PCIe max packet size (bytes)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                    access_amdsmi_pci_throughput) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    uint64_t replay = 0;
    if (amdsmi_get_gpu_pci_replay_counter_p(device_handles[d], &replay) ==
        AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "pci_replay_counter:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d PCIe replay (NAK) counter", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_pci_replay_counter) != PAPI_OK)
        return PAPI_ENOMEM;
    }

    if (amdsmi_get_gpu_pci_bandwidth_p) {
      amdsmi_pcie_bandwidth_t bw;
      if (amdsmi_get_gpu_pci_bandwidth_p(device_handles[d], &bw) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pci_bandwidth_supported:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d number of supported PCIe transfer rates", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_pci_bandwidth) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pci_bandwidth_current:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current PCIe transfer rate (MT/s)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_pci_bandwidth) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pci_bandwidth_lanes:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d current PCIe lane count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_pci_bandwidth) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
  }
  /* Additional GPU metrics and system information */
  /* GPU engine utilization metrics - test each device individually */
  for (int d = 0; d < gpu_count; ++d) {
    // Safety check for device handle
    if (!device_handles || !device_handles[d])
      continue;
    // Register GFX activity event - test directly
    amdsmi_engine_usage_t dummy_usage;
    if (amdsmi_get_gpu_activity_p &&
        amdsmi_get_gpu_activity_p(device_handles[d], &dummy_usage) ==
            AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "gfx_activity:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d GFX engine activity (%%)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_activity) != PAPI_OK)
        return PAPI_ENOMEM;
      snprintf(name_buf, sizeof(name_buf), "umc_activity:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d UMC engine activity (%%)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_activity) != PAPI_OK)
        return PAPI_ENOMEM;
      snprintf(name_buf, sizeof(name_buf), "mm_activity:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d MM engine activity (%%)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_activity) != PAPI_OK)
        return PAPI_ENOMEM;
    }
  }
  /* GPU utilization counters */
  if (amdsmi_get_utilization_count_p) {
    for (int d = 0; d < gpu_count; ++d) {
      amdsmi_utilization_counter_t uc;
      uint64_t ts;
      uc.type = AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
      if (amdsmi_get_utilization_count_p(device_handles[d], &uc, 1, &ts) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "util_counter_gfx:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d coarse grain GFX activity counter", d);
        if (add_event(&idx, name_buf, descr_buf, d,
                      AMDSMI_COARSE_GRAIN_GFX_ACTIVITY, 0, PAPI_MODE_READ,
                      access_amdsmi_utilization_count) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uc.type = AMDSMI_COARSE_GRAIN_MEM_ACTIVITY;
      if (amdsmi_get_utilization_count_p(device_handles[d], &uc, 1, &ts) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "util_counter_mem:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d coarse grain memory activity counter", d);
        if (add_event(&idx, name_buf, descr_buf, d,
                      AMDSMI_COARSE_GRAIN_MEM_ACTIVITY, 0, PAPI_MODE_READ,
                      access_amdsmi_utilization_count) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uc.type = AMDSMI_COARSE_DECODER_ACTIVITY;
      if (amdsmi_get_utilization_count_p(device_handles[d], &uc, 1, &ts) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "util_counter_dec:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d coarse grain decoder activity counter", d);
        if (add_event(&idx, name_buf, descr_buf, d,
                      AMDSMI_COARSE_DECODER_ACTIVITY, 0, PAPI_MODE_READ,
                      access_amdsmi_utilization_count) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
  }
  /* GPU clock frequency levels for multiple clock domains */
  for (int d = 0; d < gpu_count; ++d) {
    amdsmi_clk_type_t clk_types[] = {AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_TYPE_DF,
                                     AMDSMI_CLK_TYPE_DCEF};
    const char *clk_names[] = {"sys", "df", "dcef"};
    for (int t = 0; t < 3; ++t) {
      amdsmi_frequencies_t f;
      if (amdsmi_get_clk_freq_p(device_handles[d], clk_types[t], &f) !=
              AMDSMI_STATUS_SUCCESS ||
          f.num_supported == 0)
        continue;
      // Number of supported frequencies for this clock domain
      snprintf(name_buf, sizeof(name_buf), "clk_freq_%s_count:device=%d",
               clk_names[t], d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d number of supported %s clock frequencies", d,
               clk_names[t]);
      if (add_event(&idx, name_buf, descr_buf, d, t, 0, PAPI_MODE_READ,
                    access_amdsmi_clk_freq) != PAPI_OK)
        return PAPI_ENOMEM;
      // Current clock frequency for this domain
      snprintf(name_buf, sizeof(name_buf), "clk_freq_%s_current:device=%d",
               clk_names[t], d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d current %s clock frequency (MHz)", d, clk_names[t]);
      if (add_event(&idx, name_buf, descr_buf, d, t, 1, PAPI_MODE_READ,
                    access_amdsmi_clk_freq) != PAPI_OK)
        return PAPI_ENOMEM;
      // Supported frequency levels for this domain
      for (uint32_t fi = 0; fi < f.num_supported; ++fi) {
        snprintf(name_buf, sizeof(name_buf), "clk_freq_%s_level_%u:device=%d",
                 clk_names[t], fi, d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d supported %s clock frequency level %u (MHz)", d,
                 clk_names[t], fi);
        if (add_event(&idx, name_buf, descr_buf, d, t, fi + 2, PAPI_MODE_READ,
                      access_amdsmi_clk_freq) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
  }
  if (amdsmi_get_clock_info_p) {
    for (int d = 0; d < gpu_count; ++d) {
      amdsmi_clk_type_t clk_types[] = {AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_TYPE_MEM};
      const char *clk_names[] = {"sys", "mem"};
      const char *field_names[] = {"current", "min", "max", "locked",
                                   "deep_sleep"};
      const char *field_descr[] = {
          "current frequency (MHz)",     "minimum frequency (MHz)",
          "maximum frequency (MHz)",     "lock state (bool)",
          "deep sleep frequency (MHz)"};
      for (int t = 0; t < 2; ++t) {
        amdsmi_clk_info_t info;
        if (amdsmi_get_clock_info_p(device_handles[d], clk_types[t], &info) !=
            AMDSMI_STATUS_SUCCESS)
          continue;
        for (int f = 0; f < 5; ++f) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "clk_%s_%s:device=%d",
                   clk_names[t], field_names[f], d);
          snprintf(descr_buf, sizeof(descr_buf), "Device %d %s %s", d,
                   clk_names[t], field_descr[f]);
          if (add_event(&idx, name_buf, descr_buf, d, t, f, PAPI_MODE_READ,
                        access_amdsmi_clock_info) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
  }
  /* GPU identification and topology metrics */
  for (int d = 0; d < gpu_count; ++d) {
    uint16_t id16;
    uint64_t id64;
    int32_t numa;
    // GPU ID
    if (amdsmi_get_gpu_id_p(device_handles[d], &id16) ==
        AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "gpu_id:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d GPU identifier (Device ID)", d);
      if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_info) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    // GPU Revision
    if (amdsmi_get_gpu_revision_p(device_handles[d], &id16) ==
        AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "gpu_revision:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf), "Device %d GPU revision ID", d);
      if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_info) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    // GPU Subsystem ID
    if (amdsmi_get_gpu_subsystem_id_p(device_handles[d], &id16) ==
        AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "gpu_subsystem_id:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf), "Device %d GPU subsystem ID", d);
      if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_info) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    // GPU BDF ID
    if (amdsmi_get_gpu_bdf_id_p(device_handles[d], &id64) ==
        AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "gpu_bdfid:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d GPU PCI BDF identifier", d);
      if (add_event(&idx, name_buf, descr_buf, d, 3, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_info) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    // GPU device BDF components
    if (amdsmi_get_gpu_device_bdf_p) {
      amdsmi_bdf_t bdf;
      if (amdsmi_get_gpu_device_bdf_p(device_handles[d], &bdf) ==
          AMDSMI_STATUS_SUCCESS) {
        const char *bdf_names[] = {"gpu_bdf_domain", "gpu_bdf_bus",
                                   "gpu_bdf_device", "gpu_bdf_function"};
        const char *bdf_descr[] = {
            "GPU PCI domain number", "GPU PCI bus number",
            "GPU PCI device number", "GPU PCI function number"};
        for (uint32_t v = 0; v < 4; ++v) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "%s:device=%d",
                   bdf_names[v], d);
          snprintf(descr_buf, sizeof(descr_buf), "Device %d %s", d,
                   bdf_descr[v]);
          if (add_event(&idx, name_buf, descr_buf, d, v, 0, PAPI_MODE_READ,
                        access_amdsmi_device_bdf) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    if (amdsmi_get_xgmi_info_p) {
      amdsmi_xgmi_info_t xi;
      if (amdsmi_get_xgmi_info_p(device_handles[d], &xi) == AMDSMI_STATUS_SUCCESS) {
        const char *xinames[] = {"xgmi_lanes", "xgmi_hive_id", "xgmi_node_id",
                                 "xgmi_index"};
        const char *xidescr[] = {"Device %d XGMI lane count",
                                 "Device %d XGMI hive identifier",
                                 "Device %d XGMI node identifier",
                                 "Device %d XGMI link index"};
        for (uint32_t v = 0; v < 4; ++v) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "%s:device=%d", xinames[v], d);
          snprintf(descr_buf, sizeof(descr_buf), xidescr[v], d);
          if (add_event(&idx, name_buf, descr_buf, d, v, 0, PAPI_MODE_READ,
                        access_amdsmi_xgmi_info) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    if (amdsmi_get_gpu_kfd_info_p) {
      amdsmi_kfd_info_t kinfo;
      if (amdsmi_get_gpu_kfd_info_p(device_handles[d], &kinfo) ==
          AMDSMI_STATUS_SUCCESS) {
        const char *knames[] = {"kfd_id", "kfd_node_id",
                                 "kfd_current_partition_id"};
        const char *kdescr[] = {"Device %d KFD identifier",
                                "Device %d KFD node id",
                                "Device %d KFD current partition id"};
        for (uint32_t v = 0; v < 3; ++v) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "%s:device=%d", knames[v], d);
          snprintf(descr_buf, sizeof(descr_buf), kdescr[v], d);
          if (add_event(&idx, name_buf, descr_buf, d, v, 0, PAPI_MODE_READ,
                        access_amdsmi_kfd_info) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    // NUMA node via topology API
    if (amdsmi_topo_get_numa_node_number_p) {
      uint32_t node;
      if (amdsmi_topo_get_numa_node_number_p(device_handles[d], &node) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "topo_numa_node:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d NUMA node number", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_topo_numa) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    // GPU Virtualization Mode
#if AMDSMI_LIB_VERSION_MAJOR >= 25
    amdsmi_virtualization_mode_t vmode;
    if (amdsmi_lib_major >= 25 && amdsmi_get_gpu_virtualization_mode_p &&
        amdsmi_get_gpu_virtualization_mode_p(device_handles[d], &vmode) ==
            AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "gpu_virtualization_mode:device=%d",
               d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d GPU virtualization mode", d);
      if (add_event(&idx, name_buf, descr_buf, d, 4, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_info) != PAPI_OK)
        return PAPI_ENOMEM;
    }
#endif
    // GPU NUMA Node
    if (amdsmi_get_gpu_topo_numa_affinity_p(device_handles[d], &numa) ==
        AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "numa_node:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf), "Device %d NUMA node", d);
      if (add_event(&idx, name_buf, descr_buf, d, 5, 0, PAPI_MODE_READ,
                    access_amdsmi_gpu_info) != PAPI_OK)
        return PAPI_ENOMEM;
    }

    if (amdsmi_get_gpu_process_list_p) {
      amdsmi_proc_info_t plist[2];
      uint32_t maxp = 2;
      if (amdsmi_get_gpu_process_list_p(device_handles[d], &maxp, plist) ==
          AMDSMI_STATUS_SUCCESS) {
        const char *pmetric_names[] = {"pid", "mem",         "eng_gfx",
                                       "eng_enc", "gtt_mem", "cpu_mem",
                                       "vram_mem", "cu_occupancy"};
        const char *pmetric_descr[] = {
            "PID",                 "memory usage (bytes)",
            "GFX engine time (ns)", "ENC engine time (ns)",
            "GTT memory (bytes)",  "CPU memory (bytes)",
            "VRAM memory (bytes)", "Compute units utilized"};
        for (uint32_t p = 0; p < 2; ++p) {
          for (uint32_t v = 0; v < 8; ++v) {
            CHECK_EVENT_IDX(idx);
            snprintf(name_buf, sizeof(name_buf),
                     "process_%s:device=%d:proc=%u", pmetric_names[v], d, p);
            snprintf(descr_buf, sizeof(descr_buf),
                     "Device %d process %u %s", d, p, pmetric_descr[v]);
            if (add_event(&idx, name_buf, descr_buf, d, v, p, PAPI_MODE_READ,
                          access_amdsmi_process_info) != PAPI_OK)
              return PAPI_ENOMEM;
          }
        }
      }
    }

    if (amdsmi_get_gpu_process_isolation_p) {
      uint32_t pis = 0;
      if (amdsmi_get_gpu_process_isolation_p(device_handles[d], &pis) ==
          AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "process_isolation:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d process isolation status", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_process_isolation) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_gpu_xcd_counter_p) {
      uint16_t xcd = 0;
      if (amdsmi_get_gpu_xcd_counter_p(device_handles[d], &xcd) ==
          AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "xcd_counter:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d XCD counter", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_xcd_counter) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_minmax_bandwidth_between_processors_p) {
      for (int r = 0; r < gpu_count; ++r) {
        if (r == d)
          continue;
        uint64_t min_bw = 0, max_bw = 0;
        if (amdsmi_get_minmax_bandwidth_between_processors_p(
                device_handles[d], device_handles[r], &min_bw, &max_bw) ==
            AMDSMI_STATUS_SUCCESS) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "xgmi_min_bandwidth:src=%d:dst=%d", d, r);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Min XGMI bandwidth from device %d to %d (MB/s)", d, r);
          if (add_event(&idx, name_buf, descr_buf, d, 0, r, PAPI_MODE_READ,
                        access_amdsmi_xgmi_bandwidth) != PAPI_OK)
            return PAPI_ENOMEM;
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "xgmi_max_bandwidth:src=%d:dst=%d", d, r);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Max XGMI bandwidth from device %d to %d (MB/s)", d, r);
          if (add_event(&idx, name_buf, descr_buf, d, 1, r, PAPI_MODE_READ,
                        access_amdsmi_xgmi_bandwidth) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }

    if (amdsmi_gpu_counter_group_supported_p &&
        amdsmi_get_gpu_available_counters_p && amdsmi_gpu_create_counter_p &&
        amdsmi_gpu_control_counter_p && amdsmi_gpu_read_counter_p &&
        amdsmi_gpu_destroy_counter_p) {
      if (amdsmi_gpu_counter_group_supported_p(
              device_handles[d], AMDSMI_EVNT_GRP_XGMI) ==
          AMDSMI_STATUS_SUCCESS) {
        uint32_t avail = 0;
        if (amdsmi_get_gpu_available_counters_p(
                device_handles[d], AMDSMI_EVNT_GRP_XGMI, &avail) ==
                AMDSMI_STATUS_SUCCESS &&
            avail > 0) {
          static const struct {
            const char *suffix;
            amdsmi_event_type_t type[2];
          } xgmi_desc[] = {
              {"nop_tx", {AMDSMI_EVNT_XGMI_0_NOP_TX,
                          AMDSMI_EVNT_XGMI_1_NOP_TX}},
              {"request_tx",
               {AMDSMI_EVNT_XGMI_0_REQUEST_TX,
                AMDSMI_EVNT_XGMI_1_REQUEST_TX}},
              {"response_tx",
               {AMDSMI_EVNT_XGMI_0_RESPONSE_TX,
                AMDSMI_EVNT_XGMI_1_RESPONSE_TX}},
              {"beats_tx", {AMDSMI_EVNT_XGMI_0_BEATS_TX,
                            AMDSMI_EVNT_XGMI_1_BEATS_TX}},
          };
          for (int link = 0; link < 2; ++link) {
            for (size_t m = 0; m < sizeof(xgmi_desc) / sizeof(xgmi_desc[0]);
                 ++m) {
              CHECK_EVENT_IDX(idx);
              snprintf(name_buf, sizeof(name_buf),
                       "xgmi_%s:device=%d:link=%d", xgmi_desc[m].suffix, d, link);
              snprintf(descr_buf, sizeof(descr_buf),
                       "Device %d XGMI %s on link %d", d, xgmi_desc[m].suffix,
                       link);
              if (add_counter_event(&idx, name_buf, descr_buf, d,
                                    xgmi_desc[m].type[link], link) != PAPI_OK)
                return PAPI_ENOMEM;
            }
          }
        }
      }
    }

    if (amdsmi_get_fw_info_p) {
      amdsmi_fw_info_t finfo;
      if (amdsmi_get_fw_info_p(device_handles[d], &finfo) ==
          AMDSMI_STATUS_SUCCESS) {
        uint8_t n = finfo.num_fw_info;
        if (n > AMDSMI_FW_ID__MAX)
          n = AMDSMI_FW_ID__MAX;
        for (uint8_t f = 0; f < n; ++f) {
          CHECK_EVENT_IDX(idx);
          uint32_t fid = finfo.fw_info_list[f].fw_id;
          snprintf(name_buf, sizeof(name_buf), "fw_version_id%u:device=%d", fid,
                   d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d firmware id %u version", d, fid);
          if (add_event(&idx, name_buf, descr_buf, d, fid, 0, PAPI_MODE_READ,
                        access_amdsmi_fw_version) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }

    if (amdsmi_get_gpu_board_info_p) {
      amdsmi_board_info_t binfo;
      memset(&binfo, 0, sizeof(binfo));
      if (amdsmi_get_gpu_board_info_p(device_handles[d], &binfo) ==
          AMDSMI_STATUS_SUCCESS) {
        sanitize_description_text(binfo.product_serial);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "board_serial_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d board serial number hash of '%s'", d,
                 display_or_empty(binfo.product_serial));
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_board_serial_hash) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_gpu_vram_info_p) {
#if AMDSMI_LIB_VERSION_MAJOR >= 25
      if (amdsmi_lib_major >= 25) {
        amdsmi_vram_info_t vinfo;
        if (amdsmi_get_gpu_vram_info_p(device_handles[d], &vinfo) ==
            AMDSMI_STATUS_SUCCESS) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "vram_max_bandwidth:device=%d", d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d VRAM max bandwidth (GB/s)", d);
          if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                        access_amdsmi_vram_max_bandwidth) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
#endif
    }

    if (amdsmi_get_gpu_memory_reserved_pages_p) {
      uint32_t nump = 0;
      if (amdsmi_get_gpu_memory_reserved_pages_p(device_handles[d], &nump,
                                                 NULL) == AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "memory_reserved_pages:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d reserved memory pages", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_memory_reserved_pages) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_gpu_bad_page_info_p) {
      uint32_t nump = 0;
      if (amdsmi_get_gpu_bad_page_info_p(device_handles[d], &nump, NULL) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "bad_page_count:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d retired page count",
                 d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_bad_page_count) != PAPI_OK)
          return PAPI_ENOMEM;
        for (uint32_t p = 0; p < nump; ++p) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "bad_page_address:device=%d:page=%u", d, p);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d retired page %u address", d, p);
          if (add_event(&idx, name_buf, descr_buf, d, 0, p, PAPI_MODE_READ,
                        access_amdsmi_bad_page_record) != PAPI_OK)
            return PAPI_ENOMEM;
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "bad_page_size:device=%d:page=%u", d, p);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d retired page %u size", d, p);
          if (add_event(&idx, name_buf, descr_buf, d, 1, p, PAPI_MODE_READ,
                        access_amdsmi_bad_page_record) != PAPI_OK)
            return PAPI_ENOMEM;
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "bad_page_status:device=%d:page=%u", d, p);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d retired page %u status", d, p);
          if (add_event(&idx, name_buf, descr_buf, d, 2, p, PAPI_MODE_READ,
                        access_amdsmi_bad_page_record) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }

    if (amdsmi_get_gpu_bad_page_threshold_p) {
      uint32_t thr = 0;
      if (amdsmi_get_gpu_bad_page_threshold_p(device_handles[d], &thr) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "bad_page_threshold:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d bad page threshold",
                 d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_bad_page_threshold) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_power_info_v2_p) {
      /* Probe for available power sensors. */
      for (uint32_t s = 0; s < 2; ++s) {
        amdsmi_power_info_t pinfo;
        if (amdsmi_get_power_info_v2_p(device_handles[d], s, &pinfo) !=
            AMDSMI_STATUS_SUCCESS)
          break;

        /* Register current socket power in Watts */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "power_sensor_current_watts:device=%d:sensor=%u", d, s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d power sensor %u current socket power (W)", d, s);
        if (add_event(&idx, name_buf, descr_buf, d, 0, s, PAPI_MODE_READ,
                      access_amdsmi_power_sensor) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register average socket power in Watts */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "power_sensor_average_watts:device=%d:sensor=%u", d, s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d power sensor %u average socket power (W)", d, s);
        if (add_event(&idx, name_buf, descr_buf, d, 1, s, PAPI_MODE_READ,
                      access_amdsmi_power_sensor) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register socket power in microwatts */
#if AMDSMI_LIB_VERSION_MAJOR >= 25
        if (amdsmi_lib_major >= 25) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "power_sensor_socket_microwatts:device=%d:sensor=%u", d, s);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d power sensor %u socket power (uW)", d, s);
          if (add_event(&idx, name_buf, descr_buf, d, 2, s, PAPI_MODE_READ,
                        access_amdsmi_power_sensor) != PAPI_OK)
            return PAPI_ENOMEM;
        }
#endif

        /* Register GFX voltage */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "power_sensor_gfx_voltage_mv:device=%d:sensor=%u", d, s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d power sensor %u GFX voltage (mV)", d, s);
        if (add_event(&idx, name_buf, descr_buf, d, 3, s, PAPI_MODE_READ,
                      access_amdsmi_power_sensor) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register SOC voltage */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "power_sensor_soc_voltage_mv:device=%d:sensor=%u", d, s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d power sensor %u SOC voltage (mV)", d, s);
        if (add_event(&idx, name_buf, descr_buf, d, 4, s, PAPI_MODE_READ,
                      access_amdsmi_power_sensor) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register MEM voltage */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "power_sensor_mem_voltage_mv:device=%d:sensor=%u", d, s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d power sensor %u MEM voltage (mV)", d, s);
        if (add_event(&idx, name_buf, descr_buf, d, 5, s, PAPI_MODE_READ,
                      access_amdsmi_power_sensor) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register power limit */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "power_sensor_limit_watts:device=%d:sensor=%u", d, s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d power sensor %u power limit (W)", d, s);
        if (add_event(&idx, name_buf, descr_buf, d, 6, s, PAPI_MODE_READ,
                      access_amdsmi_power_sensor) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_gpu_metrics_header_info_p) {
      amd_metrics_table_header_t hdr = {0};   // <= zero-init
    
      // If the API defines a size/version field, set it before the call:
      // hdr.metrics_header_size = sizeof(hdr);   // uncomment if such a field exists
    
      if (amdsmi_get_gpu_metrics_header_info_p(device_handles[d], &hdr)
          == AMDSMI_STATUS_SUCCESS) {
        const char *hnames[] = {"metrics_header_size",
                                "metrics_header_format_rev",
                                "metrics_header_content_rev"};
        const char *hdescr[] = {"Device %d metrics header structure size",
                                "Device %d metrics header format revision",
                                "Device %d metrics header content revision"};
        for (uint32_t v = 0; v < 3; ++v) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "%s:device=%d", hnames[v], d);
          snprintf(descr_buf, sizeof(descr_buf), hdescr[v], d);
          if (add_event(&idx, name_buf, descr_buf, d, v, 0, PAPI_MODE_READ,
                        access_amdsmi_metrics_header_info) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }

    if (amdsmi_get_gpu_metrics_info_p) {
      amdsmi_gpu_metrics_t metrics;
      if (amdsmi_get_gpu_metrics_info_p(device_handles[d], &metrics) ==
          AMDSMI_STATUS_SUCCESS) {
        /* Register throttle status */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "gpu_throttle_status:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d throttle status", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register independent throttle status */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "gpu_indep_throttle_status:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d independent throttle status", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register PCIe link width */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_link_width:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe link width (lanes)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register PCIe link speed */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_link_speed:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe link speed (0.1 GT/s)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 3, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        /* Register PCIe bandwidth and replay counters */
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_bandwidth_acc:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe accumulated bandwidth (GB/s)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 4, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "pcie_bandwidth_inst:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe instantaneous bandwidth (GB/s)", d);
        if (add_event(&idx, name_buf, descr_buf, d, 5, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_l0_to_recov_count_acc:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe L0->recovery count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 6, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_replay_count_acc:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d PCIe replay count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 7, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_replay_rover_count_acc:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe replay rollover count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 8, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_nak_sent_count_acc:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d PCIe NAK sent count",
                 d);
        if (add_event(&idx, name_buf, descr_buf, d, 9, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "pcie_nak_rcvd_count_acc:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d PCIe NAK received count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 10, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_metrics) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_init_gpu_event_notification_p &&
        amdsmi_set_gpu_event_notification_mask_p &&
        amdsmi_get_gpu_event_notification_p &&
        amdsmi_stop_gpu_event_notification_p) {
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "thermal_throttle_events:device=%d", d);
      snprintf(descr_buf, sizeof(descr_buf),
               "Device %d thermal throttle event notifications", d);
      if (add_event(&idx, name_buf, descr_buf, d, AMDSMI_EVT_NOTIF_THERMAL_THROTTLE,
                    0, PAPI_MODE_READ, access_amdsmi_event_notification) != PAPI_OK)
        return PAPI_ENOMEM;
    }
  }
  /* Energy consumption counter */
  for (int d = 0; d < gpu_count; ++d) {
    uint64_t energy = 0;
    float resolution = 0.0;
    uint64_t timestamp = 0;
    if (amdsmi_get_energy_count_p(device_handles[d], &energy, &resolution,
                                  &timestamp) != AMDSMI_STATUS_SUCCESS)
      continue;
    snprintf(name_buf, sizeof(name_buf), "energy_consumed:device=%d", d);
    snprintf(descr_buf, sizeof(descr_buf),
             "Device %d energy consumed (microJoules)", d);
    if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                  access_amdsmi_energy_count) != PAPI_OK)
      return PAPI_ENOMEM;

    snprintf(name_buf, sizeof(name_buf), "energy_resolution:device=%d", d);
    snprintf(descr_buf, sizeof(descr_buf),
             "Device %d energy counter resolution (microJoules)", d);
    if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                  access_amdsmi_energy_count) != PAPI_OK)
      return PAPI_ENOMEM;

    snprintf(name_buf, sizeof(name_buf), "energy_timestamp:device=%d", d);
    snprintf(descr_buf, sizeof(descr_buf),
             "Device %d energy counter timestamp (ns)", d);
    if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                  access_amdsmi_energy_count) != PAPI_OK)
      return PAPI_ENOMEM;
  }
  /* GPU power profile information */
  for (int d = 0; d < gpu_count; ++d) {
    amdsmi_power_profile_status_t profile_status;
    if (amdsmi_get_gpu_power_profile_presets_p(
            device_handles[d], 0, &profile_status) != AMDSMI_STATUS_SUCCESS)
      continue;
    snprintf(name_buf, sizeof(name_buf), "power_profiles_count:device=%d", d);
    snprintf(descr_buf, sizeof(descr_buf),
             "Device %d number of supported power profiles", d);
    if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                  access_amdsmi_power_profile_status) != PAPI_OK)
      return PAPI_ENOMEM;
    snprintf(name_buf, sizeof(name_buf), "power_profile_current:device=%d", d);
    snprintf(descr_buf, sizeof(descr_buf),
             "Device %d current power profile mask", d);
    if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                  access_amdsmi_power_profile_status) != PAPI_OK)
      return PAPI_ENOMEM;
  }
  /* GPU violation status metrics */
  if (amdsmi_get_violation_status_p) {
    for (int d = 0; d < gpu_count; ++d) {
      amdsmi_violation_status_t vinfo;
      if (amdsmi_get_violation_status_p(device_handles[d], &vinfo) !=
          AMDSMI_STATUS_SUCCESS)
        continue;
      const char *names[] = {
          "ppt_pwr_violation_acc",    "socket_thrm_violation_acc",
          "vr_thrm_violation_acc",    "ppt_pwr_violation_pct",
          "socket_thrm_violation_pct", "vr_thrm_violation_pct",
          "ppt_pwr_violation_active",  "socket_thrm_violation_active",
          "vr_thrm_violation_active"};
      const char *descr[] = {
          "Package power tracking violation count",
          "Socket thermal violation count",
          "Voltage regulator thermal violation count",
          "Package power tracking violation percentage",
          "Socket thermal violation percentage",
          "Voltage regulator thermal violation percentage",
          "Package power tracking violation active flag",
          "Socket thermal violation active flag",
          "Voltage regulator thermal violation active flag"};
      for (int v = 0; v < 9; ++v) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "%s:device=%d", names[v], d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d %s", d, descr[v]);
        if (add_event(&idx, name_buf, descr_buf, d, v, 0, PAPI_MODE_READ,
                      access_amdsmi_violation_status) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
  }
#ifndef AMDSMI_DISABLE_ESMI
  /* CPU metrics events */
  if (cpu_count > 0) {
    // CPU socket-level events
    for (int s = 0; s < cpu_count; ++s) {
      int dev = gpu_count + s;
      uint32_t pwr;
      if (amdsmi_get_cpu_socket_power_p(device_handles[dev], &pwr) ==
          AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "power:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf), "Socket %d power (W)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_socket_power) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uint64_t sock_energy;
      if (amdsmi_get_cpu_socket_energy_p(device_handles[dev], &sock_energy) ==
          AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "energy:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d energy consumed (uJ)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_socket_energy) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uint16_t fmax, fmin;
      if (amdsmi_get_cpu_socket_freq_range_p(device_handles[dev], &fmax,
                                             &fmin) == AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "freq_max:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d maximum frequency (MHz)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_socket_freq_range) != PAPI_OK)
          return PAPI_ENOMEM;
        snprintf(name_buf, sizeof(name_buf), "freq_min:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d minimum frequency (MHz)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_socket_freq_range) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uint32_t cap;
      amdsmi_status_t st_cap =
          amdsmi_get_cpu_socket_power_cap_p(device_handles[dev], &cap);
      uint32_t cap_max;
      amdsmi_status_t st_capmax =
          amdsmi_get_cpu_socket_power_cap_max_p(device_handles[dev], &cap_max);
      if (st_cap == AMDSMI_STATUS_SUCCESS ||
          st_capmax == AMDSMI_STATUS_SUCCESS) {
        if (st_cap == AMDSMI_STATUS_SUCCESS) {
          snprintf(name_buf, sizeof(name_buf), "power_cap:socket=%d", s);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Socket %d current power cap (W)", s);
          if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                        access_amdsmi_cpu_power_cap) != PAPI_OK)
            return PAPI_ENOMEM;
        }
        if (st_capmax == AMDSMI_STATUS_SUCCESS) {
          snprintf(name_buf, sizeof(name_buf), "power_cap_max:socket=%d", s);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Socket %d max power cap (W)", s);
          if (add_event(&idx, name_buf, descr_buf, dev, 1, 0, PAPI_MODE_READ,
                        access_amdsmi_cpu_power_cap) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
      uint16_t freq;
      char *src_type = NULL;
      if (amdsmi_get_cpu_socket_current_active_freq_limit_p(
              device_handles[dev], &freq, &src_type) == AMDSMI_STATUS_SUCCESS) {
        if (src_type)
          free(src_type);
        snprintf(name_buf, sizeof(name_buf), "freq_limit:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d current frequency limit (MHz)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_socket_freq_limit) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uint32_t cclk;
      if (amdsmi_get_cpu_cclk_limit_p &&
          amdsmi_get_cpu_cclk_limit_p(device_handles[dev], &cclk) ==
              AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "cclk_limit:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d core clock limit (MHz)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_cclk_limit) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uint32_t fclk, mclk;
      if (amdsmi_get_cpu_fclk_mclk_p &&
          amdsmi_get_cpu_fclk_mclk_p(device_handles[dev], &fclk, &mclk) ==
              AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "fclk:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d fclk (MHz)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_fclk_mclk) != PAPI_OK)
          return PAPI_ENOMEM;
        snprintf(name_buf, sizeof(name_buf), "mclk:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d mclk (MHz)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_fclk_mclk) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      amdsmi_ddr_bw_metrics_t ddr_bw;
      if (amdsmi_get_cpu_ddr_bw_p &&
          amdsmi_get_cpu_ddr_bw_p(device_handles[dev], &ddr_bw) ==
              AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "ddr_bw_max:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d DDR max bandwidth (GB/s)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_ddr_bw) != PAPI_OK)
          return PAPI_ENOMEM;
        snprintf(name_buf, sizeof(name_buf), "ddr_bw_utilized:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d DDR utilized bandwidth (GB/s)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_ddr_bw) != PAPI_OK)
          return PAPI_ENOMEM;
        snprintf(name_buf, sizeof(name_buf),
                 "ddr_bw_utilized_pct:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d DDR bandwidth utilization (pct)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_ddr_bw) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      amdsmi_hsmp_driver_version_t dver;
      if (amdsmi_get_cpu_hsmp_driver_version_p &&
          amdsmi_get_cpu_hsmp_driver_version_p(device_handles[dev], &dver) ==
              AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf),
                 "hsmp_driver_major:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d HSMP driver major version", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_hsmp_driver_version) != PAPI_OK)
          return PAPI_ENOMEM;
        snprintf(name_buf, sizeof(name_buf),
                 "hsmp_driver_minor:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d HSMP driver minor version", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_hsmp_driver_version) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uint32_t proto;
      if (amdsmi_get_cpu_hsmp_proto_ver_p &&
          amdsmi_get_cpu_hsmp_proto_ver_p(device_handles[dev], &proto) ==
              AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf),
                 "hsmp_proto_ver:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d HSMP protocol version", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_hsmp_proto_ver) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uint32_t prochot;
      if (amdsmi_get_cpu_prochot_status_p &&
          amdsmi_get_cpu_prochot_status_p(device_handles[dev], &prochot) ==
              AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf),
                 "prochot_status:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d PROCHOT status", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_prochot_status) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      uint32_t svi_power;
      if (amdsmi_get_cpu_pwr_svi_telemetry_all_rails_p &&
          amdsmi_get_cpu_pwr_svi_telemetry_all_rails_p(device_handles[dev],
                                                       &svi_power) ==
              AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "svi_power:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d SVI power (all rails, W)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_cpu_svi_power) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      amdsmi_smu_fw_version_t fw;
      if (amdsmi_get_cpu_smu_fw_version_p(device_handles[dev], &fw) ==
          AMDSMI_STATUS_SUCCESS) {
        snprintf(name_buf, sizeof(name_buf), "smu_fw_version:socket=%d", s);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Socket %d SMU firmware version (encoded)", s);
        if (add_event(&idx, name_buf, descr_buf, dev, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_smu_fw_version) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      if (amdsmi_get_cpu_current_io_bandwidth_p) {
        const char *links[] = {"P0", "P1", "P2", "P3", "P4"};
        const char *bwnames[] = {"agg", "read", "write"};
        amdsmi_io_bw_encoding_t bw_types[] = {AGG_BW0, RD_BW0, WR_BW0};
        for (int l = 0; l < 5; ++l) {
          for (int t = 0; t < 3; ++t) {
            amdsmi_link_id_bw_type_t link = {bw_types[t], (char *)links[l]};
            uint32_t bw = 0;
            if (amdsmi_get_cpu_current_io_bandwidth_p(device_handles[dev], link,
                                                      &bw) !=
                AMDSMI_STATUS_SUCCESS)
              continue;
            CHECK_EVENT_IDX(idx);
            snprintf(name_buf, sizeof(name_buf),
                     "io_bw_%s_%s:socket=%d", links[l], bwnames[t], s);
            snprintf(descr_buf, sizeof(descr_buf),
                     "Socket %d IO link %s %s bandwidth (MB/s)", s,
                     links[l], bwnames[t]);
            if (add_event(&idx, name_buf, descr_buf, dev, l, t, PAPI_MODE_READ,
                          access_amdsmi_cpu_io_bw) != PAPI_OK)
              return PAPI_ENOMEM;
          }
        }
      }
      if (amdsmi_get_cpu_current_xgmi_bw_p) {
        const char *links[] = {"G0", "G1", "G2", "G3",
                               "G4", "G5", "G6", "G7"};
        const char *bwnames[] = {"agg", "read", "write"};
        amdsmi_io_bw_encoding_t bw_types[] = {AGG_BW0, RD_BW0, WR_BW0};
        for (int l = 0; l < 8; ++l) {
          for (int t = 0; t < 3; ++t) {
            amdsmi_link_id_bw_type_t link = {bw_types[t], (char *)links[l]};
            uint32_t bw = 0;
            if (amdsmi_get_cpu_current_xgmi_bw_p(device_handles[dev], link,
                                                 &bw) !=
                AMDSMI_STATUS_SUCCESS)
              continue;
            CHECK_EVENT_IDX(idx);
            snprintf(name_buf, sizeof(name_buf),
                     "xgmi_bw_%s_%s:socket=%d", links[l], bwnames[t], s);
            snprintf(descr_buf, sizeof(descr_buf),
                     "Socket %d XGMI link %s %s bandwidth (MB/s)", s,
                     links[l], bwnames[t]);
            if (add_event(&idx, name_buf, descr_buf, dev, l, t, PAPI_MODE_READ,
                          access_amdsmi_cpu_xgmi_bw) != PAPI_OK)
              return PAPI_ENOMEM;
          }
        }
      }
    }
    // CPU core-level events
    for (int s = 0; s < cpu_count; ++s) {
      int dev = gpu_count + s;
      for (uint32_t c = 0; c < cores_per_socket[s]; ++c) {
        uint64_t energy;
        if (amdsmi_get_cpu_core_energy_p(cpu_core_handles[s][c], &energy) ==
            AMDSMI_STATUS_SUCCESS) {
          snprintf(name_buf, sizeof(name_buf), "energy:socket=%d:core=%d", s, c);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Socket %d Core %d energy (uJ)", s, c);
          if (add_event(&idx, name_buf, descr_buf, dev, 0, c, PAPI_MODE_READ,
                        access_amdsmi_cpu_core_energy) != PAPI_OK)
            return PAPI_ENOMEM;
        }
        uint32_t freq;
        if (amdsmi_get_cpu_core_current_freq_limit_p(
                cpu_core_handles[s][c], &freq) == AMDSMI_STATUS_SUCCESS) {
          snprintf(name_buf, sizeof(name_buf), "freq_limit:socket=%d:core=%d",
                   s, c);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Socket %d Core %d frequency limit (MHz)", s, c);
          if (add_event(&idx, name_buf, descr_buf, dev, 0, c, PAPI_MODE_READ,
                        access_amdsmi_cpu_core_freq_limit) != PAPI_OK)
            return PAPI_ENOMEM;
        }
        uint32_t boost;
        if (amdsmi_get_cpu_core_boostlimit_p(cpu_core_handles[s][c], &boost) ==
            AMDSMI_STATUS_SUCCESS) {
          snprintf(name_buf, sizeof(name_buf), "boostlimit:socket=%d:core=%d",
                   s, c);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Socket %d Core %d boost limit (MHz)", s, c);
          if (add_event(&idx, name_buf, descr_buf, dev, 0, c, PAPI_MODE_READ,
                        access_amdsmi_cpu_core_boostlimit) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    // CPU DIMM events
    for (int s = 0; s < cpu_count; ++s) {
      int dev = gpu_count + s;
      for (uint8_t dimm = 0; dimm < 16; ++dimm) {
        amdsmi_dimm_thermal_t dimm_temp;
        amdsmi_dimm_power_t dimm_pow;
        amdsmi_temp_range_refresh_rate_t range_info;
        amdsmi_status_t st_temp = amdsmi_get_cpu_dimm_thermal_sensor_p(
            device_handles[dev], dimm, &dimm_temp);
        amdsmi_status_t st_power = amdsmi_get_cpu_dimm_power_consumption_p(
            device_handles[dev], dimm, &dimm_pow);
        amdsmi_status_t st_range =
            amdsmi_get_cpu_dimm_temp_range_and_refresh_rate_p(
                device_handles[dev], dimm, &range_info);
        if (st_temp != AMDSMI_STATUS_SUCCESS &&
            st_power != AMDSMI_STATUS_SUCCESS &&
            st_range != AMDSMI_STATUS_SUCCESS)
          continue;
        if (st_temp == AMDSMI_STATUS_SUCCESS) {
          snprintf(name_buf, sizeof(name_buf), "dimm_temp:socket=%d:dimm=%d", s,
                   dimm);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Socket %d DIMM %d temperature (C)", s, dimm);
          if (add_event(&idx, name_buf, descr_buf, dev, 0, dimm, PAPI_MODE_READ,
                        access_amdsmi_dimm_temp) != PAPI_OK)
            return PAPI_ENOMEM;
        }
        if (st_power == AMDSMI_STATUS_SUCCESS) {
          snprintf(name_buf, sizeof(name_buf), "dimm_power:socket=%d:dimm=%d",
                   s, dimm);
          snprintf(descr_buf, sizeof(descr_buf), "Socket %d DIMM %d power (mW)",
                   s, dimm);
          if (add_event(&idx, name_buf, descr_buf, dev, 0, dimm, PAPI_MODE_READ,
                        access_amdsmi_dimm_power) != PAPI_OK)
            return PAPI_ENOMEM;
        }
        if (st_range == AMDSMI_STATUS_SUCCESS) {
          snprintf(name_buf, sizeof(name_buf),
                   "dimm_temp_range:socket=%d:dimm=%d", s, dimm);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Socket %d DIMM %d temperature range", s, dimm);
          if (add_event(&idx, name_buf, descr_buf, dev, 0, dimm, PAPI_MODE_READ,
                        access_amdsmi_dimm_range_refresh) != PAPI_OK)
            return PAPI_ENOMEM;
          snprintf(name_buf, sizeof(name_buf),
                   "dimm_refresh_rate:socket=%d:dimm=%d", s, dimm);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Socket %d DIMM %d refresh rate mode", s, dimm);
          if (add_event(&idx, name_buf, descr_buf, dev, 1, dimm, PAPI_MODE_READ,
                        access_amdsmi_dimm_range_refresh) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    // System-wide CPU events
    uint32_t threads;
    if (amdsmi_get_threads_per_core_p(&threads) == AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "threads_per_core");
      snprintf(descr_buf, sizeof(descr_buf), "SMT threads per core");
      if (add_event(&idx, name_buf, descr_buf, -1, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_threads_per_core) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    uint32_t family;
    if (amdsmi_get_cpu_family_p(&family) == AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "cpu_family");
      snprintf(descr_buf, sizeof(descr_buf), "CPU family ID");
      if (add_event(&idx, name_buf, descr_buf, -1, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_cpu_family) != PAPI_OK)
        return PAPI_ENOMEM;
    }
    uint32_t model;
    if (amdsmi_get_cpu_model_p(&model) == AMDSMI_STATUS_SUCCESS) {
      snprintf(name_buf, sizeof(name_buf), "cpu_model");
      snprintf(descr_buf, sizeof(descr_buf), "CPU model ID");
      if (add_event(&idx, name_buf, descr_buf, -1, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_cpu_model) != PAPI_OK)
        return PAPI_ENOMEM;
    }
  }
#endif

  /* -------- Additional GPU discovery & version info (read-only) -------- */
  /* Library version (global) */
  if (amdsmi_get_lib_version_p) {
    amdsmi_version_t vinfo;
    if (amdsmi_get_lib_version_p(&vinfo) == AMDSMI_STATUS_SUCCESS) {
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "lib_version_major");
      snprintf(descr_buf, sizeof(descr_buf), "AMD SMI library major version");
      if (add_event(&idx, name_buf, descr_buf, -1, 0, 0, PAPI_MODE_READ,
                    access_amdsmi_lib_version) != PAPI_OK)
        return PAPI_ENOMEM;
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "lib_version_minor");
      snprintf(descr_buf, sizeof(descr_buf), "AMD SMI library minor version");
      if (add_event(&idx, name_buf, descr_buf, -1, 1, 0, PAPI_MODE_READ,
                    access_amdsmi_lib_version) != PAPI_OK)
        return PAPI_ENOMEM;
      CHECK_EVENT_IDX(idx);
      snprintf(name_buf, sizeof(name_buf), "lib_version_release");
      snprintf(descr_buf, sizeof(descr_buf),
               "AMD SMI library release/patch version");
      if (add_event(&idx, name_buf, descr_buf, -1, 2, 0, PAPI_MODE_READ,
                    access_amdsmi_lib_version) != PAPI_OK)
        return PAPI_ENOMEM;
    }
  }
  for (int d = 0; d < gpu_count; ++d) {
    if (!device_handles || !device_handles[d])
      continue;
    /* Device UUID (hash) */
    if (amdsmi_get_gpu_device_uuid_p) {
      unsigned int uuid_len = 0;
      amdsmi_status_t st =
          amdsmi_get_gpu_device_uuid_p(device_handles[d], &uuid_len, NULL);
      /* Some builds require preflight to get length; we just attempt a fixed buffer */
      char uuid_buf[128] = {0};
      uuid_len = sizeof(uuid_buf);
      st = amdsmi_get_gpu_device_uuid_p(device_handles[d], &uuid_len, uuid_buf);
      if (st == AMDSMI_STATUS_SUCCESS) {
        uuid_buf[sizeof(uuid_buf) - 1] = '\0';
        sanitize_description_text(uuid_buf);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "uuid_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d UUID hash of '%s'", d, display_or_empty(uuid_buf));
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_uuid_hash) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "uuid_length:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d UUID length", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_uuid_hash) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    /* Vendor / VRAM vendor / Subsystem name (hash) */
    if (amdsmi_get_gpu_vendor_name_p) {
      char tmp[256] = {0};
      if (amdsmi_get_gpu_vendor_name_p(device_handles[d], tmp, sizeof(tmp)) ==
          AMDSMI_STATUS_SUCCESS) {
        sanitize_description_text(tmp);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "vendor_name_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d vendor name hash of '%s'", d,
                 display_or_empty(tmp));
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_string_hash) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_gpu_vram_vendor_p) {
      char tmp[256] = {0};
      if (amdsmi_get_gpu_vram_vendor_p(device_handles[d], tmp,
                                       (uint32_t)sizeof(tmp)) ==
          AMDSMI_STATUS_SUCCESS) {
        sanitize_description_text(tmp);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "vram_vendor_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d VRAM vendor hash of '%s'", d,
                 display_or_empty(tmp));
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_string_hash) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    if (amdsmi_get_gpu_subsystem_name_p) {
      char tmp[256] = {0};
      if (amdsmi_get_gpu_subsystem_name_p(device_handles[d], tmp, sizeof(tmp)) ==
          AMDSMI_STATUS_SUCCESS) {
        sanitize_description_text(tmp);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "subsystem_name_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d subsystem name hash of '%s'", d,
                 display_or_empty(tmp));
        if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_string_hash) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }

    /* Enumeration info (drm render/card, hsa/hip ids) */
#if AMDSMI_LIB_VERSION_MAJOR >= 25
    if (amdsmi_lib_major >= 25 && amdsmi_get_gpu_enumeration_info_p) {
      amdsmi_enumeration_info_t einfo;
      if (amdsmi_get_gpu_enumeration_info_p(device_handles[d], &einfo) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "enum_drm_render:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d DRM render node", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_enumeration_info) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "enum_drm_card:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d DRM card index", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_enumeration_info) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "enum_hsa_id:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d HSA ID", d);
        if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_enumeration_info) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "enum_hip_id:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d HIP ID", d);
        if (add_event(&idx, name_buf, descr_buf, d, 3, 0, PAPI_MODE_READ,
                      access_amdsmi_enumeration_info) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
#endif
    /* ASIC info (numeric IDs & CU count) */
    if (amdsmi_get_gpu_asic_info_p) {
      amdsmi_asic_info_t ainfo;
      if (amdsmi_get_gpu_asic_info_p(device_handles[d], &ainfo) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "asic_vendor_id:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d ASIC vendor id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_asic_info) != PAPI_OK)
          return PAPI_ENOSUPP;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "asic_device_id:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d ASIC device id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 1, 0, PAPI_MODE_READ,
                      access_amdsmi_asic_info) != PAPI_OK)
          return PAPI_ENOSUPP;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "asic_subsystem_vendor_id:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d ASIC subsystem vendor id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 2, 0, PAPI_MODE_READ,
                      access_amdsmi_asic_info) != PAPI_OK)
          return PAPI_ENOSUPP;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "asic_subsystem_id:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d ASIC subsystem id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 3, 0, PAPI_MODE_READ,
                      access_amdsmi_asic_info) != PAPI_OK)
          return PAPI_ENOSUPP;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "asic_revision:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d ASIC revision id", d);
        if (add_event(&idx, name_buf, descr_buf, d, 4, 0, PAPI_MODE_READ,
                      access_amdsmi_asic_info) != PAPI_OK)
          return PAPI_ENOSUPP;

        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "compute_units:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d number of compute units", d);
        if (add_event(&idx, name_buf, descr_buf, d, 5, 0, PAPI_MODE_READ,
                      access_amdsmi_asic_info) != PAPI_OK)
          return PAPI_ENOSUPP;
      }
    }
    if (amdsmi_get_gpu_compute_partition_p) {
      char part[128] = {0};
      if (amdsmi_get_gpu_compute_partition_p(device_handles[d], part,
                                             sizeof(part)) ==
          AMDSMI_STATUS_SUCCESS) {
        part[sizeof(part) - 1] = '\0';
        sanitize_description_text(part);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "compute_partition_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d compute partition hash of '%s'", d,
                 display_or_empty(part));
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_compute_partition_hash) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    if (amdsmi_get_gpu_memory_partition_p) {
      char part[128] = {0};
      uint32_t len = (uint32_t)sizeof(part);
      amdsmi_status_t status =
          amdsmi_get_gpu_memory_partition_p(device_handles[d], part, len);
      part[sizeof(part) - 1] = '\0';  // belt-and-suspenders NUL
      if (status == AMDSMI_STATUS_SUCCESS && part[0] != '\0') {
        sanitize_description_text(part);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "memory_partition_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d memory partition hash of '%s'", d,
                 display_or_empty(part));
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_memory_partition_hash) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    /*
#if AMDSMI_LIB_VERSION_MAJOR >= 25
    if (amdsmi_get_gpu_memory_partition_config_p) {
      amdsmi_memory_partition_config_t cfg = {0};
      // Probe memory partition configuration 
      if (amdsmi_get_gpu_memory_partition_config_p(device_handles[d], &cfg) ==
          AMDSMI_STATUS_SUCCESS) {
        const char *mpc_names[] = {"memory_partition_caps",
                                   "memory_partition_mode",
                                   "memory_partition_numa_count"};
        const char *mpc_descr[] = {"Device %d memory partition capabilities",
                                   "Device %d memory partition mode",
                                   "Device %d NUMA range count"};
        for (uint32_t v = 0; v < 3; ++v) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "%s:device=%d", mpc_names[v], d);
          snprintf(descr_buf, sizeof(descr_buf), mpc_descr[v], d);
          if (add_event(&idx, name_buf, descr_buf, d, v, 0, PAPI_MODE_READ,
                        access_amdsmi_memory_partition_config) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
#endif
    if (amdsmi_get_gpu_accelerator_partition_profile_p) {
      amdsmi_accelerator_partition_profile_t prof = {0};
      uint32_t ids[AMDSMI_MAX_ACCELERATOR_PARTITIONS] = {0};
      amdsmi_status_t status =
          amdsmi_get_gpu_accelerator_partition_profile_p(device_handles[d], &prof, ids);
      if (status == AMDSMI_STATUS_SUCCESS &&
          prof.num_partitions > 0 &&
          prof.num_partitions <= AMDSMI_MAX_ACCELERATOR_PARTITIONS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "accelerator_num_partitions:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d accelerator partition count", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_accelerator_num_partitions) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    */
    /* Driver info (strings hashed) */
    if (amdsmi_get_gpu_driver_info_p) {
      amdsmi_driver_info_t dinfo = {0};
      if (amdsmi_get_gpu_driver_info_p(device_handles[d], &dinfo) ==
          AMDSMI_STATUS_SUCCESS) {
        sanitize_description_text(dinfo.driver_name);
        sanitize_description_text(dinfo.driver_date);
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "driver_name_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d driver name hash of '%s'", d,
                 display_or_empty(dinfo.driver_name));
        if (add_event(&idx, name_buf, descr_buf, d, 3, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_string_hash) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "driver_date_hash:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Device %d driver date hash of '%s'", d,
                 display_or_empty(dinfo.driver_date));
        if (add_event(&idx, name_buf, descr_buf, d, 4, 0, PAPI_MODE_READ,
                      access_amdsmi_gpu_string_hash) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    /* VBIOS info (strings hashed) */
    // (vBIOS events omitted)
    if (amdsmi_get_link_metrics_p) {
      amdsmi_link_metrics_t lm;
      if (amdsmi_get_link_metrics_p(device_handles[d], &lm) ==
          AMDSMI_STATUS_SUCCESS) {
        int types[] = {AMDSMI_LINK_TYPE_XGMI, AMDSMI_LINK_TYPE_PCIE};
        const char *type_names[] = {"xgmi", "pcie"};
        for (int ti = 0; ti < 2; ++ti) {
          uint32_t link_type = (uint32_t)types[ti];
          uint32_t sv = (link_type << 16) | 0xFFFF;
          int present = 0;
          uint32_t n = lm.num_links;
          if (n > AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK)
            n = AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK;
          for (uint32_t li = 0; li < n; ++li) {
            if (lm.links[li].link_type == link_type) {
              present = 1;
              break;
            }
          }
          if (!present)
            continue;
          const char *mnames[] = {"read_kb", "write_kb", "bit_rate",
                                   "max_bandwidth"};
          const char *mdescr[] = {"read throughput (KB)",
                                  "write throughput (KB)",
                                  "link bit rate (Gb/s)",
                                  "max bandwidth (Gb/s)"};
          for (uint32_t v = 0; v < 4; ++v) {
            CHECK_EVENT_IDX(idx);
            snprintf(name_buf, sizeof(name_buf), "%s_%s:device=%d",
                     type_names[ti], mnames[v], d);
            snprintf(descr_buf, sizeof(descr_buf), "Device %d %s %s", d,
                     type_names[ti], mdescr[v]);
            if (add_event(&idx, name_buf, descr_buf, d, v, sv, PAPI_MODE_READ,
                          access_amdsmi_link_metrics) != PAPI_OK)
              return PAPI_ENOMEM;
          }
        }
      }
    }
#if AMDSMI_LIB_VERSION_MAJOR >= 25
    if (amdsmi_get_gpu_xgmi_link_status_p) {
      amdsmi_xgmi_link_status_t st;
      if (amdsmi_get_gpu_xgmi_link_status_p(device_handles[d], &st) ==
          AMDSMI_STATUS_SUCCESS) {
        uint32_t n = st.total_links;
        if (n > AMDSMI_MAX_NUM_XGMI_LINKS)
          n = AMDSMI_MAX_NUM_XGMI_LINKS;
        for (uint32_t li = 0; li < n; ++li) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf),
                   "xgmi_link_status:device=%d:link=%u", d, li);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d XGMI link %u status", d, li);
          if (add_event(&idx, name_buf, descr_buf, d, 0, li, PAPI_MODE_READ,
                        access_amdsmi_xgmi_link_status) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
#endif
    if (amdsmi_gpu_xgmi_error_status_p) {
      amdsmi_xgmi_status_t st;
      if (amdsmi_gpu_xgmi_error_status_p(device_handles[d], &st) ==
          AMDSMI_STATUS_SUCCESS) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf), "xgmi_error_status:device=%d", d);
        snprintf(descr_buf, sizeof(descr_buf), "Device %d XGMI error status", d);
        if (add_event(&idx, name_buf, descr_buf, d, 0, 0, PAPI_MODE_READ,
                      access_amdsmi_xgmi_error_status) != PAPI_OK)
          return PAPI_ENOMEM;
      }
    }
    if (amdsmi_get_link_topology_nearest_p) {
      amdsmi_link_type_t lt_types[] = {AMDSMI_LINK_TYPE_XGMI,
                                       AMDSMI_LINK_TYPE_PCIE};
      const char *lt_names[] = {"xgmi", "pcie"};
      for (int ti = 0; ti < 2; ++ti) {
        amdsmi_topology_nearest_t info;
        memset(&info, 0, sizeof(info));
        if (amdsmi_get_link_topology_nearest_p(device_handles[d], lt_types[ti],
                                               &info) == AMDSMI_STATUS_SUCCESS) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "%s_nearest_count:device=%d",
                   lt_names[ti], d);
          snprintf(descr_buf, sizeof(descr_buf),
                   "Device %d %s nearest GPU count", d, lt_names[ti]);
          if (add_event(&idx, name_buf, descr_buf, d, (uint32_t)lt_types[ti], 0,
                        PAPI_MODE_READ, access_amdsmi_link_topology_nearest) !=
              PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
    }
    for (int p = 0; p < gpu_count; ++p) {
      if (p == d)
        continue;
      if (amdsmi_topo_get_link_weight_p) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "link_weight:device=%d,peer=%d", d, p);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Link weight between device %d and %d", d, p);
        if (add_event(&idx, name_buf, descr_buf, d, 0, p, PAPI_MODE_READ,
                      access_amdsmi_link_weight) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      if (amdsmi_topo_get_link_type_p) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "link_hops:device=%d,peer=%d", d, p);
        snprintf(descr_buf, sizeof(descr_buf),
                 "Hops between device %d and %d", d, p);
        if (add_event(&idx, name_buf, descr_buf, d, 0, p, PAPI_MODE_READ,
                      access_amdsmi_link_type) != PAPI_OK)
          return PAPI_ENOMEM;
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "link_type:device=%d,peer=%d", d, p);
        snprintf(descr_buf, sizeof(descr_buf),
                 "IO link type between device %d and %d", d, p);
        if (add_event(&idx, name_buf, descr_buf, d, 1, p, PAPI_MODE_READ,
                      access_amdsmi_link_type) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      /*
      if (amdsmi_topo_get_p2p_status_p) {
        const char *p2p_names[] = {"p2p_type",       "p2p_coherent",
                                   "p2p_atomics32", "p2p_atomics64",
                                   "p2p_dma",       "p2p_bidir"};
        const char *p2p_desc[] = {
            "P2P IO link type",      "P2P coherent support",
            "P2P 32-bit atomics",   "P2P 64-bit atomics",
            "P2P DMA support",      "P2P bidirectional support"};
        for (int v = 0; v < 6; ++v) {
          CHECK_EVENT_IDX(idx);
          snprintf(name_buf, sizeof(name_buf), "%s:device=%d,peer=%d",
                   p2p_names[v], d, p);
          snprintf(descr_buf, sizeof(descr_buf), "Device %d vs %d %s", d, p,
                   p2p_desc[v]);
          if (add_event(&idx, name_buf, descr_buf, d, v, p, PAPI_MODE_READ,
                        access_amdsmi_p2p_status) != PAPI_OK)
            return PAPI_ENOMEM;
        }
      }
      if (amdsmi_is_P2P_accessible_p) {
        CHECK_EVENT_IDX(idx);
        snprintf(name_buf, sizeof(name_buf),
                 "p2p_accessible:device=%d,peer=%d", d, p);
        snprintf(descr_buf, sizeof(descr_buf),
                 "P2P accessibility between device %d and %d", d, p);
        if (add_event(&idx, name_buf, descr_buf, d, 0, p, PAPI_MODE_READ,
                      access_amdsmi_p2p_accessible) != PAPI_OK)
          return PAPI_ENOMEM;
      }
      */
    }
  }
  ntv_table.count = idx;
  return PAPI_OK;
}

