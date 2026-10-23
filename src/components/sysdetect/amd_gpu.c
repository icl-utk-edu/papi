/**
 * @file    amd_gpu.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  Scan functions for AMD GPU subsystems.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <errno.h>

#include "sysdetect.h"
#include "amd_gpu.h"

#ifdef HAVE_HSA
#include "hsa.h"
#include "hsa_ext_amd.h"

static void *hsa_dlp = NULL;

static hsa_status_t (*hsa_initPtr)( void ) = NULL;
static hsa_status_t (*hsa_shut_downPtr)( void ) = NULL;
static hsa_status_t (*hsa_iterate_agentsPtr)( hsa_status_t (*)(hsa_agent_t agent,
                                                               void *value),
                                              void *value ) = NULL;
static hsa_status_t (*hsa_agent_get_infoPtr)( hsa_agent_t agent,
                                              hsa_agent_info_t attribute,
                                              void *value ) = NULL;
static hsa_status_t (*hsa_amd_agent_iterate_memory_poolsPtr)( hsa_agent_t agent,
                                                              hsa_status_t (*)(hsa_amd_memory_pool_t pool,
                                                                               void *value),
                                                              void *value ) = NULL;
static hsa_status_t (*hsa_amd_memory_pool_get_infoPtr)( hsa_amd_memory_pool_t pool,
                                                        hsa_amd_memory_pool_info_t attribute,
                                                        void *value ) = NULL;
static hsa_status_t (*hsa_status_stringPtr)( hsa_status_t status,
                                             const char **string ) = NULL;

#define CHECK_HSA_API_CALL(call, err_handle) do {    \
    hsa_status_t _status = (call);         \
    if (_status == HSA_STATUS_SUCCESS ||   \
        _status == HSA_STATUS_INFO_BREAK)  \
        break;                             \
    err_handle;                            \
} while(0)

static hsa_status_t count_devices( hsa_agent_t agent, void *data );
static hsa_status_t get_device_count( int *count );
static hsa_status_t get_device_memory( hsa_amd_memory_pool_t pool, void *info );
static hsa_status_t get_device_properties( hsa_agent_t agent, void *info );

static void fill_dev_info( _sysdetect_gpu_info_u *dev_info );
static int hsa_is_enabled( void );
static int load_hsa_sym( char *status );
static int unload_hsa_sym( void );
#endif /* HAVE_HSA */

#if defined(HAVE_AMD_SMI) || defined(HAVE_ROCM_SMI)
static void *smi_dlp = NULL;
static void fill_dev_affinity_info( _sysdetect_gpu_info_u *dev_info, int dev_count );
static void set_status( char *status, int size, const char *message );
static int smi_is_enabled( void );
static int load_smi_sym( char *status );
static void unload_smi_sym( void );

#if defined(HAVE_AMD_SMI)
#include "amdsmi.h"

#define SMI_STATUS_SUCCESS AMDSMI_STATUS_SUCCESS
typedef amdsmi_status_t smi_status_t;
static smi_status_t (*smi_initPtr)( uint64_t init_flags ) = NULL;
static smi_status_t (*smi_shut_downPtr)( void ) = NULL;
static smi_status_t (*smi_get_socket_handlesPtr)( uint32_t *socket_count, amdsmi_socket_handle* socket_handles ) = NULL;
static smi_status_t (*smi_get_processor_handlesPtr)( amdsmi_socket_handle socket_handle,
                                                     uint32_t *processor_count,
                                                     amdsmi_processor_handle* processor_handles ) = NULL;
static smi_status_t (*smi_get_device_pci_idPtr)( amdsmi_processor_handle processor_handle, uint64_t *bdfid ) = NULL;
#else
#include "rocm_smi.h"

#define SMI_STATUS_SUCCESS RSMI_STATUS_SUCCESS
typedef rsmi_status_t smi_status_t;
static smi_status_t (*smi_initPtr)( uint64_t init_flags ) = NULL;
static smi_status_t (*smi_shut_downPtr)( void ) = NULL;
static smi_status_t (*smi_get_device_pci_idPtr)( uint32_t dv_ind, uint64_t *bdfid ) = NULL;
#endif /* HAVE_AMD_SMI */

#define SMI_CALL(call, err_handle) do {        \
    smi_status_t _status = (call);             \
    if (_status == SMI_STATUS_SUCCESS)         \
        break;                                 \
    err_handle;                                \
} while(0)
#endif /* defined(HAVE_AMD_SMI) || defined(HAVE_ROCM_SMI) */

#ifdef HAVE_HSA
hsa_status_t
count_devices( hsa_agent_t agent, void *data )
{
    int *count = (int *) data;

    hsa_device_type_t type;
    CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent, HSA_AGENT_INFO_DEVICE, &type),
                       return _status);

    if (type == HSA_DEVICE_TYPE_GPU) {
        ++(*count);
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_device_count( int *count )
{
    *count = 0;

    CHECK_HSA_API_CALL((*hsa_iterate_agentsPtr)(&count_devices, count),
                       return _status);

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_device_memory( hsa_amd_memory_pool_t pool, void *info )
{
    hsa_region_segment_t seg_info;
    _sysdetect_gpu_info_u *dev_info = info;

    CHECK_HSA_API_CALL((*hsa_amd_memory_pool_get_infoPtr)(pool,
                                                         HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                                         &seg_info),
                                                         return _status);

    if (seg_info == HSA_REGION_SEGMENT_GROUP) {
        CHECK_HSA_API_CALL((*hsa_amd_memory_pool_get_infoPtr)(pool,
                                                              HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                              &dev_info->amd.max_shmmem_per_workgroup),
                                                              return _status);
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_device_properties( hsa_agent_t agent, void *info )
{
    static int count;

    hsa_device_type_t type;
    CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent, HSA_AGENT_INFO_DEVICE, &type),
                       return _status);

    if (type == HSA_DEVICE_TYPE_GPU) {
        /* query attributes for this device */
        _sysdetect_gpu_info_u *dev_info = &((_sysdetect_gpu_info_u *) info)[count];

        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent,
                                                    HSA_AGENT_INFO_NAME,
                                                    dev_info->amd.name),
                                                    return _status);
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent,
                                                    HSA_AGENT_INFO_WAVEFRONT_SIZE,
                                                    &dev_info->amd.wavefront_size),
                                                    return _status);
        unsigned short wg_dims[3];
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent,
                                                    HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                                                    wg_dims),
                                                    return _status);
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent,
                                                    HSA_AGENT_INFO_WORKGROUP_MAX_SIZE,
                                                    &dev_info->amd.max_threads_per_workgroup),
                                                    return _status);
        hsa_dim3_t gr_dims;
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent,
                                                    HSA_AGENT_INFO_GRID_MAX_DIM,
                                                    &gr_dims),
                                                    return _status);
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent,
                                                    HSA_AGENT_INFO_VERSION_MAJOR,
                                                    &dev_info->amd.major),
                                                    return _status);
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent,
                                                    HSA_AGENT_INFO_VERSION_MINOR,
                                                    &dev_info->amd.minor),
                                                    return _status);
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent, (hsa_agent_info_t)
                                                    HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU,
                                                    &dev_info->amd.simd_per_compute_unit),
                                                    return _status);
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent, (hsa_agent_info_t)
                                                    HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                                                    &dev_info->amd.compute_unit_count),
                                                    return _status);
        CHECK_HSA_API_CALL((*hsa_agent_get_infoPtr)(agent, (hsa_agent_info_t)
                                                    HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
                                                    &dev_info->amd.max_waves_per_compute_unit),
                                                    return _status);
        CHECK_HSA_API_CALL((*hsa_amd_agent_iterate_memory_poolsPtr)(agent,
                                                                    &get_device_memory,
                                                                    dev_info),
                                                                    return _status);

        dev_info->amd.max_workgroup_dim_x = wg_dims[0];
        dev_info->amd.max_workgroup_dim_y = wg_dims[1];
        dev_info->amd.max_workgroup_dim_z = wg_dims[2];
        dev_info->amd.max_grid_dim_x  = gr_dims.x;
        dev_info->amd.max_grid_dim_y  = gr_dims.y;
        dev_info->amd.max_grid_dim_z  = gr_dims.z;

        ++count;
    }

    return HSA_STATUS_SUCCESS;
}

void
fill_dev_info( _sysdetect_gpu_info_u *dev_info )
{
    hsa_status_t status = HSA_STATUS_SUCCESS;
    const char *string = NULL;

    CHECK_HSA_API_CALL((*hsa_iterate_agentsPtr)(&get_device_properties, dev_info),
                       status = _status);

    if (status != HSA_STATUS_SUCCESS) {
        (*hsa_status_stringPtr)(status, &string);
        SUBDBG( "error: %s\n", string );
    }
}

int
hsa_is_enabled( void )
{
    return (hsa_initPtr                           != NULL &&
            hsa_shut_downPtr                      != NULL &&
            hsa_iterate_agentsPtr                 != NULL &&
            hsa_agent_get_infoPtr                 != NULL &&
            hsa_amd_agent_iterate_memory_poolsPtr != NULL &&
            hsa_amd_memory_pool_get_infoPtr       != NULL &&
            hsa_status_stringPtr                  != NULL);
}

int
load_hsa_sym( char *status )
{
    char pathname[PATH_MAX] = "libhsa-runtime64.so";
    char *hsa_root = NULL;
    // Default to use PAPI_ROCP_SDK_ROOT env.
    if ((hsa_root = getenv("PAPI_ROCP_SDK_ROOT")) != NULL) {
        sprintf(pathname, "%s/lib/libhsa-runtime64.so", hsa_root);
    }
    // Fallback to use PAPI_ROCM_ROOT env.
    else if ((hsa_root = getenv("PAPI_ROCM_ROOT")) != NULL) {
        sprintf(pathname, "%s/lib/libhsa-runtime64.so", hsa_root);
    }

    hsa_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (hsa_dlp == NULL) {
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", dlerror());
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    hsa_initPtr                           = dlsym(hsa_dlp, "hsa_init");
    hsa_shut_downPtr                      = dlsym(hsa_dlp, "hsa_shut_down");
    hsa_iterate_agentsPtr                 = dlsym(hsa_dlp, "hsa_iterate_agents");
    hsa_agent_get_infoPtr                 = dlsym(hsa_dlp, "hsa_agent_get_info");
    hsa_amd_agent_iterate_memory_poolsPtr = dlsym(hsa_dlp, "hsa_amd_agent_iterate_memory_pools");
    hsa_amd_memory_pool_get_infoPtr       = dlsym(hsa_dlp, "hsa_amd_memory_pool_get_info");
    hsa_status_stringPtr                  = dlsym(hsa_dlp, "hsa_status_string");

    if (!hsa_is_enabled() || (*hsa_initPtr)()) {
        const char *message = "dlsym() of HSA symbols failed or hsa_init() "
                              "failed";
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", message);
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    return 0;
}

int
unload_hsa_sym( void )
{
    if (hsa_dlp != NULL) {
        (*hsa_shut_downPtr)();
        dlclose(hsa_dlp);
    }

    hsa_initPtr                           = NULL;
    hsa_shut_downPtr                      = NULL;
    hsa_iterate_agentsPtr                 = NULL;
    hsa_agent_get_infoPtr                 = NULL;
    hsa_amd_agent_iterate_memory_poolsPtr = NULL;
    hsa_amd_memory_pool_get_infoPtr       = NULL;
    hsa_status_stringPtr                  = NULL;

    return hsa_is_enabled();
}
#endif /* HAVE_HSA */

#if defined(HAVE_AMD_SMI) || defined(HAVE_ROCM_SMI)
void
fill_dev_affinity_info( _sysdetect_gpu_info_u *info, int dev_count )
{
    uint64_t uid = 0;
#if defined(HAVE_AMD_SMI)
    (void) dev_count;

    uint32_t socket_count = 0;
    SMI_CALL((*smi_get_socket_handlesPtr)(&socket_count, NULL), return);
    if (socket_count == 0) {
        SUBDBG("No sockets detected with AMD GPUs.\n");
        return;
    }

    amdsmi_socket_handle *socket_handles = (amdsmi_socket_handle *) malloc(socket_count * sizeof(amdsmi_socket_handle));
    if (socket_handles == NULL) {
        SUBDBG("Failed to allocate memory for variable socket_handles.\n");
        return;
    }

    SMI_CALL((*smi_get_socket_handlesPtr)(&socket_count, socket_handles), goto cleanup_socket_handle);

    // From testing on Odyssey at Oregon and Frontier at ORNL (10/15/2026), the number of sockets will match the number of devices
    // obtained via the internal function call get_device_count (uses hsa_iterate_agents) if amdsmi_init is called
    // with the argument AMDSMI_INIT_AMD_GPUS. Therefore, socket is the index to use below for &info[].
    uint32_t socket;
    for (socket = 0; socket < socket_count; socket++) {
        uint32_t processor_count = 0;
        SMI_CALL((*smi_get_processor_handlesPtr)(socket_handles[socket], &processor_count, NULL), goto cleanup_socket_handle);
        if (processor_count == 0) {
            SUBDBG("No processors are available to read for the socket.\n");
            continue; 
        }

        // From testing on Odyssey at Oregon and Frontier at ORNL (10/15/2026), the number of processor handles per socket is 1.
        amdsmi_processor_handle processor_handle;
        SMI_CALL((*smi_get_processor_handlesPtr)(socket_handles[socket], &processor_count, &processor_handle), goto cleanup_socket_handle);

        SMI_CALL((*smi_get_device_pci_idPtr)(processor_handle, &uid), goto cleanup_socket_handle);

        _sysdetect_gpu_info_u *dev_info = &info[socket];
        dev_info->amd.uid = uid;
    }

cleanup_socket_handle:
    free(socket_handles);
    socket_handles = NULL;
#else
    int dev;
    for (dev = 0; dev < dev_count; ++dev) {
        SMI_CALL((*smi_get_device_pci_idPtr)(dev, &uid), return);

        _sysdetect_gpu_info_u *dev_info = &info[dev];
        dev_info->amd.uid = uid;
    }
#endif /* HAVE_AMD_SMI */

    return;
}

void
set_status( char *status, int size, const char *message )
{
    int str_len = snprintf(status, size, "%s", message);
    if (str_len < 0 || str_len >= size) {
        SUBDBG("The status string has been truncated; however, proceeding.\n");
    }

    return;
}

int
smi_is_enabled( void )
{
    return (smi_initPtr                     != NULL
            && smi_shut_downPtr             != NULL
            && smi_get_device_pci_idPtr     != NULL
#if defined(HAVE_AMD_SMI)
            && smi_get_socket_handlesPtr    != NULL
            && smi_get_processor_handlesPtr != NULL
#endif /* HAVE_AMD_SMI */
           ); 
}

int
load_smi_sym( char *status )
{
    const char *smi_lib_name = NULL, *smi_env_name = NULL;
    const char *init_symbol = NULL, *shutdown_symbol = NULL, *get_device_pci_id_symbol = NULL;
    uint64_t init_flags = -1; 
#if defined(HAVE_AMD_SMI)
    smi_lib_name = "libamd_smi.so";
    smi_env_name = "PAPI_AMDSMI_ROOT";

    init_symbol = "amdsmi_init";
    shutdown_symbol = "amdsmi_shut_down";
    get_device_pci_id_symbol = "amdsmi_get_gpu_bdf_id";

    const char *get_socket_handles_symbol = "amdsmi_get_socket_handles";
    const char *get_processor_handles_symbol = "amdsmi_get_processor_handles";

    init_flags = AMDSMI_INIT_AMD_GPUS;
#else
    smi_lib_name = "librocm_smi64.so";
    smi_env_name = "PAPI_ROCMSMI_ROOT";

    init_symbol = "rsmi_init";
    shutdown_symbol = "rsmi_shut_down";
    get_device_pci_id_symbol = "rsmi_dev_pci_id_get";

    init_flags = 0;
#endif /* HAVE_AMD_SMI */

    int str_len;
    char path_to_smi[PATH_MAX] = { 0 };
    const char *smi_root = getenv(smi_env_name);
    if (smi_root != NULL) {
        str_len = snprintf(path_to_smi, sizeof(path_to_smi), "%s/lib/%s", smi_root, smi_lib_name);
    }
    else {
        str_len = snprintf(path_to_smi, sizeof(path_to_smi), "%s", smi_lib_name);
    }

    const char *error_message = NULL;
    if (str_len < 0 || (size_t) str_len >= sizeof(path_to_smi)) {
        error_message = "The path to the smi library is greater than PATH_MAX.";
        set_status(status, PAPI_MAX_STR_LEN, error_message);

        return PAPI_EBUF; 
    }

    smi_dlp = dlopen(path_to_smi, RTLD_NOW | RTLD_GLOBAL);
    if (smi_dlp == NULL) {
        error_message = dlerror();
        set_status(status, PAPI_MAX_STR_LEN, error_message);

        return PAPI_ESYS;
    }

    smi_initPtr = dlsym(smi_dlp, init_symbol);
    smi_shut_downPtr = dlsym(smi_dlp, shutdown_symbol);
    smi_get_device_pci_idPtr = dlsym(smi_dlp, get_device_pci_id_symbol);
#if defined(HAVE_AMD_SMI)
    smi_get_socket_handlesPtr = dlsym(smi_dlp, get_socket_handles_symbol);
    smi_get_processor_handlesPtr = dlsym(smi_dlp, get_processor_handles_symbol);
#endif /* HAVE_AMD_SMI */

    if (!smi_is_enabled()) {
        error_message = "Failed to load smi symbols.";
        set_status(status, PAPI_MAX_STR_LEN, error_message);
        unload_smi_sym();

        return PAPI_ESYS;
    }

    if ((*smi_initPtr)(init_flags)) {
        error_message = "Failed to initialize the smi library.";
        set_status(status, PAPI_MAX_STR_LEN, error_message);
        unload_smi_sym();

        return PAPI_ESYS;
    }

    return PAPI_OK;
}

void
unload_smi_sym( void )
{
    if (smi_dlp != NULL) {
        (*smi_shut_downPtr)();
        int status = dlclose(smi_dlp);
        if (status != 0) {
            SUBDBG("Failed to dlclose handle smi_dlp. The reason given: %s\n", dlerror());
        }
        smi_dlp = NULL;
    }

    smi_initPtr                  = NULL;
    smi_shut_downPtr             = NULL;
    smi_get_device_pci_idPtr     = NULL;
#if defined(HAVE_AMD_SMI)
    smi_get_socket_handlesPtr    = NULL;
    smi_get_processor_handlesPtr = NULL;
#endif /* HAVE_AMD_SMI */

    return;
}
#endif /* defined(HAVE_AMD_SMI) || defined(HAVE_ROCM_SMI) */

void
open_amd_gpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info )
{
    memset(dev_type_info, 0, sizeof(*dev_type_info));
    dev_type_info->id = PAPI_DEV_TYPE_ID__ROCM;
    strcpy(dev_type_info->vendor, "AMD/ATI");
    strcpy(dev_type_info->status, "Device Initialized");

#ifdef HAVE_HSA
    if (load_hsa_sym(dev_type_info->status)) {
        return;
    }

    int dev_count = 0;
    hsa_status_t status = get_device_count(&dev_count);
    if (status != HSA_STATUS_SUCCESS) {
        const char *message = "Unable to iterate over avaiable agents.";
        int count = snprintf(dev_type_info->status, PAPI_MAX_STR_LEN, "%s", message);
        if (count < 0 || count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Error message truncated.\n");
        }

        return;
    }
    dev_type_info->num_devices = dev_count;

    _sysdetect_gpu_info_u *arr = papi_calloc(dev_count, sizeof(*arr));
    fill_dev_info(arr);

#if defined(HAVE_AMD_SMI) || defined(HAVE_ROCM_SMI)
    if (!load_smi_sym(dev_type_info->status)) {
        fill_dev_affinity_info(arr, dev_count);
        unload_smi_sym();
    }
#else
    const char *message = "PAPI_AMDSMI_ROOT (for ROCm >= 6.4.0) nor PAPI_ROCMSMI_ROOT (for ROCm < 6.4.0) set; therefore, no device affinity available.";
    int count = snprintf(dev_type_info->status, PAPI_MAX_STR_LEN, "%s", message);
    if (count < 0 || count >= PAPI_MAX_STR_LEN) {
        SUBDBG("The status string for rocm_smi/amd_smi has been truncated; however, proceeding.\n");
    }
#endif /* HAVE_AMD_SMI || HAVE_ROCM_SMI */

    unload_hsa_sym();
    dev_type_info->dev_info_arr = (_sysdetect_dev_info_u *)arr;
#else
    const char *message = "rocm nor rocp_sdk configured. Set PAPI_ROCP_SDK_ROOT for ROCm versions >= 6.3.2 or"
                          " PAPI_ROCM_ROOT for ROCm versions < 6.3.2.";
    int count = snprintf(dev_type_info->status, PAPI_MAX_STR_LEN, "%s", message);
    if (count >= PAPI_MAX_STR_LEN) {
        SUBDBG("Error message truncated.");
    }
#endif /* HAVE_HSA */
}

void
close_amd_gpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info )
{
    papi_free(dev_type_info->dev_info_arr);
}
