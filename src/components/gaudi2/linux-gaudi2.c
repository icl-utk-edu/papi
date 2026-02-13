/**
 * @file    linux-gaudi2.c
 *
 * @author  Tokey Tahmid ttahmid@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This file implements a PAPI component for the Intel Gaudi2 SPMU counters
 *  Accesses hardware performance counters on Gaudi2 AI accelerators
 *  via the hlthunk library debug interface.
 *
 * The open source software license for PAPI conforms to the BSD
 * License template.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <errno.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/types.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "gaudi2_events.h"

#define GAUDI2_MAX_COUNTERS 32
#define GAUDI2_MAX_DEVICES  16  /* Max supported devices per node */

/* Eventset status flags */
#define GAUDI2_EVENTS_STOPPED   (0x0)
#define GAUDI2_EVENTS_RUNNING   (0x2)

/* Event code encoding:
 * Bits 0-7:   name ID (base event index in catalog)
 * Bits 8-15:  device index (0-255)
 * Bits 16-23: flags (DEVICE_FLAG for device qualifier display)
 *
 * When flags=0: base event (for PAPI_ENUM_EVENTS enumeration)
 * When flags=DEVICE_FLAG: device qualifier entry (for PAPI_NTV_ENUM_UMASKS)
 */
#define GAUDI2_NAMEID_SHIFT   0
#define GAUDI2_NAMEID_WIDTH   8
#define GAUDI2_DEVICE_SHIFT   8
#define GAUDI2_DEVICE_WIDTH   8
#define GAUDI2_FLAGS_SHIFT    16
#define GAUDI2_FLAGS_WIDTH    8

#define GAUDI2_NAMEID_MASK   ((0xFF) << GAUDI2_NAMEID_SHIFT)
#define GAUDI2_DEVICE_MASK   ((0xFF) << GAUDI2_DEVICE_SHIFT)
#define GAUDI2_FLAGS_MASK    ((0xFF) << GAUDI2_FLAGS_SHIFT)

/* Flag definitions */
#define GAUDI2_DEVICE_FLAG   0x1  /* Device qualifier entry */

/* Event info structure for encoding/decoding */
typedef struct {
    int nameid;   /* Index in catalog */
    int device;   /* Device index */
    int flags;    /* GAUDI2_DEVICE_FLAG or 0 */
} gaudi2_event_info_t;

static int gaudi2_evt_id_create(gaudi2_event_info_t *info, unsigned int *event_code)
{
    *event_code  = (unsigned int)(info->nameid << GAUDI2_NAMEID_SHIFT);
    *event_code |= (unsigned int)(info->device << GAUDI2_DEVICE_SHIFT);
    *event_code |= (unsigned int)(info->flags  << GAUDI2_FLAGS_SHIFT);
    return PAPI_OK;
}

static int gaudi2_evt_id_to_info(unsigned int event_code, gaudi2_event_info_t *info)
{
    info->nameid = (event_code & GAUDI2_NAMEID_MASK) >> GAUDI2_NAMEID_SHIFT;
    info->device = (event_code & GAUDI2_DEVICE_MASK) >> GAUDI2_DEVICE_SHIFT;
    info->flags  = (event_code & GAUDI2_FLAGS_MASK)  >> GAUDI2_FLAGS_SHIFT;
    return PAPI_OK;
}

/* hlthunk library header - provides:
 *   struct hl_debug_args, hl_debug_params_spmu, hlthunk_hw_ip_info
 *   enum hlthunk_device_name (HLTHUNK_DEVICE_GAUDI2, etc.)
 *   HLTHUNK_NODE_PRIMARY, HLTHUNK_NODE_CONTROL, HLTHUNK_MAX_MINOR
 *   HL_DEBUG_OP_*, HL_DEBUG_MAX_AUX_VALUES
 *   Function declarations for hlthunk_open, hlthunk_debug, etc.
 */
#include "hlthunk.h"

/* Function pointer types for dlsym */
typedef int (*hlthunk_open_fn)(enum hlthunk_device_name device_name, const char *busid);
typedef int (*hlthunk_close_fn)(int fd);
typedef int (*hlthunk_debug_fn)(int fd, struct hl_debug_args *debug);
typedef enum hlthunk_device_name (*hlthunk_get_device_name_from_fd_fn)(int fd);
typedef int (*hlthunk_get_hw_ip_info_fn)(int fd, struct hlthunk_hw_ip_info *hw_ip);
typedef int (*hlthunk_get_device_count_fn)(enum hlthunk_device_name device_name);

static void *hlthunk_handle = NULL;

static hlthunk_open_fn  p_hlthunk_open  = NULL;
static hlthunk_close_fn p_hlthunk_close = NULL;
static hlthunk_debug_fn p_hlthunk_debug = NULL;
static hlthunk_get_device_name_from_fd_fn p_hlthunk_get_device_name_from_fd = NULL;
static hlthunk_get_hw_ip_info_fn p_hlthunk_get_hw_ip_info = NULL;
static hlthunk_get_device_count_fn p_hlthunk_get_device_count = NULL;

/* Per-device state */
typedef struct {
    int device_idx;                      /* Index in device array (0 to num_devices-1) */
    int device_fd;                       /* File descriptor for this device */
    int device_type;                     /* HLTHUNK_DEVICE_GAUDI2/2B/2C/2D */
    struct hlthunk_hw_ip_info hw_ip;     /* Hardware IP info */
    int tpc_avail;                       /* TPC engine available */
    int edma_avail;                      /* EDMA engine available */
    int mme_avail;                       /* MME engine available */
    int pdma_avail;                      /* PDMA engine available */
} gaudi2_device_t;

/* Device table - populated during init */
static gaudi2_device_t *gaudi2_devices = NULL;
static int gaudi2_num_devices = 0;

/** 
* Event Catalog
* TODO: add all gaudi2 events 
*/
static gaudi2_native_event_t gaudi2_event_catalog[] = {
    /* TPC backpressure */
    {"TPC_MEMORY2SB_BP", "Memory to SB backpressure", GAUDI2_ENGINE_TPC, TPC_SPMU_MEMORY2SB_BP},
    {"TPC_SB2MEMORY_BP", "SB to memory backpressure", GAUDI2_ENGINE_TPC, TPC_SPMU_SB2MEMORY_BP},
    {"TPC_PQ_NOT_EMPTY_BUT_CQ_EMPTY", "PQ not empty but CQ empty", GAUDI2_ENGINE_TPC, TPC_SPMU_PQ_NOT_EMPTY_BUT_CQ_EMPTY},
    {"TPC_QM_PREFETCH_BUFFER_EMPTY", "QM prefetch buffer empty", GAUDI2_ENGINE_TPC, TPC_SPMU_QM_PREFETCH_BUFFER_EMPTY},
    {"TPC_SB_2_CORE_BP", "SB to core backpressure", GAUDI2_ENGINE_TPC, TPC_SPMU_SB_2_CORE_BP},
    {"TPC_SB_2_CORE_BP_SB_FULL", "SB to core BP - SB full", GAUDI2_ENGINE_TPC, TPC_SPMU_SB_2_CORE_BP_SB_FULL},
    {"TPC_SB_2_CORE_BP_SB_MEMORY", "SB to core BP - SB memory", GAUDI2_ENGINE_TPC, TPC_SPMU_SB_2_CORE_BP_SB_MEMORY},
    {"TPC_SB_2_CORE_BP_SB_LD_TNSR_FIFO_FULL", "SB to core BP - LD tensor FIFO full", GAUDI2_ENGINE_TPC, TPC_SPMU_SB_2_CORE_BP_SB_LD_TNSR_FIFO_FULL},
    {"TPC_WB2CORE_BP", "Write buffer to core backpressure", GAUDI2_ENGINE_TPC, TPC_SPMU_WB2CORE_BP},

    /* TPC stalls */
    {"TPC_STALL_ON_ICACHE_MISS", "Scalar pipe stall on instruction cache miss", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_ICACHE_MISS},
    {"TPC_STALL_ON_DCACHE_MISS", "Scalar pipe stall on data cache miss", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_DCACHE_MISS},
    {"TPC_STALL_ON_POP_FROM_SB", "Stall on pop from SB", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_POP_FROM_SB},
    {"TPC_STALL_ON_LOOKUP_CACHE_MISS", "Stall on lookup cache miss", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_LOOKUP_CACHE_MISS},
    {"TPC_STALL_ON_IRQ_FULL", "Stall on IRQ full", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_IRQ_FULL},
    {"TPC_STALL_ON_MAX_COLORS", "Stall on max colors", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_MAX_COLORS},
    {"TPC_STALL_ON_UARCH_BUBBLE", "Stall on microarchitecture bubble", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_UARCH_BUBBLE},
    {"TPC_STALL_VPU", "Vector processing unit stall", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_VPU},
    {"TPC_STALL_SPU_ANY", "Scalar processing unit any stall", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_SPU_ANY},
    {"TPC_STALL_ON_TSB_FULL", "Stall on TSB full", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_TSB_FULL},
    {"TPC_STALL_ON_ST_L_EXT", "Stall on store local external", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_ST_L_EXT},
    {"TPC_STALL_ON_LD_L_EXT", "Stall on load local external", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL_ON_LD_L_EXT},
    {"TPC_STALL", "Total stall cycles", GAUDI2_ENGINE_TPC, TPC_SPMU_STALL},

    /* TPC opcode execution */
    {"TPC_NUM_OF_OPCODE1_EXECUTED", "Number of opcode1 executed", GAUDI2_ENGINE_TPC, TPC_SPMU_NUM_OF_OPCODE1_EXECUTED},
    {"TPC_NUM_OF_OPCODE2_EXECUTED", "Number of opcode2 executed", GAUDI2_ENGINE_TPC, TPC_SPMU_NUM_OF_OPCODE2_EXECUTED},
    {"TPC_NUM_OF_OPCODE3_EXECUTED", "Number of opcode3 executed", GAUDI2_ENGINE_TPC, TPC_SPMU_NUM_OF_OPCODE3_EXECUTED},
    {"TPC_NUM_OF_OPCODE4_EXECUTED", "Number of opcode4 executed", GAUDI2_ENGINE_TPC, TPC_SPMU_NUM_OF_OPCODE4_EXECUTED},

    /* TPC execution */
    {"TPC_KERNEL_EXECUTED", "TPC kernels executed", GAUDI2_ENGINE_TPC, TPC_SPMU_KERNEL_EXECUTED},
    {"TPC_SCALAR_PIPE_EXEC", "Scalar pipe execution cycles", GAUDI2_ENGINE_TPC, TPC_SPMU_SCALAR_PIPE_EXEC},
    {"TPC_VECTOR_PIPE_EXEC", "Vector pipe execution cycles", GAUDI2_ENGINE_TPC, TPC_SPMU_VECTOR_PIPE_EXEC},

    /* TPC cache */
    {"TPC_ICACHE_MISS", "Instruction cache miss", GAUDI2_ENGINE_TPC, TPC_SPMU_ICACHE_MISS},
    {"TPC_ICACHE_HIT", "Instruction cache hit", GAUDI2_ENGINE_TPC, TPC_SPMU_ICACHE_HIT},
    {"TPC_KILLED_INSTRUCTION", "Killed instructions", GAUDI2_ENGINE_TPC, TPC_SPMU_KILLED_INSTRUCTION},
    {"TPC_LUT_MISS", "Lookup table miss", GAUDI2_ENGINE_TPC, TPC_SPMU_LUT_MISS},
    {"TPC_DCACHE_MISS", "Data cache miss", GAUDI2_ENGINE_TPC, TPC_SPMU_DCACHE_MISS},
    {"TPC_DCACHE_HIT", "Data cache hit", GAUDI2_ENGINE_TPC, TPC_SPMU_DCACHE_HIT},

    /* TPC exceptions */
    {"TPC_DIV_BY_0", "Division by zero", GAUDI2_ENGINE_TPC, TPC_SPMU_DIV_BY_0},
    {"TPC_SPU_MAC_OVERFLOW", "SPU MAC overflow", GAUDI2_ENGINE_TPC, TPC_SPMU_SPU_MAC_OVERFLOW},
    {"TPC_VPU_MAC_OVERFLOW", "VPU MAC overflow", GAUDI2_ENGINE_TPC, TPC_SPMU_VPU_MAC_OVERFLOW},
    {"TPC_LUT_HIT", "Lookup table hit", GAUDI2_ENGINE_TPC, TPC_SPMU_LUT_HIT},
    {"TPC_DCACHE_HW_PREF", "Data cache hardware prefetch", GAUDI2_ENGINE_TPC, TPC_SPMU_DCACHE_HW_PREF},

    /* EDMA */
    {"EDMA_TRACE_FENCE_START", "EDMA fence start", GAUDI2_ENGINE_EDMA, EDMA_SPMU_TRACE_FENCE_START},
    {"EDMA_TRACE_FENCE_DONE", "EDMA fence done", GAUDI2_ENGINE_EDMA, EDMA_SPMU_TRACE_FENCE_DONE},
    {"EDMA_DESC_PUSH", "EDMA descriptor push", GAUDI2_ENGINE_EDMA, EDMA_SPMU_TRACE_CHOICE_WIN_PUSH},

    /* MME */
    {"MME_NUM_OUTER_PRODUCTS", "MME outer products", GAUDI2_ENGINE_MME, MME_CTRL_SPMU_NUM_OUTER_PRODUCTS},
    {"MME_STALL_ON_A", "MME stall waiting for A matrix", GAUDI2_ENGINE_MME, MME_CTRL_SPMU_OUTER_PRODUCT_STALL_ON_A},
    {"MME_STALL_ON_B", "MME stall waiting for B matrix", GAUDI2_ENGINE_MME, MME_CTRL_SPMU_OUTER_PRODUCT_STALL_ON_B},

    {NULL, NULL, 0, 0}
};

/* Number of base events in catalog (computed at init) */
static int gaudi2_num_catalog_events = 0;

/* Per-event tracking for an eventset */
typedef struct {
    unsigned int event_code;       /* Encoded event code (device + index) */
    int device_idx;                /* Device index */
    int catalog_idx;               /* Catalog event index */
    unsigned int counter_idx;      /* Counter slot (0-5 per SPMU) */
    uint64_t spmu_base;            /* SPMU base address */
    long long last_value;
    long long accumulated;
} gaudi2_counter_t;

/* Per-device tracking within an eventset */
typedef struct {
    int device_idx;
    int num_events;                           /* Events for this device */
    int event_indices[GAUDI2_MAX_COUNTERS];   /* Indices into counters[] */
    int debug_mode_enabled;
    int spmu_enabled;
} gaudi2_device_ctl_t;

/* Per-eventset state */
typedef struct {
    gaudi2_counter_t counters[GAUDI2_MAX_COUNTERS];
    int num_counters;
    long long values[GAUDI2_MAX_COUNTERS];
    int running;
    /* Per-device control within this eventset */
    gaudi2_device_ctl_t device_ctl[GAUDI2_MAX_DEVICES];
    uint32_t active_device_mask;  /* Bitmap of devices with events */
    int num_active_devices;
} gaudi2_control_t;

/* Per-thread context - tracks debug mode per device */
typedef struct {
    int debug_mode_enabled[GAUDI2_MAX_DEVICES];
} gaudi2_context_t;
static unsigned int gaudi2_lock;

papi_vector_t _gaudi2_vector;

/* Load hlthunk library */
static int load_hlthunk_library(void)
{
    char root_lib_path[PAPI_HUGE_STR_LEN];
    const char *gaudi2_root;
    int strLen;

    gaudi2_root = getenv("PAPI_GAUDI2_ROOT");

    if (gaudi2_root != NULL) {
        strLen = snprintf(root_lib_path, sizeof(root_lib_path),
                          "%s/lib/habanalabs/libhl-thunk.so", gaudi2_root);
        if (strLen > 0 && strLen < (int)sizeof(root_lib_path)) {
            hlthunk_handle = dlopen(root_lib_path, RTLD_NOW | RTLD_GLOBAL);
            if (hlthunk_handle) {
                SUBDBG("Loaded libhl-thunk.so from PAPI_GAUDI2_ROOT: %s\n", root_lib_path);
            }
        }
    }

    /* Fallback */
    if (!hlthunk_handle) {
        const char *fallback_paths[] = {
            "/usr/lib/habanalabs/libhl-thunk.so",
            "libhl-thunk.so",
            NULL
        };

        for (int i = 0; fallback_paths[i] != NULL; i++) {
            hlthunk_handle = dlopen(fallback_paths[i], RTLD_NOW | RTLD_GLOBAL);
            if (hlthunk_handle) {
                SUBDBG("Loaded libhl-thunk.so from fallback: %s\n", fallback_paths[i]);
                break;
            }
        }
    }

    if (!hlthunk_handle) {
        SUBDBG("Failed to load libhl-thunk.so: %s\n", dlerror());
        return PAPI_ENOSUPP;
    }

    p_hlthunk_open = (hlthunk_open_fn)dlsym(hlthunk_handle, "hlthunk_open");
    p_hlthunk_close = (hlthunk_close_fn)dlsym(hlthunk_handle, "hlthunk_close");
    p_hlthunk_debug = (hlthunk_debug_fn)dlsym(hlthunk_handle, "hlthunk_debug");
    p_hlthunk_get_device_name_from_fd = (hlthunk_get_device_name_from_fd_fn)
        dlsym(hlthunk_handle, "hlthunk_get_device_name_from_fd");
    p_hlthunk_get_hw_ip_info = (hlthunk_get_hw_ip_info_fn)
        dlsym(hlthunk_handle, "hlthunk_get_hw_ip_info");
    p_hlthunk_get_device_count = (hlthunk_get_device_count_fn)
        dlsym(hlthunk_handle, "hlthunk_get_device_count");

    if (!p_hlthunk_open || !p_hlthunk_close || !p_hlthunk_debug ||
        !p_hlthunk_get_device_name_from_fd || !p_hlthunk_get_hw_ip_info ||
        !p_hlthunk_get_device_count) {
        SUBDBG("Failed to find required hlthunk symbols\n");
        dlclose(hlthunk_handle);
        hlthunk_handle = NULL;
        return PAPI_ENOSUPP;
    }

    return PAPI_OK;
}

/* Open a device by minor number */
static int open_device_by_minor(int minor, int node_type)
{
    char path[64];
    const char *fmt;
    int strLen;

    if (node_type == HLTHUNK_NODE_PRIMARY)
        fmt = "/dev/accel/accel%d";
    else
        fmt = "/dev/accel/accel_controlD%d";

    strLen = snprintf(path, sizeof(path), fmt, minor);
    if (strLen < 0 || strLen >= (int)sizeof(path))
        return -1;
    return open(path, O_RDWR | O_CLOEXEC, 0);
}

/* Check if a device type is a Gaudi2 variant */
static int is_gaudi2_device(int device_type)
{
    return (device_type == HLTHUNK_DEVICE_GAUDI2  ||
            device_type == HLTHUNK_DEVICE_GAUDI2B ||
            device_type == HLTHUNK_DEVICE_GAUDI2C ||
            device_type == HLTHUNK_DEVICE_GAUDI2D);
}

/**
 * Find existing Gaudi2 device fd from /proc/self/fd
 * Returns fd for the first found device, or -1 if none found.
 */
static int find_gaudi2_device_fd(void)
{
    DIR *dir;
    struct dirent *entry;
    char link_path[PAPI_MIN_STR_LEN];
    char target[PAPI_HUGE_STR_LEN];
    ssize_t len;
    int found_fd = -1;
    int strLen;
    int status;

    dir = opendir("/proc/self/fd");
    if (!dir) {
        SUBDBG("Failed to open /proc/self/fd: %s\n", strerror(errno));
        return PAPI_ESYS;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.')
            continue;

        strLen = snprintf(link_path, sizeof(link_path), "/proc/self/fd/%s", entry->d_name);
        if (strLen < 0 || strLen >= (int)sizeof(link_path)) {
            SUBDBG("snprintf overflow for /proc/self/fd/%s\n", entry->d_name);
            continue;
        }

        len = readlink(link_path, target, sizeof(target) - 1);
        if (len < 0)
            continue;
        target[len] = '\0';

        if (strstr(target, "/dev/accel/accel") != NULL &&
            strstr(target, "control") == NULL) {
            found_fd = atoi(entry->d_name);
            SUBDBG("Found Gaudi2 device: fd=%d -> %s\n", found_fd, target);
            break;
        }
    }

    status = closedir(dir);
    if (status == -1) {
        SUBDBG("closedir failed for /proc/self/fd: %s\n", strerror(errno));
        return PAPI_ESYS;
    }

    return found_fd;
}

static int enable_debug_mode(int fd)
{
    struct hl_debug_args debug;

    memset(&debug, 0, sizeof(debug));
    debug.op = HL_DEBUG_OP_SET_MODE;
    debug.enable = 1;

    if (p_hlthunk_debug(fd, &debug) < 0) {
        SUBDBG("Failed to enable debug mode on fd=%d\n", fd);
        return PAPI_ESYS;
    }

    return PAPI_OK;
}

static int disable_debug_mode(int fd)
{
    struct hl_debug_args debug;

    memset(&debug, 0, sizeof(debug));
    debug.op = HL_DEBUG_OP_SET_MODE;
    debug.enable = 0;

    p_hlthunk_debug(fd, &debug);
    return PAPI_OK;
}

static int enable_spmu(int fd, int reg_idx, uint64_t *events, int num_events)
{
    struct hl_debug_params_spmu params;
    struct hl_debug_args debug;

    memset(&params, 0, sizeof(params));
    for (int i = 0; i < num_events && i < HL_DEBUG_MAX_AUX_VALUES; i++) {
        params.event_types[i] = events[i];
    }
    params.event_types_num = num_events;

    memset(&debug, 0, sizeof(debug));
    debug.op = HL_DEBUG_OP_SPMU;
    debug.reg_idx = reg_idx;
    debug.enable = 1;
    debug.input_ptr = (uint64_t)&params;
    debug.input_size = sizeof(params);

    if (p_hlthunk_debug(fd, &debug) < 0) {
        SUBDBG("Failed to enable SPMU on fd=%d reg_idx=%d\n", fd, reg_idx);
        return PAPI_ESYS;
    }

    return PAPI_OK;
}

static int disable_spmu(int fd, int reg_idx)
{
    struct hl_debug_args debug;

    memset(&debug, 0, sizeof(debug));
    debug.op = HL_DEBUG_OP_SPMU;
    debug.reg_idx = reg_idx;
    debug.enable = 0;

    p_hlthunk_debug(fd, &debug);
    return PAPI_OK;
}

/* Read SPMU counters via READBLOCK */
static int read_spmu_counters(int fd, uint64_t base_addr, int num_counters, long long *values)
{
    struct hl_debug_params_read_block params;
    struct hl_debug_args debug;
    void *read_buffer;
    int papi_errno = PAPI_OK;
    int i;

    for (i = 0; i < num_counters; i++)
        values[i] = 0;

    read_buffer = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (read_buffer == MAP_FAILED) {
        SUBDBG("mmap failed for SPMU read buffer\n");
        papi_errno = PAPI_ENOMEM;
        goto cleanup;
    }

    memset(read_buffer, 0, 4096);

    memset(&params, 0, sizeof(params));
    params.cfg_address = base_addr;
    params.user_address = (uint64_t)read_buffer;
    params.size = 256;
    params.flags = 0;

    memset(&debug, 0, sizeof(debug));
    debug.op = HL_DEBUG_OP_READBLOCK;
    debug.input_ptr = (uint64_t)&params;
    debug.input_size = sizeof(params);

    if (p_hlthunk_debug(fd, &debug) < 0) {
        SUBDBG("READBLOCK failed for base_addr=0x%llx\n", (unsigned long long)base_addr);
        papi_errno = PAPI_ESYS;
        goto cleanup;
    }

    /* Extract lower 32 bits of each 64-bit counter */
    uint32_t *counter_data = (uint32_t *)read_buffer;
    for (i = 0; i < num_counters; i++)
        values[i] = (long long)counter_data[i * 2];

cleanup:
    if (read_buffer != MAP_FAILED)
        munmap(read_buffer, 4096);
    return papi_errno;
}

static uint64_t get_spmu_base_address(gaudi2_engine_type_t engine, int dcore, int instance)
{
    static const uint64_t tpc_spmu_bases[GAUDI2_NUM_DCORES][GAUDI2_TPC_PER_DCORE] = {
        {GAUDI2_DCORE0_TPC0_SPMU_BASE, GAUDI2_DCORE0_TPC1_SPMU_BASE,
         GAUDI2_DCORE0_TPC2_SPMU_BASE, GAUDI2_DCORE0_TPC3_SPMU_BASE,
         GAUDI2_DCORE0_TPC4_SPMU_BASE, GAUDI2_DCORE0_TPC5_SPMU_BASE},
        {GAUDI2_DCORE1_TPC0_SPMU_BASE, GAUDI2_DCORE1_TPC1_SPMU_BASE,
         GAUDI2_DCORE1_TPC2_SPMU_BASE, GAUDI2_DCORE1_TPC3_SPMU_BASE,
         GAUDI2_DCORE1_TPC4_SPMU_BASE, GAUDI2_DCORE1_TPC5_SPMU_BASE},
        {GAUDI2_DCORE2_TPC0_SPMU_BASE, GAUDI2_DCORE2_TPC1_SPMU_BASE,
         GAUDI2_DCORE2_TPC2_SPMU_BASE, GAUDI2_DCORE2_TPC3_SPMU_BASE,
         GAUDI2_DCORE2_TPC4_SPMU_BASE, GAUDI2_DCORE2_TPC5_SPMU_BASE},
        {GAUDI2_DCORE3_TPC0_SPMU_BASE, GAUDI2_DCORE3_TPC1_SPMU_BASE,
         GAUDI2_DCORE3_TPC2_SPMU_BASE, GAUDI2_DCORE3_TPC3_SPMU_BASE,
         GAUDI2_DCORE3_TPC4_SPMU_BASE, GAUDI2_DCORE3_TPC5_SPMU_BASE}
    };

    static const uint64_t edma_spmu_bases[GAUDI2_NUM_DCORES][GAUDI2_EDMA_PER_DCORE] = {
        {GAUDI2_DCORE0_EDMA0_SPMU_BASE, GAUDI2_DCORE0_EDMA1_SPMU_BASE},
        {GAUDI2_DCORE1_EDMA0_SPMU_BASE, GAUDI2_DCORE1_EDMA1_SPMU_BASE},
        {GAUDI2_DCORE2_EDMA0_SPMU_BASE, GAUDI2_DCORE2_EDMA1_SPMU_BASE},
        {GAUDI2_DCORE3_EDMA0_SPMU_BASE, GAUDI2_DCORE3_EDMA1_SPMU_BASE}
    };

    switch (engine) {
        case GAUDI2_ENGINE_TPC:
            if (dcore < GAUDI2_NUM_DCORES && instance < GAUDI2_TPC_PER_DCORE) {
                return tpc_spmu_bases[dcore][instance];
            }
            break;
        case GAUDI2_ENGINE_EDMA:
            if (dcore < GAUDI2_NUM_DCORES && instance < GAUDI2_EDMA_PER_DCORE) {
                return edma_spmu_bases[dcore][instance];
            }
            break;
        case GAUDI2_ENGINE_PDMA:
            if (instance == 0) return GAUDI2_PDMA0_SPMU_BASE;
            if (instance == 1) return GAUDI2_PDMA1_SPMU_BASE;
            break;
        default:
            break;
    }

    return GAUDI2_DCORE0_TPC0_SPMU_BASE;
}

/*
 * PAPI component interface
 */

/* Enumerate all Gaudi2 devices and populate device table */
static int enumerate_gaudi2_devices(void)
{
    int minor, ctrl_fd, dev_fd, device_type;
    int num_found = 0;

    /* First pass: count Gaudi2 devices */
    for (minor = 0; minor < HLTHUNK_MAX_MINOR && num_found < GAUDI2_MAX_DEVICES; minor++) {
        ctrl_fd = open_device_by_minor(minor, HLTHUNK_NODE_CONTROL);
        if (ctrl_fd < 0)
            continue;

        device_type = p_hlthunk_get_device_name_from_fd(ctrl_fd);
        close(ctrl_fd);

        if (is_gaudi2_device(device_type))
            num_found++;
    }

    if (num_found == 0) {
        SUBDBG("No Gaudi2 devices found\n");
        return 0;
    }

    /* Allocate device table */
    gaudi2_devices = (gaudi2_device_t *)papi_calloc(num_found, sizeof(gaudi2_device_t));
    if (!gaudi2_devices) {
        SUBDBG("Failed to allocate device table\n");
        return -1;
    }

    /* Second pass: populate device table */
    gaudi2_num_devices = 0;
    for (minor = 0; minor < HLTHUNK_MAX_MINOR && gaudi2_num_devices < num_found; minor++) {
        ctrl_fd = open_device_by_minor(minor, HLTHUNK_NODE_CONTROL);
        if (ctrl_fd < 0)
            continue;

        device_type = p_hlthunk_get_device_name_from_fd(ctrl_fd);
        if (!is_gaudi2_device(device_type)) {
            close(ctrl_fd);
            continue;
        }

        /* Open primary device */
        dev_fd = open_device_by_minor(minor, HLTHUNK_NODE_PRIMARY);
        close(ctrl_fd);

        if (dev_fd < 0) {
            SUBDBG("Failed to open primary device for minor %d\n", minor);
            continue;
        }

        gaudi2_device_t *dev = &gaudi2_devices[gaudi2_num_devices];
        dev->device_idx = gaudi2_num_devices;
        dev->device_fd = dev_fd;
        dev->device_type = device_type;

        /* Query hardware IP info */
        memset(&dev->hw_ip, 0, sizeof(dev->hw_ip));
        if (p_hlthunk_get_hw_ip_info(dev_fd, &dev->hw_ip) != 0) {
            SUBDBG("Failed to get hw_ip_info for device %d\n", gaudi2_num_devices);
            close(dev_fd);
            continue;
        }

        /* Determine engine availability */
        dev->tpc_avail = (dev->hw_ip.tpc_enabled_mask_ext != 0);
        dev->edma_avail = (dev->hw_ip.edma_enabled_mask != 0);
        dev->mme_avail = 1;  /* Always present on Gaudi2 */
        dev->pdma_avail = 1; /* Always present on Gaudi2 */

        SUBDBG("Device %d: fd=%d type=%d TPC=%d EDMA=%d MME=%d PDMA=%d\n",
               gaudi2_num_devices, dev_fd, device_type,
               dev->tpc_avail, dev->edma_avail, dev->mme_avail, dev->pdma_avail);

        gaudi2_num_devices++;
    }

    return gaudi2_num_devices;
}

/* Check if an event is available on a specific device */
static int event_available_on_device(gaudi2_native_event_t *event, gaudi2_device_t *dev)
{
    switch (event->engine) {
        case GAUDI2_ENGINE_TPC:  return dev->tpc_avail;
        case GAUDI2_ENGINE_EDMA: return dev->edma_avail;
        case GAUDI2_ENGINE_MME:  return dev->mme_avail;
        case GAUDI2_ENGINE_PDMA: return dev->pdma_avail;
        default: return 0;
    }
}

static int gaudi2_init_component(int cidx)
{
    int papi_errno = PAPI_OK;

    SUBDBG("Initializing Gaudi2 component (cidx=%d)\n", cidx);

    _gaudi2_vector.cmp_info.CmpIdx = cidx;

    /* Load hlthunk library */
    papi_errno = load_hlthunk_library();
    if (papi_errno != PAPI_OK) {
        int strLen = snprintf(_gaudi2_vector.cmp_info.disabled_reason,
                 PAPI_HUGE_STR_LEN, "Failed to load libhl-thunk.so");
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN)
            _gaudi2_vector.cmp_info.disabled_reason[0] = '\0';
        _gaudi2_vector.cmp_info.disabled = papi_errno;
        return papi_errno;
    }

    /* Enumerate all Gaudi2 devices */
    int num_devices = enumerate_gaudi2_devices();
    if (num_devices <= 0) {
        papi_errno = PAPI_ENOSUPP;
        int strLen = snprintf(_gaudi2_vector.cmp_info.disabled_reason,
                 PAPI_HUGE_STR_LEN, "No Gaudi2 devices found");
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN)
            _gaudi2_vector.cmp_info.disabled_reason[0] = '\0';
        _gaudi2_vector.cmp_info.disabled = papi_errno;
        return papi_errno;
    }

    SUBDBG("Found %d Gaudi2 device(s)\n", num_devices);

    /* Count catalog events */
    gaudi2_num_catalog_events = 0;
    while (gaudi2_event_catalog[gaudi2_num_catalog_events].name != NULL)
        gaudi2_num_catalog_events++;

    if (gaudi2_num_catalog_events == 0) {
        papi_errno = PAPI_ENOSUPP;
        SUBDBG("No events in catalog\n");
        int strLen = snprintf(_gaudi2_vector.cmp_info.disabled_reason,
                 PAPI_HUGE_STR_LEN, "No events defined in catalog");
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN)
            _gaudi2_vector.cmp_info.disabled_reason[0] = '\0';
        _gaudi2_vector.cmp_info.disabled = papi_errno;
        return papi_errno;
    }

    SUBDBG("Catalog has %d base events, %d devices available\n",
           gaudi2_num_catalog_events, gaudi2_num_devices);

    /* Device qualifiers are enumerated via PAPI_NTV_ENUM_UMASKS */
    _gaudi2_vector.cmp_info.num_native_events = gaudi2_num_catalog_events;
    _gaudi2_vector.cmp_info.num_cntrs = GAUDI2_MAX_SPMU_COUNTERS;
    _gaudi2_vector.cmp_info.num_mpx_cntrs = GAUDI2_MAX_COUNTERS;

    gaudi2_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;
    _gaudi2_vector.cmp_info.initialized = 1;

    return PAPI_OK;
}

static int gaudi2_shutdown_component(void)
{
    int d;

    /* Close all device file descriptors */
    if (gaudi2_devices) {
        for (d = 0; d < gaudi2_num_devices; d++) {
            if (gaudi2_devices[d].device_fd >= 0) {
                close(gaudi2_devices[d].device_fd);
                gaudi2_devices[d].device_fd = -1;
            }
        }
        papi_free(gaudi2_devices);
        gaudi2_devices = NULL;
    }
    gaudi2_num_devices = 0;
    gaudi2_num_catalog_events = 0;

    if (hlthunk_handle) {
        dlclose(hlthunk_handle);
        hlthunk_handle = NULL;
    }

    _gaudi2_vector.cmp_info.initialized = 0;
    return PAPI_OK;
}

static int gaudi2_init_thread(hwd_context_t *ctx)
{
    gaudi2_context_t *gaudi2_ctx = (gaudi2_context_t *)ctx;

    memset(gaudi2_ctx, 0, sizeof(gaudi2_context_t));
    return PAPI_OK;
}

static int gaudi2_shutdown_thread(hwd_context_t *ctx)
{
    gaudi2_context_t *gaudi2_ctx = (gaudi2_context_t *)ctx;
    int d;

    /* Disable debug mode on all devices where we enabled it */
    for (d = 0; d < gaudi2_num_devices; d++) {
        if (gaudi2_ctx->debug_mode_enabled[d] && gaudi2_devices[d].device_fd >= 0) {
            disable_debug_mode(gaudi2_devices[d].device_fd);
            gaudi2_ctx->debug_mode_enabled[d] = 0;
        }
    }

    return PAPI_OK;
}

static int gaudi2_init_control_state(hwd_control_state_t *ctl)
{
    memset(ctl, 0, sizeof(gaudi2_control_t));
    return PAPI_OK;
}

static int gaudi2_cleanup_eventset(hwd_control_state_t *ctl)
{
    memset(ctl, 0, sizeof(gaudi2_control_t));
    return PAPI_OK;
}

static int gaudi2_update_control_state(hwd_control_state_t *ctl,
                                       NativeInfo_t *native,
                                       int count,
                                       hwd_context_t *ctx)
{
    gaudi2_control_t *gaudi2_ctl = (gaudi2_control_t *)ctl;
    gaudi2_event_info_t evt_info;
    int i, d, papi_errno;
    (void)ctx;

    if (count > GAUDI2_MAX_COUNTERS) {
        SUBDBG("Event count %d exceeds max %d\n", count, GAUDI2_MAX_COUNTERS);
        return PAPI_ECOUNT;
    }

    /* Reset control state */
    memset(gaudi2_ctl->device_ctl, 0, sizeof(gaudi2_ctl->device_ctl));
    gaudi2_ctl->active_device_mask = 0;
    gaudi2_ctl->num_active_devices = 0;
    gaudi2_ctl->num_counters = count;

    /* Process each event and organize by device */
    for (i = 0; i < count; i++) {
        unsigned int event_code = native[i].ni_event;

        /* Decode event code to get nameid, device, and flags */
        papi_errno = gaudi2_evt_id_to_info(event_code, &evt_info);
        if (papi_errno != PAPI_OK) {
            SUBDBG("Failed to decode event code %u\n", event_code);
            return papi_errno;
        }

        if (evt_info.nameid < 0 || evt_info.nameid >= gaudi2_num_catalog_events) {
            SUBDBG("Invalid nameid %d (max %d)\n", evt_info.nameid, gaudi2_num_catalog_events);
            return PAPI_EINVAL;
        }

        if (evt_info.device < 0 || evt_info.device >= gaudi2_num_devices) {
            SUBDBG("Invalid device %d (max %d)\n", evt_info.device, gaudi2_num_devices);
            return PAPI_EINVAL;
        }

        gaudi2_native_event_t *cat_evt = &gaudi2_event_catalog[evt_info.nameid];

        /* Set up counter tracking */
        gaudi2_ctl->counters[i].event_code = event_code;
        gaudi2_ctl->counters[i].device_idx = evt_info.device;
        gaudi2_ctl->counters[i].catalog_idx = evt_info.nameid;
        gaudi2_ctl->counters[i].spmu_base = get_spmu_base_address(cat_evt->engine, 0, 0);
        gaudi2_ctl->counters[i].last_value = 0;
        gaudi2_ctl->counters[i].accumulated = 0;

        /* Track this device */
        if (!(gaudi2_ctl->active_device_mask & (1 << evt_info.device))) {
            gaudi2_ctl->active_device_mask |= (1 << evt_info.device);
            gaudi2_ctl->device_ctl[evt_info.device].device_idx = evt_info.device;
        }

        /* Add event to device's event list */
        gaudi2_device_ctl_t *dev_ctl = &gaudi2_ctl->device_ctl[evt_info.device];
        dev_ctl->event_indices[dev_ctl->num_events] = i;
        gaudi2_ctl->counters[i].counter_idx = dev_ctl->num_events % GAUDI2_MAX_SPMU_COUNTERS;
        dev_ctl->num_events++;

        native[i].ni_position = i;
    }

    /* Count active devices */
    for (d = 0; d < gaudi2_num_devices; d++) {
        if (gaudi2_ctl->active_device_mask & (1 << d))
            gaudi2_ctl->num_active_devices++;
    }

    SUBDBG("Configured %d events across %d devices (mask=0x%x)\n",
           count, gaudi2_ctl->num_active_devices, gaudi2_ctl->active_device_mask);

    return PAPI_OK;
}

static int gaudi2_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    gaudi2_context_t *gaudi2_ctx = (gaudi2_context_t *)ctx;
    gaudi2_control_t *gaudi2_ctl = (gaudi2_control_t *)ctl;
    int d, i, papi_errno;

    /* For each device with events, enable debug mode and SPMU */
    for (d = 0; d < gaudi2_num_devices; d++) {
        if (!(gaudi2_ctl->active_device_mask & (1 << d)))
            continue;

        gaudi2_device_ctl_t *dev_ctl = &gaudi2_ctl->device_ctl[d];
        int dev_fd = gaudi2_devices[d].device_fd;

        /* Enable debug mode on this device if not already enabled */
        if (!gaudi2_ctx->debug_mode_enabled[d]) {
            papi_errno = enable_debug_mode(dev_fd);
            if (papi_errno != PAPI_OK) {
                SUBDBG("Failed to enable debug mode on device %d\n", d);
                return papi_errno;
            }
            gaudi2_ctx->debug_mode_enabled[d] = 1;
        }

        /* Build event array for this device */
        uint64_t events[HL_DEBUG_MAX_AUX_VALUES];
        int num_dev_events = dev_ctl->num_events;
        if (num_dev_events > HL_DEBUG_MAX_AUX_VALUES)
            num_dev_events = HL_DEBUG_MAX_AUX_VALUES;

        for (i = 0; i < num_dev_events; i++) {
            int counter_idx = dev_ctl->event_indices[i];
            int catalog_idx = gaudi2_ctl->counters[counter_idx].catalog_idx;
            events[i] = gaudi2_event_catalog[catalog_idx].event_id;
        }

        /* Enable SPMU on this device */
        papi_errno = enable_spmu(dev_fd, 0, events, num_dev_events);
        if (papi_errno != PAPI_OK) {
            SUBDBG("Failed to enable SPMU on device %d\n", d);
            return papi_errno;
        }
        dev_ctl->spmu_enabled = 1;

        SUBDBG("Started %d events on device %d\n", num_dev_events, d);
    }

    /* Reset all counter values */
    for (i = 0; i < gaudi2_ctl->num_counters; i++) {
        gaudi2_ctl->counters[i].last_value = 0;
        gaudi2_ctl->counters[i].accumulated = 0;
    }

    gaudi2_ctl->running = GAUDI2_EVENTS_RUNNING;
    return PAPI_OK;
}

static int gaudi2_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    gaudi2_context_t *gaudi2_ctx = (gaudi2_context_t *)ctx;
    gaudi2_control_t *gaudi2_ctl = (gaudi2_control_t *)ctl;
    int d, i;
    (void)gaudi2_ctx;

    if (gaudi2_ctl->running == GAUDI2_EVENTS_RUNNING) {
        /* Read final values from each device */
        for (d = 0; d < gaudi2_num_devices; d++) {
            if (!(gaudi2_ctl->active_device_mask & (1 << d)))
                continue;

            gaudi2_device_ctl_t *dev_ctl = &gaudi2_ctl->device_ctl[d];
            int dev_fd = gaudi2_devices[d].device_fd;

            /* Read counters for this device */
            long long temp_values[GAUDI2_MAX_SPMU_COUNTERS];
            int num_dev_events = dev_ctl->num_events;
            if (num_dev_events > GAUDI2_MAX_SPMU_COUNTERS)
                num_dev_events = GAUDI2_MAX_SPMU_COUNTERS;

            /* Get SPMU base for first event on this device */
            int first_counter_idx = dev_ctl->event_indices[0];
            uint64_t base = gaudi2_ctl->counters[first_counter_idx].spmu_base;

            if (read_spmu_counters(dev_fd, base, num_dev_events, temp_values) == PAPI_OK) {
                for (i = 0; i < num_dev_events; i++) {
                    int counter_idx = dev_ctl->event_indices[i];
                    gaudi2_ctl->counters[counter_idx].accumulated += temp_values[i];
                    gaudi2_ctl->values[counter_idx] = gaudi2_ctl->counters[counter_idx].accumulated;
                }
            }

            /* Disable SPMU on this device */
            if (dev_ctl->spmu_enabled) {
                disable_spmu(dev_fd, 0);
                dev_ctl->spmu_enabled = 0;
            }
        }
    }

    gaudi2_ctl->running = GAUDI2_EVENTS_STOPPED;
    return PAPI_OK;
}

static int gaudi2_read(hwd_context_t *ctx, hwd_control_state_t *ctl,
                       long long **events, int flags)
{
    gaudi2_context_t *gaudi2_ctx = (gaudi2_context_t *)ctx;
    gaudi2_control_t *gaudi2_ctl = (gaudi2_control_t *)ctl;
    int d, i;
    (void)gaudi2_ctx;
    (void)flags;

    if (gaudi2_ctl->running == GAUDI2_EVENTS_RUNNING) {
        /* Read current values from each device */
        for (d = 0; d < gaudi2_num_devices; d++) {
            if (!(gaudi2_ctl->active_device_mask & (1 << d)))
                continue;

            gaudi2_device_ctl_t *dev_ctl = &gaudi2_ctl->device_ctl[d];
            int dev_fd = gaudi2_devices[d].device_fd;

            long long temp_values[GAUDI2_MAX_SPMU_COUNTERS];
            int num_dev_events = dev_ctl->num_events;
            if (num_dev_events > GAUDI2_MAX_SPMU_COUNTERS)
                num_dev_events = GAUDI2_MAX_SPMU_COUNTERS;

            int first_counter_idx = dev_ctl->event_indices[0];
            uint64_t base = gaudi2_ctl->counters[first_counter_idx].spmu_base;

            if (read_spmu_counters(dev_fd, base, num_dev_events, temp_values) == PAPI_OK) {
                for (i = 0; i < num_dev_events; i++) {
                    int counter_idx = dev_ctl->event_indices[i];
                    gaudi2_ctl->values[counter_idx] =
                        gaudi2_ctl->counters[counter_idx].accumulated + temp_values[i];
                }
            }
        }
    }

    *events = gaudi2_ctl->values;
    return PAPI_OK;
}

static int gaudi2_reset(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    gaudi2_control_t *gaudi2_ctl = (gaudi2_control_t *)ctl;
    (void)ctx;

    for (int i = 0; i < gaudi2_ctl->num_counters; i++) {
        gaudi2_ctl->counters[i].last_value = 0;
        gaudi2_ctl->counters[i].accumulated = 0;
        gaudi2_ctl->values[i] = 0;
    }

    return PAPI_OK;
}

static int gaudi2_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    gaudi2_event_info_t info;
    int papi_errno = PAPI_OK;

    switch (modifier) {
        case PAPI_ENUM_FIRST:
            /* Return first base event */
            if (gaudi2_num_catalog_events == 0)
                return PAPI_ENOEVNT;
            info.nameid = 0;
            info.device = 0;
            info.flags = 0;
            papi_errno = gaudi2_evt_id_create(&info, EventCode);
            return papi_errno;

        case PAPI_ENUM_EVENTS:
            /* Iterate through base events only */
            papi_errno = gaudi2_evt_id_to_info(*EventCode, &info);
            if (papi_errno != PAPI_OK)
                return papi_errno;
            if (info.nameid + 1 < gaudi2_num_catalog_events) {
                info.nameid++;
                info.device = 0;
                info.flags = 0;
                return gaudi2_evt_id_create(&info, EventCode);
            }
            return PAPI_ENOEVNT;

        case PAPI_NTV_ENUM_UMASKS:
            /* Enumerate device qualifier */
            papi_errno = gaudi2_evt_id_to_info(*EventCode, &info);
            if (papi_errno != PAPI_OK)
                return papi_errno;
            /* If flags=0 (base event), return device qualifier entry */
            if (info.flags == 0) {
                info.device = 0;
                info.flags = GAUDI2_DEVICE_FLAG;
                return gaudi2_evt_id_create(&info, EventCode);
            }
            /* Only one qualifier (device) */
            return PAPI_ENOEVNT;

        default:
            return PAPI_EINVAL;
    }
}

static int gaudi2_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    gaudi2_event_info_t info;
    int papi_errno;

    papi_errno = gaudi2_evt_id_to_info(EventCode, &info);
    if (papi_errno != PAPI_OK)
        return papi_errno;

    if (info.nameid < 0 || info.nameid >= gaudi2_num_catalog_events) {
        SUBDBG("nameid %d out of range (max %d)\n", info.nameid, gaudi2_num_catalog_events);
        return PAPI_ENOEVNT;
    }

    int strLen;

    switch (info.flags) {
        case GAUDI2_DEVICE_FLAG:
            /* Event with device qualifier */
            strLen = snprintf(name, len, "%s:device=%d",
                     gaudi2_event_catalog[info.nameid].name, info.device);
            break;
        default:
            /* Base event (flags=0) */
            strLen = snprintf(name, len, "%s", gaudi2_event_catalog[info.nameid].name);
            break;
    }

    if (strLen < 0 || strLen >= len)
        return PAPI_EINVAL;

    return PAPI_OK;
}

/* Parse event name and convert to event code */
static int gaudi2_ntv_name_to_code(const char *name, unsigned int *EventCode)
{
    char base_name[PAPI_HUGE_STR_LEN];
    gaudi2_event_info_t info;
    const char *device_ptr;
    int i;

    /* Copy name to extract base */
    strncpy(base_name, name, sizeof(base_name) - 1);
    base_name[sizeof(base_name) - 1] = '\0';

    /* Default device and flags */
    info.device = 0;
    info.flags = GAUDI2_DEVICE_FLAG;

    device_ptr = strstr(name, ":device=");
    if (device_ptr != NULL) {
        info.device = atoi(device_ptr + 8);
        base_name[device_ptr - name] = '\0';
    }

    /* Validate device index */
    if (info.device < 0 || info.device >= gaudi2_num_devices) {
        SUBDBG("Invalid device %d in event name '%s' (max %d)\n",
               info.device, name, gaudi2_num_devices - 1);
        return PAPI_ENOEVNT;
    }

    /* Find base event in catalog */
    info.nameid = -1;
    for (i = 0; i < gaudi2_num_catalog_events; i++) {
        if (strcmp(base_name, gaudi2_event_catalog[i].name) == 0) {
            info.nameid = i;
            break;
        }
    }

    if (info.nameid < 0) {
        SUBDBG("Event '%s' (base='%s') not found in catalog\n", name, base_name);
        return PAPI_ENOEVNT;
    }

    /* Check if event is available on specified device */
    if (!event_available_on_device(&gaudi2_event_catalog[info.nameid],
                                    &gaudi2_devices[info.device])) {
        SUBDBG("Event '%s' not available on device %d\n", base_name, info.device);
        return PAPI_ENOEVNT;
    }

    return gaudi2_evt_id_create(&info, EventCode);
}

static int gaudi2_ntv_code_to_descr(unsigned int EventCode, char *descr, int len)
{
    gaudi2_event_info_t info;
    int papi_errno;

    papi_errno = gaudi2_evt_id_to_info(EventCode, &info);
    if (papi_errno != PAPI_OK)
        return papi_errno;

    if (info.nameid < 0 || info.nameid >= gaudi2_num_catalog_events) {
        SUBDBG("nameid %d out of range (max %d)\n", info.nameid, gaudi2_num_catalog_events);
        return PAPI_ENOEVNT;
    }

    int strLen = snprintf(descr, len, "%s", gaudi2_event_catalog[info.nameid].description);
    if (strLen < 0 || strLen >= len)
        return PAPI_EINVAL;
    return PAPI_OK;
}

static int gaudi2_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info)
{
    gaudi2_event_info_t evt_info;
    char devices[PAPI_HUGE_STR_LEN];
    int papi_errno;
    int d, first_avail_device = 0;
    int strLen;
    size_t offset;

    papi_errno = gaudi2_evt_id_to_info(EventCode, &evt_info);
    if (papi_errno != PAPI_OK)
        return papi_errno;

    if (evt_info.nameid < 0 || evt_info.nameid >= gaudi2_num_catalog_events) {
        SUBDBG("nameid %d out of range (max %d)\n", evt_info.nameid, gaudi2_num_catalog_events);
        return PAPI_ENOEVNT;
    }

    gaudi2_native_event_t *cat_evt = &gaudi2_event_catalog[evt_info.nameid];

    devices[0] = '\0';
    offset = 0;
    for (d = 0; d < gaudi2_num_devices; d++) {
        if (event_available_on_device(cat_evt, &gaudi2_devices[d])) {
            if (offset == 0) {
                first_avail_device = d;
            }
            if (offset > 0) {
                strLen = snprintf(devices + offset, sizeof(devices) - offset, ",");
                if (strLen < 0 || strLen >= (int)(sizeof(devices) - offset))
                    return PAPI_EINVAL;
                offset += strLen;
            }
            strLen = snprintf(devices + offset, sizeof(devices) - offset, "%d", d);
            if (strLen < 0 || strLen >= (int)(sizeof(devices) - offset))
                return PAPI_EINVAL;
            offset += strLen;
        }
    }

    switch (evt_info.flags) {
        case GAUDI2_DEVICE_FLAG:
            /* Device qualifier entry - shown when enumerating UMASKS */
            strLen = snprintf(info->symbol, sizeof(info->symbol),
                              "%s:device=%d", cat_evt->name, first_avail_device);
            if (strLen < 0 || strLen >= (int)sizeof(info->symbol))
                return PAPI_EINVAL;
            strLen = snprintf(info->long_descr, sizeof(info->long_descr),
                              "%s masks:Mandatory device qualifier [%s]",
                              cat_evt->description, devices);
            if (strLen < 0 || strLen >= (int)sizeof(info->long_descr))
                return PAPI_EINVAL;
            break;

        default:
            /* Base event (flags=0) - shown when enumerating events */
            strLen = snprintf(info->symbol, sizeof(info->symbol),
                              "%s", cat_evt->name);
            if (strLen < 0 || strLen >= (int)sizeof(info->symbol))
                return PAPI_EINVAL;
            strLen = snprintf(info->long_descr, sizeof(info->long_descr),
                              "%s", cat_evt->description);
            if (strLen < 0 || strLen >= (int)sizeof(info->long_descr))
                return PAPI_EINVAL;
            break;
    }

    strLen = snprintf(info->short_descr, sizeof(info->short_descr),
                      "%s", cat_evt->description);
    if (strLen < 0 || strLen >= (int)sizeof(info->short_descr))
        return PAPI_EINVAL;
    info->event_code = EventCode;
    info->component_index = _gaudi2_vector.cmp_info.CmpIdx;

    return PAPI_OK;
}

static int gaudi2_set_domain(hwd_control_state_t *ctl, int domain)
{
    (void)ctl;
    (void)domain;
    return PAPI_OK;
}

static int gaudi2_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
    (void)ctx;
    (void)code;
    (void)option;
    return PAPI_OK;
}

/* PAPI vector table */
papi_vector_t _gaudi2_vector = {
    .cmp_info = {
        .name = "gaudi2",
        .short_name = "gaudi2",
        .version = "1.0",
        .description = "Intel Gaudi2 AI Accelerator hardware counters",
        .num_mpx_cntrs = GAUDI2_MAX_COUNTERS,
        .num_cntrs = GAUDI2_MAX_SPMU_COUNTERS,
        .default_domain = PAPI_DOM_USER,
        .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
    },

    .size = {
        .context = sizeof(gaudi2_context_t),
        .control_state = sizeof(gaudi2_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },

    .init_component = gaudi2_init_component,
    .init_thread = gaudi2_init_thread,
    .init_control_state = gaudi2_init_control_state,
    .shutdown_component = gaudi2_shutdown_component,
    .shutdown_thread = gaudi2_shutdown_thread,
    .cleanup_eventset = gaudi2_cleanup_eventset,

    .update_control_state = gaudi2_update_control_state,
    .start = gaudi2_start,
    .stop = gaudi2_stop,
    .read = gaudi2_read,
    .reset = gaudi2_reset,

    .ntv_enum_events = gaudi2_ntv_enum_events,
    .ntv_code_to_name = gaudi2_ntv_code_to_name,
    .ntv_name_to_code = gaudi2_ntv_name_to_code,
    .ntv_code_to_descr = gaudi2_ntv_code_to_descr,
    .ntv_code_to_info = gaudi2_ntv_code_to_info,

    .set_domain = gaudi2_set_domain,
    .ctl = gaudi2_ctl,
};
