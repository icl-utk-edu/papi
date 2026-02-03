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

#define GAUDI2_COMPONENT_NAME "gaudi2"
#define GAUDI2_MAX_COUNTERS 32

/* Eventset status flags */
#define GAUDI2_EVENTS_STOPPED   (0x0)
#define GAUDI2_EVENTS_RUNNING   (0x2)

#define HLTHUNK_DEVICE_DONT_CARE 4
#define HLTHUNK_DEVICE_GAUDI2    5

/* hlthunk structures (matches kernel header) */
struct hl_debug_args {
    __u64 input_ptr;
    __u64 output_ptr;
    __u32 input_size;
    __u32 output_size;
    __u32 op;
    __u32 reg_idx;
    __u32 enable;
    __u32 ctx_id;
};

struct hl_debug_params_spmu {
    __u64 event_types[HL_DEBUG_MAX_AUX_VALUES];
    __u32 event_types_num;
    __u32 pmtrc_val;
    __u32 trc_ctrl_host_val;
    __u32 trc_en_host_val;
};

struct hl_debug_params_read_block {
    __u64 cfg_address;
    __u64 user_address;
    __u32 size;
    __u32 flags;
};

/* hlthunk function pointers (loaded via dlopen) */
static void *hlthunk_handle = NULL;

typedef int (*hlthunk_open_fn)(int device_type, const char *busid);
typedef int (*hlthunk_close_fn)(int fd);
typedef int (*hlthunk_debug_fn)(int fd, struct hl_debug_args *debug);

static hlthunk_open_fn  p_hlthunk_open  = NULL;
static hlthunk_close_fn p_hlthunk_close = NULL;
static hlthunk_debug_fn p_hlthunk_debug = NULL;

/* Native event table */
static gaudi2_native_event_t gaudi2_native_events[] = {
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

/* Per-event tracking */
typedef struct {
    unsigned int event_idx;
    unsigned int counter_idx;
    uint64_t spmu_base;
    long long last_value;
    long long accumulated;
} gaudi2_counter_t;

/* Per-eventset state */
typedef struct {
    gaudi2_counter_t counters[GAUDI2_MAX_COUNTERS];
    int num_counters;
    long long values[GAUDI2_MAX_COUNTERS];
    int running;
} gaudi2_control_t;

/* Per-thread context */
typedef struct {
    int device_fd;
    int debug_mode_enabled;
    int spmu_enabled;
} gaudi2_context_t;

static int gaudi2_device_fd = -1;
static unsigned int gaudi2_lock;

papi_vector_t _gaudi2_vector;

/* Load hlthunk library, checking PAPI_GAUDI2_ROOT first */
static int load_hlthunk_library(void)
{
    char root_lib_path[PAPI_MAX_STR_LEN];
    const char *gaudi2_root;
    int strLen;

    gaudi2_root = getenv("PAPI_GAUDI2_ROOT");

    /* Try PAPI_GAUDI2_ROOT first if set */
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

    /* Fallback paths */
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

    if (!p_hlthunk_open || !p_hlthunk_close || !p_hlthunk_debug) {
        SUBDBG("Failed to find required hlthunk symbols\n");
        dlclose(hlthunk_handle);
        hlthunk_handle = NULL;
        return PAPI_ENOSUPP;
    }

    return PAPI_OK;
}

/*
 * Find existing Gaudi2 device fd from /proc/self/fd.
 * When PyTorch or another framework has already opened the device,
 * we reuse that fd. Supports multiple devices by returning the first found.
 *
 * TODO: For multi-device support, extend to return an array of fds
 * (similar to cuda/rocp_sdk components).
 */
static int find_gaudi2_device_fd(void)
{
    DIR *dir;
    struct dirent *entry;
    char link_path[PAPI_MIN_STR_LEN];
    char target[PAPI_MAX_STR_LEN];
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

        /* Look for /dev/accel/accel* (not control device) */
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

/* Read SPMU counters via READBLOCK. */
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

static int gaudi2_init_component(int cidx)
{
    int papi_errno = PAPI_OK;
    int num_events;

    SUBDBG("Initializing Gaudi2 component (cidx=%d)\n", cidx);

    _gaudi2_vector.cmp_info.CmpIdx = cidx;

    papi_errno = load_hlthunk_library();
    if (papi_errno != PAPI_OK) {
        snprintf(_gaudi2_vector.cmp_info.disabled_reason,
                 PAPI_MAX_STR_LEN, "Failed to load libhl-thunk.so");
        _gaudi2_vector.cmp_info.disabled = papi_errno;
        return papi_errno;
    }

    /* Try to find existing fd from PyTorch, else open ourselves */
    gaudi2_device_fd = find_gaudi2_device_fd();
    if (gaudi2_device_fd < 0)
        gaudi2_device_fd = p_hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);

    if (gaudi2_device_fd < 0) {
        papi_errno = PAPI_ENOSUPP;
        snprintf(_gaudi2_vector.cmp_info.disabled_reason,
                 PAPI_MAX_STR_LEN, "No Gaudi2 device found");
        _gaudi2_vector.cmp_info.disabled = papi_errno;
        return papi_errno;
    }

    num_events = 0;
    while (gaudi2_native_events[num_events].name != NULL)
        num_events++;

    _gaudi2_vector.cmp_info.num_native_events = num_events;
    _gaudi2_vector.cmp_info.num_cntrs = GAUDI2_MAX_SPMU_COUNTERS;
    _gaudi2_vector.cmp_info.num_mpx_cntrs = GAUDI2_MAX_COUNTERS;

    gaudi2_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;
    _gaudi2_vector.cmp_info.initialized = 1;

    return PAPI_OK;
}

static int gaudi2_shutdown_component(void)
{
    gaudi2_device_fd = -1;

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
    gaudi2_ctx->device_fd = gaudi2_device_fd;
    return PAPI_OK;
}

static int gaudi2_shutdown_thread(hwd_context_t *ctx)
{
    gaudi2_context_t *gaudi2_ctx = (gaudi2_context_t *)ctx;

    if (gaudi2_ctx->spmu_enabled)
        disable_spmu(gaudi2_ctx->device_fd, 0);

    if (gaudi2_ctx->debug_mode_enabled)
        disable_debug_mode(gaudi2_ctx->device_fd);

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
    int num_events = _gaudi2_vector.cmp_info.num_native_events;
    (void)ctx;

    if (count > GAUDI2_MAX_COUNTERS) {
        SUBDBG("Event count %d exceeds max %d\n", count, GAUDI2_MAX_COUNTERS);
        return PAPI_ECOUNT;
    }

    gaudi2_ctl->num_counters = count;

    for (int i = 0; i < count; i++) {
        int event_idx = native[i].ni_event;

        if (event_idx < 0 || event_idx >= num_events) {
            SUBDBG("Invalid event index %d (max %d)\n", event_idx, num_events);
            return PAPI_EINVAL;
        }

        gaudi2_native_event_t *event = &gaudi2_native_events[event_idx];

        gaudi2_ctl->counters[i].event_idx = event_idx;
        gaudi2_ctl->counters[i].counter_idx = i % GAUDI2_MAX_SPMU_COUNTERS;
        gaudi2_ctl->counters[i].spmu_base = get_spmu_base_address(event->engine, 0, 0);
        gaudi2_ctl->counters[i].last_value = 0;
        gaudi2_ctl->counters[i].accumulated = 0;

        native[i].ni_position = i;
    }

    return PAPI_OK;
}

static int gaudi2_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    gaudi2_context_t *gaudi2_ctx = (gaudi2_context_t *)ctx;
    gaudi2_control_t *gaudi2_ctl = (gaudi2_control_t *)ctl;
    uint64_t events[HL_DEBUG_MAX_AUX_VALUES];
    int papi_errno;

    if (!gaudi2_ctx->debug_mode_enabled) {
        papi_errno = enable_debug_mode(gaudi2_ctx->device_fd);
        if (papi_errno != PAPI_OK)
            return papi_errno;
        gaudi2_ctx->debug_mode_enabled = 1;
    }

    int num_events = gaudi2_ctl->num_counters;
    if (num_events > HL_DEBUG_MAX_AUX_VALUES)
        num_events = HL_DEBUG_MAX_AUX_VALUES;

    for (int i = 0; i < num_events; i++) {
        int event_idx = gaudi2_ctl->counters[i].event_idx;
        events[i] = gaudi2_native_events[event_idx].event_id;
    }

    papi_errno = enable_spmu(gaudi2_ctx->device_fd, 0, events, num_events);
    if (papi_errno != PAPI_OK)
        return papi_errno;
    gaudi2_ctx->spmu_enabled = 1;

    for (int i = 0; i < gaudi2_ctl->num_counters; i++) {
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

    if (gaudi2_ctl->running == GAUDI2_EVENTS_RUNNING) {
        long long temp_values[GAUDI2_MAX_SPMU_COUNTERS];
        uint64_t base = gaudi2_ctl->counters[0].spmu_base;

        if (read_spmu_counters(gaudi2_ctx->device_fd, base,
                               gaudi2_ctl->num_counters, temp_values) == PAPI_OK) {
            for (int i = 0; i < gaudi2_ctl->num_counters; i++) {
                gaudi2_ctl->counters[i].accumulated += temp_values[i];
                gaudi2_ctl->values[i] = gaudi2_ctl->counters[i].accumulated;
            }
        }
    }

    if (gaudi2_ctx->spmu_enabled) {
        disable_spmu(gaudi2_ctx->device_fd, 0);
        gaudi2_ctx->spmu_enabled = 0;
    }

    gaudi2_ctl->running = GAUDI2_EVENTS_STOPPED;
    return PAPI_OK;
}

static int gaudi2_read(hwd_context_t *ctx, hwd_control_state_t *ctl,
                       long long **events, int flags)
{
    gaudi2_context_t *gaudi2_ctx = (gaudi2_context_t *)ctx;
    gaudi2_control_t *gaudi2_ctl = (gaudi2_control_t *)ctl;
    (void)flags;

    if (gaudi2_ctl->running == GAUDI2_EVENTS_RUNNING) {
        long long temp_values[GAUDI2_MAX_SPMU_COUNTERS];
        uint64_t base = gaudi2_ctl->counters[0].spmu_base;

        if (read_spmu_counters(gaudi2_ctx->device_fd, base,
                               gaudi2_ctl->num_counters, temp_values) == PAPI_OK) {
            for (int i = 0; i < gaudi2_ctl->num_counters; i++)
                gaudi2_ctl->values[i] = gaudi2_ctl->counters[i].accumulated + temp_values[i];
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
    int num_events = _gaudi2_vector.cmp_info.num_native_events;

    switch (modifier) {
        case PAPI_ENUM_FIRST:
            *EventCode = 0;
            return PAPI_OK;
        case PAPI_ENUM_EVENTS:
            if (*EventCode + 1 < (unsigned int)num_events) {
                *EventCode = *EventCode + 1;
                return PAPI_OK;
            }
            return PAPI_ENOEVNT;
        default:
            return PAPI_EINVAL;
    }
}

static int gaudi2_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    int num_events = _gaudi2_vector.cmp_info.num_native_events;

    if (EventCode >= (unsigned int)num_events) {
        SUBDBG("EventCode %u out of range (max %d)\n", EventCode, num_events);
        return PAPI_ENOEVNT;
    }

    snprintf(name, len, "%s", gaudi2_native_events[EventCode].name);
    return PAPI_OK;
}

/*
 * NOTE: Linear scan is acceptable for the current event count (~47).
 * If the event list grows significantly, consider using a hash table
 * for O(1) lookup (similar to rocp_sdk component).
 */
static int gaudi2_ntv_name_to_code(const char *name, unsigned int *EventCode)
{
    int num_events = _gaudi2_vector.cmp_info.num_native_events;

    for (int i = 0; i < num_events; i++) {
        if (strcmp(name, gaudi2_native_events[i].name) == 0) {
            *EventCode = i;
            return PAPI_OK;
        }
    }
    return PAPI_ENOEVNT;
}

static int gaudi2_ntv_code_to_descr(unsigned int EventCode, char *descr, int len)
{
    int num_events = _gaudi2_vector.cmp_info.num_native_events;

    if (EventCode >= (unsigned int)num_events) {
        SUBDBG("EventCode %u out of range (max %d)\n", EventCode, num_events);
        return PAPI_ENOEVNT;
    }

    snprintf(descr, len, "%s", gaudi2_native_events[EventCode].description);
    return PAPI_OK;
}

static int gaudi2_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info)
{
    int num_events = _gaudi2_vector.cmp_info.num_native_events;

    if (EventCode >= (unsigned int)num_events) {
        SUBDBG("EventCode %u out of range (max %d)\n", EventCode, num_events);
        return PAPI_ENOEVNT;
    }

    gaudi2_native_event_t *event = &gaudi2_native_events[EventCode];

    snprintf(info->symbol, sizeof(info->symbol), "%s", event->name);
    snprintf(info->long_descr, sizeof(info->long_descr), "%s", event->description);
    snprintf(info->short_descr, sizeof(info->short_descr), "%s", event->description);

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
