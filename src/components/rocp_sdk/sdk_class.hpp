#ifndef __VENDOR_PROFILER_V1_H__
#define __VENDOR_PROFILER_V1_H__

#include <stdint.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

#include <dlfcn.h>
#include <cxxabi.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <map>
#include <unordered_map>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <regex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define DLL_SYM_CHECK(name, type)                \
do {                                             \
    char *err;                                   \
    name##_FPTR = (type) dlsym(dllHandle, #name);\
    err = dlerror();                             \
    if(NULL != err) {                            \
        return err;                              \
    }                                            \
} while (0)

#if defined(PAPI_ROCPSDK_DEBUG)
#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string_FPTR(CHECKSTATUS);              \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }
#else
#define ROCPROFILER_CALL(result, msg) {(void)result;}
#endif

#define RPSDK_MODE_CALLBACK_DISPATCH (0)
#define RPSDK_MODE_AGENT_PROFILE     (1)

typedef struct {
    char name[PAPI_MAX_STR_LEN];
    char descr[PAPI_2MAX_STR_LEN];
} ntv_event_t;

typedef struct {
    ntv_event_t *events;
    int num_events;
} ntv_event_table_t;

struct vendord_ctx {
//    int state;
    int *event_ids;
    long long *counters;
    int num_events;
};

typedef struct vendord_ctx *vendorp_ctx_t;

static int init_ctx(int *event_ids, int num_events, vendorp_ctx_t ctx);
static int finalize_ctx(vendorp_ctx_t ctx);

extern unsigned int _rocp_sdk_lock;

#endif
