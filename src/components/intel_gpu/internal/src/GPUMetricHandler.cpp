/*
 * GPUMetricHandler.cpp:   Intel® Graphics Component for PAPI
 *
 * Copyright (c) 2020 Intel Corp. All rights reserved
 * Contributed by Peinan Zhang  <peinan.zhang@intel.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstdarg>
#include <stdexcept>
#include <semaphore.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include <vector>

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include <level_zero/ze_ddi.h>
#include <level_zero/zet_ddi.h>

#include "GPUMetricHandler.h"

#define  DebugPrintError(format, args...)     fprintf(stderr, format, ## args)

#if defined(_DEBUG)
#define DebugPrint(format, args...)           fprintf(stderr, format, ## args)
#else 
#define DebugPrint(format, args...)           {do {} while(0); }
#endif

#define METRIC_BITS          8
#define METRIC_GROUP_MASK    0xff00
#define METRIC_MASK          0x00ff

#define MAX_REPORTS          32768
#define MAX_KERNERLS         1024

#define GROUP_CODE_BY_ID(mGroupId)               (((mGroupId+1)<<METRIC_BITS) & METRIC_GROUP_MASK)
#define METRIC_CODE_BY_ID(mGroupId, mId)         (GROUP_CODE_BY_ID(mGroupId) | ((mId+1)&METRIC_MASK))

#define CHECK_N_RETURN_STATUS(status, retVal)    {if (status) return retVal; }
#define CHECK_N_RETURN(status)                   {if (status) return;}

struct InstanceData {
  uint32_t kernelId;
  zet_metric_query_handle_t metricQuery;
};

QueryState   *gQueryState = nullptr;


/* this is a temp solution
 * It is better to keep this in a metric database file
 */
static char avgMetricList[][32]={
    "The percentage",
    "Average",
    "The average",
    "The ratio",
    "Percentage"
};

static char staticMetricList[][32] = {
     "QueryBeginTime",
     "CoreFrequencyMHz", 
     "EuSliceFrequencyMHz",
     "ReportReason", 
     "ContextId"
};

#define M_OP_AGGREGATE   0x0
#define M_OP_AVERAGE     0x1
#define M_OP_STATIC      0x2

/************************************************************************/
/* Helper function                                                      */
/************************************************************************/

void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

ze_pfnInit_t                                zeInitFunc;
ze_pfnDriverGet_t                           zeDriverGetFunc;
ze_pfnDriverGetProperties_t                 zeDriverGetPropertiesFunc;
ze_pfnDeviceGet_t                           zeDeviceGetFunc;
ze_pfnDeviceGetProperties_t                 zeDeviceGetPropertiesFunc;
ze_pfnEventPoolCreate_t                     zeEventPoolCreateFunc;
ze_pfnEventPoolDestroy_t                    zeEventPoolDestroyFunc;
ze_pfnContextCreate_t                       zeContextCreateFunc;
ze_pfnEventCreate_t                         zeEventCreateFunc;
ze_pfnEventDestroy_t                        zeEventDestroyFunc;
ze_pfnEventHostSynchronize_t                zeEventHostSynchronizeFunc;

zet_pfnMetricGroupGet_t                     zetMetricGroupGetFunc;
zet_pfnMetricGroupGetProperties_t           zetMetricGroupGetPropertiesFunc;
zet_pfnMetricGroupCalculateMetricValues_t   zetMetricGroupCalculateMetricValuesFunc;
zet_pfnMetricGet_t                          zetMetricGetFunc;
zet_pfnMetricGetProperties_t                zetMetricGetPropertiesFunc;
zet_pfnContextActivateMetricGroups_t        zetContextActivateMetricGroupsFunc;
zet_pfnMetricStreamerOpen_t                 zetMetricStreamerOpenFunc;
zet_pfnMetricStreamerClose_t                zetMetricStreamerCloseFunc;
zet_pfnMetricStreamerReadData_t             zetMetricStreamerReadDataFunc;
zet_pfnMetricQueryPoolCreate_t              zetMetricQueryPoolCreateFunc;
zet_pfnMetricQueryPoolDestroy_t             zetMetricQueryPoolDestroyFunc;
zet_pfnMetricQueryCreate_t                  zetMetricQueryCreateFunc;
zet_pfnMetricQueryDestroy_t                 zetMetricQueryDestroyFunc;
zet_pfnMetricQueryReset_t                   zetMetricQueryResetFunc;
zet_pfnMetricQueryGetData_t                 zetMetricQueryGetDataFunc;
zet_pfnTracerExpCreate_t                    zetTracerExpCreateFunc;
zet_pfnTracerExpDestroy_t                   zetTracerExpDestroyFunc;
zet_pfnTracerExpSetPrologues_t              zetTracerExpSetProloguesFunc;
zet_pfnTracerExpSetEpilogues_t              zetTracerExpSetEpiloguesFunc;
zet_pfnTracerExpSetEnabled_t                zetTracerExpSetEnabledFunc;
zet_pfnCommandListAppendMetricQueryBegin_t  zetCommandListAppendMetricQueryBeginFunc;
zet_pfnCommandListAppendMetricQueryEnd_t    zetCommandListAppendMetricQueryEndFunc;

#define DLL_SYM_CHECK(handle, name, type)                  \
    do {                                                   \
        name##Func = (type) dlsym(handle, #name);          \
        if (dlerror() != nullptr) {                        \
            DebugPrintError("failed: %s\n", #name);        \
           return 1;                                       \
        }                                                  \
   } while (0)

static int
functionInit(void *dllHandle)
{
    int ret  = 0;
    DLL_SYM_CHECK(dllHandle, zeInit, ze_pfnInit_t);
    DLL_SYM_CHECK(dllHandle, zeDriverGet, ze_pfnDriverGet_t);
    DLL_SYM_CHECK(dllHandle, zeDriverGetProperties, ze_pfnDriverGetProperties_t);
    DLL_SYM_CHECK(dllHandle, zeDeviceGet, ze_pfnDeviceGet_t);
    DLL_SYM_CHECK(dllHandle, zeDeviceGetProperties, ze_pfnDeviceGetProperties_t);
    DLL_SYM_CHECK(dllHandle, zetMetricGroupGet, zet_pfnMetricGroupGet_t);
    DLL_SYM_CHECK(dllHandle, zetMetricGroupGetProperties, zet_pfnMetricGroupGetProperties_t);
    DLL_SYM_CHECK(dllHandle, zetMetricGet, zet_pfnMetricGet_t);
    DLL_SYM_CHECK(dllHandle, zetMetricGetProperties, zet_pfnMetricGetProperties_t);
    DLL_SYM_CHECK(dllHandle, zetContextActivateMetricGroups, zet_pfnContextActivateMetricGroups_t);
    DLL_SYM_CHECK(dllHandle, zeEventPoolCreate, ze_pfnEventPoolCreate_t);
    DLL_SYM_CHECK(dllHandle, zeEventPoolDestroy, ze_pfnEventPoolDestroy_t);
    DLL_SYM_CHECK(dllHandle, zeContextCreate, ze_pfnContextCreate_t);
    DLL_SYM_CHECK(dllHandle, zeEventCreate, ze_pfnEventCreate_t);
    DLL_SYM_CHECK(dllHandle, zeEventDestroy, ze_pfnEventDestroy_t);
    DLL_SYM_CHECK(dllHandle, zeEventHostSynchronize, ze_pfnEventHostSynchronize_t);
    DLL_SYM_CHECK(dllHandle, zetMetricStreamerOpen, zet_pfnMetricStreamerOpen_t);
    DLL_SYM_CHECK(dllHandle, zetMetricStreamerClose, zet_pfnMetricStreamerClose_t);
    DLL_SYM_CHECK(dllHandle, zetMetricStreamerReadData, zet_pfnMetricStreamerReadData_t);
    DLL_SYM_CHECK(dllHandle, zetMetricQueryPoolCreate, zet_pfnMetricQueryPoolCreate_t);
    DLL_SYM_CHECK(dllHandle, zetMetricQueryPoolDestroy, zet_pfnMetricQueryPoolDestroy_t);
    DLL_SYM_CHECK(dllHandle, zetMetricQueryCreate, zet_pfnMetricQueryCreate_t);
    DLL_SYM_CHECK(dllHandle, zetMetricQueryDestroy, zet_pfnMetricQueryDestroy_t);
    DLL_SYM_CHECK(dllHandle, zetMetricQueryGetData, zet_pfnMetricQueryGetData_t);
    DLL_SYM_CHECK(dllHandle, zetMetricGroupCalculateMetricValues, zet_pfnMetricGroupCalculateMetricValues_t);
    DLL_SYM_CHECK(dllHandle, zetTracerExpCreate, zet_pfnTracerExpCreate_t);
    DLL_SYM_CHECK(dllHandle, zetTracerExpDestroy, zet_pfnTracerExpDestroy_t);
    DLL_SYM_CHECK(dllHandle, zetTracerExpSetPrologues, zet_pfnTracerExpSetPrologues_t);
    DLL_SYM_CHECK(dllHandle, zetTracerExpSetEpilogues, zet_pfnTracerExpSetEpilogues_t);
    DLL_SYM_CHECK(dllHandle, zetTracerExpSetEnabled, zet_pfnTracerExpSetEnabled_t);
    DLL_SYM_CHECK(dllHandle, zetCommandListAppendMetricQueryBegin, zet_pfnCommandListAppendMetricQueryBegin_t);
    DLL_SYM_CHECK(dllHandle, zetCommandListAppendMetricQueryEnd, zet_pfnCommandListAppendMetricQueryEnd_t);
    return ret;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn    static void kernelCreateCB(
 *                     ze_command_list_append_launch_kernel_params_t* params,
 *                     ze_result_t result, void* globalData, void** instanceData)
 *
 * @brief callback function called to add kernel into name mapping.
 *        It is called when exiting kerenl creation.
 *
 * @param IN    params        -- kerne parametes
 * @param IN    result        -- kernel launch result
 * @param INOUT globalData    -- pointer to the location which stores global data
 * @param INOUT instanceData  -- pointer to the location which stores the data for this query
 *
 * @return
 */
static void kernelCreateCB(ze_kernel_create_params_t *params,
                ze_result_t result, void *globalData, void **instanceData)
{
    (void)globalData;
    (void)instanceData;
    if (result != ZE_RESULT_SUCCESS) {
        return;
    }
    gQueryState->lock.lock();
    ze_kernel_handle_t kernel = **(params->pphKernel);
    gQueryState->nameMap[kernel] = (*(params->pdesc))->pKernelName;
    gQueryState->lock.unlock();
    return;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn    static void kernelDestroyCB(
 *                     ze_command_list_append_launch_kernel_params_t* params,
 *                     ze_result_t result, void* globalData, void** instanceData)
 *
 * @brief callback function to remove kernel from maintained mapping
 *        It is called when exiting kerenl destroy.
 *
 * @param IN    params        -- kerne parametes
 * @param IN    result        -- kernel launch result
 * @param INOUT globalData    -- pointer to the location which stores global data
 * @param INOUT instanceData  -- pointer to the location which stores the data for this query
 *
 * @return
 */
static void
kernelDestroyCB(ze_kernel_destroy_params_t *params,
        ze_result_t result, void *globalData, void **instanceData)
{
    (void)globalData;
    (void)instanceData;
    if (result != ZE_RESULT_SUCCESS) {
        return;
    }
    gQueryState->lock.lock();
    ze_kernel_handle_t kernel = *(params->phKernel);
    if (gQueryState->nameMap.count(kernel) != 1) {
        gQueryState->lock.unlock();
        return;
    }
    gQueryState->nameMap.erase(kernel);
    gQueryState->lock.unlock();
    return;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn    static void metricQueryBeginCB(
 *                     ze_command_list_append_launch_kernel_params_t* params,
 *                     ze_result_t result, void* globalData, void** instanceData)
 *
 * @brief callback function to append "query begin" request into command list before kernel launches.
 *
 * @param IN    params        -- kerne parametes
 * @param IN    result        -- kernel launch result
 * @param INOUT globalData    -- pointer to the location which stores global data
 * @param OUT   instanceData  -- pointer to the location which stores the data for this query
 *
 * @return
 *
 */
static void
metricQueryBeginCB(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result, void* globalData, void** instanceData)
{
    (void)result;
    (void)globalData;
    (void)instanceData;

    // assign each kernel an id for reference.
    uint32_t kernId = gQueryState->kernelId.fetch_add(1, std::memory_order_acq_rel);
    if (kernId >=  MAX_KERNERLS) {
        *instanceData = nullptr;
        return;
    }

    ze_result_t status = ZE_RESULT_SUCCESS;
    zet_metric_query_handle_t metricQuery = nullptr;
    status = zetMetricQueryCreateFunc(gQueryState->queryPool, kernId, &metricQuery);
    CHECK_N_RETURN(status!=ZE_RESULT_SUCCESS);
    ze_command_list_handle_t commandList = *(params->phCommandList);
    CHECK_N_RETURN(commandList== nullptr);
    status = zetCommandListAppendMetricQueryBeginFunc(commandList, metricQuery);
    CHECK_N_RETURN(status!=ZE_RESULT_SUCCESS);

    // maintain the query for current kernel
    InstanceData* data = new InstanceData;
    CHECK_N_RETURN(data==nullptr);
    data->kernelId = kernId;
    data->metricQuery = metricQuery;
    *instanceData = reinterpret_cast<void*>(data);

    return;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn    static void metricQueryEndCB(
 *                    ze_command_list_append_launch_kernel_params_t* params,
 *                    ze_result_t result, void* globalData, void** instanceData)
 *
 * @brief callback function to append "query end" request into command list after kernel completes
 *
 * @param IN    params        -- kerne parametes
 * @param IN    result        -- kernel launch result
 * @param INOUT globalData    -- pointer to the location which stores global data
 * @param INOUT instanceData  -- pointer to the location which stores the data for this query
 *
 */
static void
metricQueryEndCB(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result, void* globalData, void** instanceData)
{
    (void)result;
    (void)globalData;

    InstanceData* data = reinterpret_cast<InstanceData*>(*instanceData);
    CHECK_N_RETURN(data==nullptr);
    ze_command_list_handle_t commandList = *(params->phCommandList);
    CHECK_N_RETURN(commandList==nullptr);

    ze_event_desc_t eventDesc;
    eventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    eventDesc.index = data->kernelId;
    eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

    ze_event_handle_t event = nullptr;
    ze_result_t status      = ZE_RESULT_SUCCESS;
    status = zeEventCreateFunc(gQueryState->eventPool, &eventDesc, &event);
    CHECK_N_RETURN(status!=ZE_RESULT_SUCCESS);

    status = zetCommandListAppendMetricQueryEndFunc(commandList, data->metricQuery, event,
                                            0, nullptr);
    CHECK_N_RETURN(status!=ZE_RESULT_SUCCESS);

    gQueryState->lock.lock();
    ze_kernel_handle_t kernel = *(params->phKernel);
    if (gQueryState->nameMap.count(kernel) == 1) {
        QueryData queryData = { gQueryState->nameMap[kernel],
                               data->metricQuery, event };
        gQueryState->queryList.push_back(queryData);
    }
    gQueryState->lock.unlock();
    return;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn       int getMetricOp(const char *name, const char *desc)
 *
 * @brief    Find the metric report summary type (aggregate, average) based on name and description. 
 *
 * @param    IN name  -- metric name
 * @param    IN desc  -- metric description
 *
 * @ return  operation tyep, <0: aggregate, 1:averge, 2: static>
 */
static int
getMetricOp(const char *name, const char *desc) {

    int num = sizeof(staticMetricList)/32;
    for (int i=0; i<num; i++) {
        if (strncmp(staticMetricList[i], name, strlen(staticMetricList[i])) == 0) {
            return M_OP_STATIC;
        }
    }

    num = sizeof(avgMetricList)/32;
    for (int i=0; i<num; i++) {
        if (strncmp(avgMetricList[i], desc, strlen(avgMetricList[i])) == 0) {
            return M_OP_AVERAGE;
        }
    }
    return M_OP_AGGREGATE;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn         void * typedValue2Value(zet_typed_value_t data, std::fstream *foutstream, 
 *                                     int outflag, int *dtype, long long *iVal,  
 *                                     double *fpVal, int isLast)
 *
 *  @brief    Convert typed value to long long or double.  if foutstream is open, log the data
 *
 *  @param    In data         -- typed data
 *  @param    IN foutstream   -- output stream to dump raw data  (used for debug)
 *  @param    IN outflag      -- 1 for output to stdout. 0 for disable output to stdout
 *  @param    OUT dtype       -- data type,  0: long long,  1: double
 *  @param    OUT ival        -- converted long long integer value
 *  @param    OUT dval        -- converted double value
 *  @param    IN isLast        -- last data
 *
 */
static void
typedValue2Value(zet_typed_value_t data, std::fstream *foutstream, int outflag, int *dtype, long long *iVal,  double *fpVal, int isLast)
{
     *dtype = 0;
#if defined(_DEBUG)
     outflag = 1;
#endif
    switch( data.type )
    {
        case ZET_VALUE_TYPE_UINT32:
        case ZET_VALUE_TYPE_BOOL8:
             if (foutstream->is_open()) {
                *foutstream << data.value.ui32 << ",";
             }
             if (outflag) {
                cout << data.value.ui32 << ",";
             }
             *iVal = (long long)data.value.ui32;
             break;
        case ZET_VALUE_TYPE_UINT64:
             if (foutstream->is_open()) {
                *foutstream << data.value.ui64 << ",";
             }
             if (outflag) {
                cout << data.value.ui64 << ",";
             }
             *iVal = (long long)data.value.ui64;
             break;
        case ZET_VALUE_TYPE_FLOAT32:
             if (foutstream->is_open()) {
                *foutstream << data.value.fp32 << ",";
             }
             if (outflag) {
                cout << data.value.fp32 << ",";
             }
             *dtype = 1;
             *fpVal = (double)data.value.fp32;
             break;
        case ZET_VALUE_TYPE_FLOAT64:
             if (foutstream->is_open()) {
                *foutstream << data.value.fp32 << ",";
             }
             if (outflag) {
                cout << data.value.fp64 << ",";
             }
             *dtype = 1;
             *fpVal = (double)data.value.fp64;
             break;
        default:
             break;
    }
    if (isLast) {
        if (foutstream->is_open()) {
            *foutstream << endl;
        }
        if (outflag) {
            cout << endl;
        }
    }

}



/***************************************************************************/
/* Handler class                                                           */
/***************************************************************************/

using namespace std;

GPUMetricHandler* GPUMetricHandler::m_handlerInstance = nullptr;

/************************************************************************/
/* GPUMetricHandler constructor                                         */
/************************************************************************/
GPUMetricHandler::GPUMetricHandler()
{
    m_driver              = nullptr;
    m_device              = nullptr;
    m_context             = nullptr;
    m_domainId            = 0;
    m_groupList           = nullptr;
    m_numGroups           = 0;
    m_numMetrics          = 0;
    m_maxMetricsPerGroup  = 0;
    m_groupId             = -1;
    m_metricsSelected     = nullptr;
    m_dataDumpFileName    = "";
    m_dataDumpFilePath    = "";
    m_status              = COLLECTION_IDLE;
    m_stdout              = 0;
    m_reportCount         = 0;
    m_reportData          = nullptr;
    m_eventPool           = nullptr;
    m_event               = nullptr;
    m_metricStreamer      = nullptr;
    m_tracer              = nullptr;
    m_queryPool           = nullptr;
    m_groupType           = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;
}

/************************************************************************/
/* ~GPUMetricHandler                                                    */
/************************************************************************/
GPUMetricHandler::~GPUMetricHandler()
{
    DestroyMetricDevice();
    m_handlerInstance = nullptr;
}


/************************************************************************/
/* GetInstance                                                    */
/************************************************************************/
GPUMetricHandler *GPUMetricHandler::GetInstance()
{
    if( m_handlerInstance == nullptr )
    {
        m_handlerInstance = new GPUMetricHandler();
    }
    return (m_handlerInstance);
}

/*------------------------------------------------------------------------------*/
/*!
 * fn      int GPUMetricHandler::InitMetricDevice(int *numDevices)
 * 
 * @brief  Discover and initiate GPU Metric Device
 *
 * @param  OUT numDevice -- number of devices initiated
 *
 * @return Status.  0 for success, otherwise 1. 
 */
int GPUMetricHandler::InitMetricDevice(int *numDevices)
{
    uint32_t driver_count = 0;
    int ret       = 0;
    int retError  = 1;

    if (m_device) {
        return retError;  // device in use
    }

    void *dlHandle = dlopen("libze_loader.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dlHandle) {
        DebugPrintError("dlopen libze_loader.so failed\n");
        return retError;
    }

    *numDevices  = 0;
    ret = functionInit(dlHandle);

    if (ret) {
        DebugPrintError("Failed in finding functions in libze_loader.so\n");
        return ret;
    }

    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zeInitFunc(ZE_INIT_FLAG_GPU_ONLY);
    CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

    status = zeDriverGetFunc(&driver_count, nullptr);
    if (status != ZE_RESULT_SUCCESS || driver_count == 0) {
        return retError;
    }
    vector<ze_driver_handle_t> driver_list(driver_count, nullptr);

    status = zeDriverGetFunc(&driver_count, driver_list.data());
    CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

    for (uint32_t i = 0; i < driver_count; ++i) {
        uint32_t device_count = 0;
        status = zeDeviceGetFunc(driver_list[i], &device_count, nullptr);
        if (status != ZE_RESULT_SUCCESS || device_count == 0) {
            continue;
        }
        vector<ze_device_handle_t> device_list(device_count, nullptr);
        status = zeDeviceGetFunc(driver_list[i], &device_count, device_list.data());
        CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

        for (uint32_t j = 0; j < device_count; ++j) {
            ze_device_properties_t props;
            status = zeDeviceGetPropertiesFunc(device_list[j], &props);
            CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);
            if ((props.type == ZE_DEVICE_TYPE_GPU) && strstr(props.name, "Intel") != nullptr) {
                m_device = device_list.at(j);
                m_driver = driver_list.at(i);
                break;
            }
        }
    }

    *numDevices = 1;
    ret =  InitMetricGroups(m_device);
    if (!ret) {
        DebugPrint("m_maxMetricsPerGroup %d, m_numGropus %d, total_metrics %d\n",  m_maxMetricsPerGroup, m_numGroups, m_numMetrics);
    }
    return ret;

}

/*------------------------------------------------------------------------------*/
/*!
 * @fn        int GPUMetricHandler::InitMetricGroups(ze_device_handle_t device)
 *
 * @brief     Initiate Metric Group
 *
 * @param     IN device -- device handle for the metric gropu
 *
 * @return    Status.  0 for success, otherwise 1. 
 */
int GPUMetricHandler::InitMetricGroups(ze_device_handle_t device)
{

    int ret         = 0;
    int retError    = 1;
    if (!device) {
        return retError;
    }
    ze_result_t status = ZE_RESULT_SUCCESS;

#if defined(_DEBUG)
    std::cout << "Target device: " <<
                 GetDeviceName(device) << std::endl;
#endif

    uint32_t group_count = 0;
    status = zetMetricGroupGetFunc(device, &group_count, nullptr);
    if (status != ZE_RESULT_SUCCESS || group_count == 0) {
        std::cout << "[WARNING] No metrics found" << std::endl;
        return retError;
    }

    m_groupList = new TMetricGroupNode[group_count];
    CHECK_N_RETURN_STATUS((m_groupList==nullptr), 1);
    m_numGroups = group_count;
    zet_metric_group_handle_t* group_list = new zet_metric_group_handle_t[group_count];
    CHECK_N_RETURN_STATUS((group_list==nullptr), 1);
    status = zetMetricGroupGetFunc(device, &group_count, group_list);
    CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

    m_numMetrics = 0;
    int eventBasedCount = 0;
    int timeBasedCount = 0;

    for (uint32_t gid = 0; gid < group_count; ++gid) {
        zet_metric_group_properties_t group_props = {};
        status = zetMetricGroupGetPropertiesFunc(group_list[gid], &group_props);
        CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

        uint32_t metric_count = group_props.metricCount;
        if (metric_count > m_maxMetricsPerGroup) {
            m_maxMetricsPerGroup = metric_count;
        }
        m_groupList[gid].code = GROUP_CODE_BY_ID(gid);
        m_groupList[gid].props = group_props;
        m_groupList[gid].handle = group_list[gid];
        m_groupList[gid].metricList = new TMetricNode[metric_count];
        CHECK_N_RETURN_STATUS(( m_groupList[gid].metricList ==nullptr), 1);

        DebugPrint("group[%d]: name %s, desc %s\n", gid, group_props.name, group_props.description);

        zet_metric_handle_t*  metric_list = new zet_metric_handle_t[metric_count];
        CHECK_N_RETURN_STATUS((metric_list ==nullptr), 1);
        status = zetMetricGetFunc(group_list[gid], &metric_count, metric_list);
        CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

        for (uint32_t mid = 0; mid < metric_count; ++mid) {
            zet_metric_properties_t metric_props = {};
            status = zetMetricGetPropertiesFunc(metric_list[mid], &metric_props);
            CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);
            m_groupList[gid].metricList[mid].props = metric_props;
            m_groupList[gid].metricList[mid].handle = metric_list[mid];
            m_groupList[gid].metricList[mid].code = METRIC_CODE_BY_ID(gid,mid);
            m_groupList[gid].metricList[mid].metricGroupId = gid;
            m_groupList[gid].metricList[mid].metricId = mid;
            m_groupList[gid].metricList[mid].summaryOp = getMetricOp(metric_props.name, metric_props.description);
            DebugPrint("   metric[%d][%d] name %s, desc %s\n", gid, mid, metric_props.name, metric_props.description);
        }
        m_numMetrics += metric_count;
        if (m_groupList[gid].props.samplingType & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED) {
            eventBasedCount += metric_count;
        } else {
            timeBasedCount += metric_count;
        }            
        delete [] metric_list;
    }
    DebugPrint("init metric groups return:  group_count %d, metric %d, TBS %d, EBS %d\n", m_numGroups, m_numMetrics, timeBasedCount, eventBasedCount);
    delete [] group_list;
    m_status = COLLECTION_INIT;
    DebugPrint("InitMetric End\n");
    return ret;
}


/*------------------------------------------------------------------------------*/
/*!
 * fn      void GPUMetricHandler::DestroyMetricDevice()
 *
 * @brief  Clean up metric device
 */
void GPUMetricHandler::DestroyMetricDevice()
{
    DebugPrint("DestroyMetricDevice\n");

    m_device = nullptr;

    if (m_groupList) {
        for (uint32_t i=0; i<m_numGroups; i++) {
            if (m_groupList[i].metricList) {
                 delete [] m_groupList[i].metricList;
        }
    }
        delete [] m_groupList;
    }
    if (m_reportData) {
        if (m_reportData->dataEntries) {
            delete m_reportData->dataEntries;
    }
        delete m_reportData;
    }

    if (m_event) {
        zeEventDestroyFunc(m_event);
    }
    if (m_eventPool) {
        zeEventPoolDestroyFunc(m_eventPool);
    }
    if (m_metricStreamer) {
    zetMetricStreamerCloseFunc(m_metricStreamer);
    }
    if (m_tracer) {
        zetTracerExpDestroyFunc(m_tracer);
    }
    if (m_queryPool) {
        zetMetricQueryPoolDestroyFunc(m_queryPool);
    }
    m_driver = nullptr;
    m_device = nullptr;
}

/*------------------------------------------------------------------------------*/
/*!
 * fn       string  GPUMetricHandler::GetDeviceName(ze_device_handle_t device)
 * 
 * @brief   Get metric device name
 *
 * @parapm  In device   Device handler
 *
 * @return  The device name
 */
string  GPUMetricHandler::GetDeviceName(ze_device_handle_t device)
{
    ze_result_t status = ZE_RESULT_SUCCESS;
    ze_device_properties_t props;
    status = zeDeviceGetPropertiesFunc(device, &props);
    if (status == ZE_RESULT_SUCCESS) {
        return props.name;
    }
    return "";
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn     int GPUMetricHandler::GetMetricsInfo(int type, MetricInfo *data)
 * 
 * @brief  Get available metric info
 *
 * @param  IN  type  -- metric group type,  0 for timed-based,  1 for query based
 * @param  OUT data  -- pointer to the MetricInfo data contains a list of metrics
 *
 * @reutrn           -- 0 if success. 
 */
int GPUMetricHandler::GetMetricsInfo(int type, MetricInfo *data)
{
    uint32_t      i        = 0;
    int           ret      = 0;
    int           retError = 1;
    uint32_t      stype    = 0;

    if(!m_device) 
    {
        DebugPrintError("MetricsDevice not opened\n");
        return retError;
    }
    if (!data) {
        DebugPrintError("GetMetricsInfo: invalid out data\n");
        return retError;
    }

    // get all metricGroups
    if (type) {
        stype = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED;
    } else {
        stype = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;
    }

    data->code = 0;
    allocMetricInfo(data->infoEntries,  m_numGroups);
    CHECK_N_RETURN_STATUS((data->infoEntries==nullptr), 1);
    int index = 0;
    for (i = 0; i < m_numGroups; i++ ) {
        if (m_groupList[i].props.samplingType & stype) {
            strncpy_se(data->infoEntries[index].name, MAX_STR_LEN,
               m_groupList[i].props.name,
               strlen(m_groupList[i].props.name));
            strncpy_se(data->infoEntries[index].desc, MAX_STR_LEN,
               m_groupList[i].props.description,
               strlen(m_groupList[i].props.description));
            data->infoEntries[index].numEntries = m_groupList[i].props.metricCount;
            data->infoEntries[index].dataType = m_groupList[i].props.samplingType;
            data->infoEntries[index++].code = m_groupList[i].code;
        }
        data->numEntries = index;
    }
    return ret;
}

/*------------------------------------------------------------------------------*/
/**
 * @fn     int GPUMetricHandler::GetMetricsInfo(const char *name, int type, MetricInfo *data)
 *
 * @brief  Get available metrics in a certain metrics group. 
 *         If metric group is not specified, get all available metrics from all metric groups.
 *
 * @param  IN  name  -- metric group name.  If nullptr or empty, means all metric groups
 * @param  IN  type  -- metric group type,  0 for timed-based,  1 for query based
 * @param  OUT data  -- pointer to the MetricInfo data contains a list of metrics
 *
 * @reutrn           -- 0 if success. 
 */
int GPUMetricHandler::GetMetricsInfo(const char *name, int type, MetricInfo *data)
{
    uint32_t i          = 0;
    uint32_t numMetrics = 0;
    int ret             = 0;
    int retError        = 1;
     uint32_t   stype    = (type)?ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED:ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;

    if(!m_device) 
    {
        DebugPrintError("MetricsDevice not opened\n");
        return retError;
    }
    if (!data) {
        DebugPrintError("GetMetricsInfo: invalid out data\n");
        return retError;
    }

    // get all metricGroups

    numMetrics = m_numMetrics;
    int selectedAll = 0;
    if (!name || !strlen(name)) {
        allocMetricInfo(data->infoEntries, numMetrics);
        CHECK_N_RETURN_STATUS((data->infoEntries==nullptr), 1);
        data->code = 0;
        selectedAll = 1;
    }
    int index = 0;
    TMetricGroupNode *mGroup;
    for (i = 0; i < m_numGroups; i++ ) {
        mGroup = &(m_groupList[i]);
        if (!(mGroup->props.samplingType & stype) || strlen(mGroup->props.name) == 0) {
            continue;
        }
        if (!selectedAll) {
            if (strncmp(name, mGroup->props.name, MAX_STR_LEN) != 0) {
                 continue;
            }
            numMetrics = mGroup->props.metricCount;
            allocMetricInfo(data->infoEntries, numMetrics);
            CHECK_N_RETURN_STATUS((data->infoEntries==nullptr), 1);
            m_groupId = i;
            data->code = mGroup->code;
            index = 0;
        }
        for (uint32_t j=0; j<mGroup->props.metricCount; j++) {
             strncpy_se((char *)(data->infoEntries[index].name), MAX_STR_LEN,
                       mGroup->props.name, strlen(mGroup->props.name));
             size_t mnLen = strlen(data->infoEntries[index].name);
             if ((mnLen+3) < MAX_STR_LEN) {
                data->infoEntries[index].name[mnLen++] = '.';
                strncpy_se((char *)&(data->infoEntries[index].name[mnLen]), (MAX_STR_LEN-mnLen),
                mGroup->metricList[j].props.name,
                strlen(mGroup->metricList[j].props.name));
             }
             strncpy_se(data->infoEntries[index].desc, MAX_STR_LEN,
               mGroup->metricList[j].props.description,
               strlen(mGroup->metricList[j].props.description));
             data->infoEntries[index].code = mGroup->metricList[j].code;
             data->infoEntries[index].numEntries = 0;
             data->infoEntries[index].dataType = mGroup->metricList[j].props.resultType;
             data->infoEntries[index].op = mGroup->metricList[j].summaryOp;
             index++;
         }
         if (!selectedAll) {
             break;
         }
    }
    if ( !selectedAll && (i == m_numGroups)) {
        DebugPrintError( "GetMetricsInfo: metricGroup %s is not found, abort\n", name);
        return retError;
    } 
    data->numEntries = index;
    return ret;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn       int GPUMetricHandler::GetMetricCode(
 *                       const char *mGroupName, const char *metricName, uint32_t mtype,
 *                        uint32_t *mGroupCode,  uint32_t *metricCode)
 * 
 * @ brief   Get metric code and metric group code for a given valid metric group name and metric name.
 *
 * @param    IN  mGroupName  - Metric group name
 * @param    IN  metricName  - metric name in the 
 * @param    IN  mtype       - metric type:  < time_based,  event_based>
 * @param    OUT mGroupCode  - metric group code,  0 if the metric group dose not exist
 * @param    OUT metricCode  - metric code,  0 if the metric dose not exist
 *
 * @return   Status,  0 if success, 1 if no such metric or metric group exist.
 */
int GPUMetricHandler::GetMetricCode(
     const char *mGroupName, const char *metricName, uint32_t mtype,
     uint32_t *mGroupCode,  uint32_t *metricCode)
{
    int ret       = 0;
    int retError  = 1;

    DebugPrint( "GetMetricCode: metricGroup %s, metric %s == ", mGroupName, metricName);

    int metricType = (mtype)?ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED:ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;

    for (uint32_t i=0; i< m_numGroups; i++) {
        if (m_groupList[i].code                                                  &&
            (metricType & m_groupList[i].props.samplingType)                     && 
            (strncmp(mGroupName,  m_groupList[i].props.name, MAX_STR_LEN) == 0 )) {
             *mGroupCode = m_groupList[i].code;
            if (!metricName || (strlen(metricName)==0) || !metricCode) {
                // return metricGroup code only.
                DebugPrint( " mGroupCode 0x%x \n", *mGroupCode);
                return ret;
            }
            for (uint32_t j=0; j< m_groupList[i].props.metricCount; j++) {
                if (strncmp(metricName, m_groupList[i].metricList[j].props.name,MAX_STR_LEN)==0) {
                    *metricCode = m_groupList[i].metricList[j].code;
                    DebugPrint( "  mGroupCode 0x%x , metricCode 0x%x\n", *mGroupCode, *metricCode);
                    return ret;
                }
            }
        }
    }
    return retError;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn      int GPUMetricHandler::EnableMetricGroup(
 *                 const char *metricGroupName, uint32_t *selectedList, uint32_t mtype)
 *
 * @brief   Enable a named metric group for collection
 *
 * @param   IN metricGroupName -- the metric group name
 * @param   IN selectedList    -- selected metrics in the metric groups
 * @param   IN mtype           -- metric group type
 *
 * @return  Status, 0 for success.
 */
int
GPUMetricHandler::EnableMetricGroup(const char *metricGroupName,
                            uint32_t *selectedList, uint32_t mtype)
{

    uint32_t     i          = 0;
    ze_result_t  status     = ZE_RESULT_SUCCESS;
    int          ret        = 0;
    int          retError   = 1;

    if ( m_status == COLLECTION_ENABLED) {
        DebugPrint( "EnableMetricGroup: already in enable status\n");
        return ret;
    }
    if (mtype == EVENT_BASED) {
        m_groupType    = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED;
    } else {
        m_groupType    = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;
    }
    DebugPrint("EnableMetricGroup: name %s, type 0x%x\n", metricGroupName, m_groupType);

    for (i=0; i< m_numGroups; i++) {
        zet_metric_group_properties_t  props = m_groupList[i].props;
        if ((strncmp(metricGroupName, props.name, MAX_STR_LEN) == 0 ) && 
            (props.samplingType & m_groupType)) {
            m_groupId = i;
            break;
        }
    }
    if (i == m_numGroups) {
        DebugPrintError("MetricGroup %s is not found\n", metricGroupName);
        return retError;
    }
    m_metricsSelected = selectedList;

    uint32_t metricCount = m_groupList[m_groupId].props.metricCount;
    if (!metricCount) {
        DebugPrintError("MetricGroup %s dose not have metrics\n", metricGroupName);
        return retError;
    }
    if (!m_context) {
    ze_context_desc_t ctxtDesc = {
            ZE_STRUCTURE_TYPE_CONTEXT_DESC,
            nullptr, 
            0
        }; 
        zeContextCreateFunc(m_driver, &ctxtDesc, &m_context);
    }

    // activate metric group
    zet_metric_group_handle_t mGroup = m_groupList[m_groupId].handle;
     
    status = zetContextActivateMetricGroupsFunc(m_context, m_device, 1, &mGroup);
    if (status != ZE_RESULT_SUCCESS) {
        DebugPrintError("ActivateMetricGroup %s failed.\n", metricGroupName);
        return retError;
    }

    m_eventPool    = nullptr;
    m_event        = nullptr;
    m_tracer       = nullptr;
    // create buffer for report data
    if (!m_reportData) {
        m_reportData = new MetricData[1];
        CHECK_N_RETURN_STATUS((m_reportData==nullptr), 1);
    }
    m_reportCount              = 0;
    m_reportData->grpCode      = m_groupId + 1;
    m_reportData->numEntries   = metricCount;
    m_reportData->dataEntries  = new DataEntry[metricCount];
    CHECK_N_RETURN_STATUS((m_reportData->dataEntries==nullptr), 1);
    for( uint32_t i = 0; i < metricCount; i++ ) {
        m_reportData->dataEntries[i].value.ival = 0;
        m_reportData->dataEntries[i].code = m_groupList[m_groupId].metricList[i].code;
    }
    return ret;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn      int GPUMetricHandler::EnableTimeBasedStream(
 *                     uint32_t timePeriod, uint32_t numReports)
 *
 * @brief   Enable a named metric group for collection
 *
 * @param   IN  timePeriod    - The timer period for sampling
 * @param   OUT numReports    - total number of sample reports (not used).
 *                              Default is to collect all sample record available
 *
 * @return  Status, 0 for success.
 */
int GPUMetricHandler::EnableTimeBasedStream(uint32_t timePeriod, uint32_t numReports)
{
    int          ret        = 0;
    int          retError   = 1;
    ze_result_t  status     = ZE_RESULT_SUCCESS;

    if (m_groupId < 0)  {
        DebugPrintError("No metrics enabled. Data collection abort\n");
        return retError;
    }
    ze_event_pool_desc_t  eventPoolDesc;
    eventPoolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    eventPoolDesc.pNext = nullptr;
    eventPoolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    eventPoolDesc.count = 1;
    status = zeEventPoolCreateFunc(m_context, &eventPoolDesc, 1, &m_device, &m_eventPool);
    ze_event_desc_t eventDesc;
    eventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    eventDesc.index   = 0;
    eventDesc.signal  = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait    = ZE_EVENT_SCOPE_FLAG_HOST;
    if (status == ZE_RESULT_SUCCESS) {
        status = zeEventCreateFunc(m_eventPool, &eventDesc, &m_event);
    }
    if (status == ZE_RESULT_SUCCESS) {
        zet_metric_streamer_desc_t  metricStreamerDesc;
        metricStreamerDesc.stype = ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC;
        if (numReports == 0) {
            metricStreamerDesc.notifyEveryNReports = MAX_REPORTS;
        } else {
            metricStreamerDesc.notifyEveryNReports = numReports;
        }
        metricStreamerDesc.samplingPeriod      = timePeriod;
        zet_metric_group_handle_t mGroup = m_groupList[m_groupId].handle;
        status = zetMetricStreamerOpenFunc(m_context, m_device, mGroup,
                                    &metricStreamerDesc, m_event, &m_metricStreamer);
    }
    if (status == ZE_RESULT_SUCCESS) {
        ret = 0;
        m_status = COLLECTION_ENABLED;
    } else {
        DebugPrintError("EnableTimeBasedStream: failed with status 0x%x, cleanup\n", status);
        if (m_metricStreamer) {
            status = zetMetricStreamerCloseFunc(m_metricStreamer);
            m_metricStreamer = nullptr;
        }
        if (m_event) {
            status = zeEventDestroyFunc(m_event);
            m_event = nullptr;
        }
        if (m_eventPool) {
            status = zeEventPoolDestroyFunc(m_eventPool);
            m_eventPool = nullptr;
        }
        status = zetContextActivateMetricGroupsFunc(m_context, m_device, 0, nullptr);
        ret = 1;
    }
    return ret;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn      int GPUMetricHandler::EnableEventBasedQuery(
 *                     uint32_t timePeriod, uint32_t numReports)
 *
 * @brief   Enable metric query on enabled metrics
 *
 * @param   IN  timePeriod    - The timer period for sampling
 * @param   OUT numReports    - total number of sample reports (not used).
 *                              Default is to collect all sample record available
 *
 * @return  Status, 0 for success.
 */
int GPUMetricHandler::EnableEventBasedQuery()
{
    ze_result_t  status     = ZE_RESULT_SUCCESS;
    int          ret        = 0;
    int          retError   = 0;

    if (m_groupId < 0)  {
        DebugPrintError("No metrics enabled. Data collection abort\n");
        return retError;
    }

    zet_metric_group_handle_t mGroup = m_groupList[m_groupId].handle;
    gQueryState = new QueryState;
    zet_metric_query_pool_desc_t metricQueryPoolDesc;
    if (m_context) {
        ze_context_desc_t ctxtDesc = {
            ZE_STRUCTURE_TYPE_CONTEXT_DESC,
            nullptr,
            0
        };
        zeContextCreateFunc(m_driver, &ctxtDesc, &m_context);
    }
    metricQueryPoolDesc.stype = ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC;
    metricQueryPoolDesc.type = ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE;
    metricQueryPoolDesc.count = MAX_KERNERLS;
    status = zetMetricQueryPoolCreateFunc(m_context, m_device, mGroup,
            &metricQueryPoolDesc, &m_queryPool);
    if (status == ZE_RESULT_SUCCESS) {
        gQueryState->queryPool = m_queryPool; 
        ze_event_pool_desc_t  eventPoolDesc;
        eventPoolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        eventPoolDesc.flags= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        eventPoolDesc.count =  MAX_KERNERLS;

        // create event to wait
        status = zeEventPoolCreateFunc(m_context, &eventPoolDesc, 1, &m_device, &m_eventPool);
        gQueryState->eventPool = m_eventPool;
    }
    zet_tracer_exp_desc_t tracerDesc;
    tracerDesc.stype = ZET_STRUCTURE_TYPE_TRACER_EXP_DESC;
    tracerDesc.pUserData = nullptr;

    if (status == ZE_RESULT_SUCCESS) {
        status = zetTracerExpCreateFunc(m_context, &tracerDesc, &m_tracer);
    }
    zet_core_callbacks_t prologCB = {};
    zet_core_callbacks_t epilogCB = {};

    if (status == ZE_RESULT_SUCCESS) {
        epilogCB.Kernel.pfnCreateCb = kernelCreateCB;
        epilogCB.Kernel.pfnDestroyCb = kernelDestroyCB;
        prologCB.CommandList.pfnAppendLaunchKernelCb = metricQueryBeginCB;
        epilogCB.CommandList.pfnAppendLaunchKernelCb = metricQueryEndCB;

        status = zetTracerExpSetProloguesFunc(m_tracer, &prologCB);
    }
    if (status == ZE_RESULT_SUCCESS) {
        status = zetTracerExpSetEpiloguesFunc(m_tracer, &epilogCB);
    }
    if (status == ZE_RESULT_SUCCESS) {
        status = zetTracerExpSetEnabledFunc(m_tracer, true);
    }
    if (status == ZE_RESULT_SUCCESS) {
        m_status = COLLECTION_ENABLED;

        ret  = 0;
    } else {
        DebugPrintError("EnableEventBasedQuery: failed with status 0x%x, abort.\n", status);
        if (m_tracer) {
            status = zetTracerExpDestroyFunc(m_tracer);
            m_tracer = nullptr;
        }
        if (m_event) {
            status = zeEventDestroyFunc(m_event);
            m_event = nullptr;
        }
        if (m_eventPool) {
            status = zeEventPoolDestroyFunc(m_eventPool);
            m_eventPool = nullptr;
        }
        if (m_queryPool) {
            status = zetMetricQueryPoolDestroyFunc(m_queryPool);
            m_queryPool = nullptr;
        }
        status = zetContextActivateMetricGroupsFunc(m_context, m_device, 0, nullptr);
        ret  = retError;
    }
    return ret;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn         void GPUMetricHandler::DisableMetricGroup()
 * @brief      Disable current metric group
 *             After metric group disabled, cannot read data anymore
 * 
 */
void
GPUMetricHandler::DisableMetricGroup()
{
    DebugPrint("enter DisableMetricGroup()\n");

    if (m_status == COLLECTION_ENABLED) {
        m_status = COLLECTION_DISABLED;
    }
    if (m_groupType == ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED) {
        if (m_metricStreamer) {
            zetMetricStreamerCloseFunc(m_metricStreamer);
            m_metricStreamer = nullptr;
        }
    }
    if (m_groupType == ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED) {
        if (m_tracer) {
            zetTracerExpDestroyFunc(m_tracer);
            m_tracer = nullptr;
        }
        if (m_queryPool) {
            zetMetricQueryPoolDestroyFunc(m_queryPool);
            m_queryPool = nullptr;
        }
    }
    if (m_event) {
        zeEventDestroyFunc(m_event);
        m_event = nullptr;
    }
    if (m_eventPool) {
        zeEventPoolDestroyFunc(m_eventPool);
        m_eventPool = nullptr;
    }
    zetContextActivateMetricGroupsFunc(m_context,  m_device, 0, nullptr);
    return;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn     MetricData * GPUMetricHandler::GetMetricsData(
 *                       uint32_t  mode, uint32_t *numReports, uint32_t *numMetrics)
 *
 * @brief  Read raw event data and calculate metrics, and return the overall aggregated metrics data.
 *
 *  @param In  mode             - report mode, summary or samples
 *  @param OUT numReports       - total number of sample reports
 *  @param OUT numMetrics       - number of metrics in each report
 *
 *  @return                     - metric data
 */
MetricData *
GPUMetricHandler::GetMetricsData(
    uint32_t         mode,
    uint32_t        *numReports,
    uint32_t        *numMetrics
)
{
    uint8_t     *rawBuffer   = nullptr;
    size_t       rawDataSize = 0;
    MetricData  *reportData        = nullptr;
    *numReports              = 0;
    *numMetrics              = 0;

    if  (m_groupType == ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED) {
        rawBuffer = ReadStreamData(&rawDataSize);
        if (rawBuffer) {
            GenerateMetricsData(rawBuffer, rawDataSize,  mode);
            delete [] rawBuffer;
        }
    } else {
        gQueryState->lock.lock();
        vector<QueryData>  qList;
        for (auto query : gQueryState->queryList) {
            rawDataSize = 0;
            rawBuffer = nullptr;
            ze_result_t status = ZE_RESULT_SUCCESS;
            rawBuffer = ReadQueryData(query, &rawDataSize, &status);
            if (status == ZE_RESULT_NOT_READY) {
                qList.push_back(query);
            }
            if (rawBuffer) {
                 GenerateMetricsData(rawBuffer, rawDataSize,  mode);
                 delete [] rawBuffer;
            }
        }
        gQueryState->queryList.clear();
        for (auto query : qList) {
             gQueryState->queryList.push_back(query);
        }
        gQueryState->lock.unlock();
    }

    // only SUMMARY mode
    TMetricNode *metricList = m_groupList[m_groupId].metricList;
    *numMetrics = m_groupList[m_groupId].props.metricCount;
    *numReports = 1;

    allocMetricData(reportData, 1);
    CHECK_N_RETURN_STATUS((reportData==nullptr), nullptr);
    reportData->numEntries = *numMetrics;
    allocMetricDataEntries(reportData->dataEntries, *numMetrics);
    if (reportData->dataEntries == nullptr) {
        freeMetricData(reportData, 1);
        return nullptr;
    }

    for (uint32_t j=0; j < *numMetrics; j++) {
        reportData->dataEntries[j].code = m_reportData->dataEntries[j].code;
        reportData->dataEntries[j].type = m_reportData->dataEntries[j].type;
        if (!m_reportCount) {
            continue;
        }
        if (metricList[j].summaryOp != M_OP_AVERAGE) {
            reportData->dataEntries[j].value.ival =
                        m_reportData->dataEntries[j].value.ival;
        } else {
            if (m_reportData->dataEntries[j].type) {  // calculate avg
                if (m_reportData->dataEntries[j].value.fpval != 0.0) {
                    reportData->dataEntries[j].value.fpval =
                          (m_reportData->dataEntries[j].value.fpval)/m_reportCount;
                }
            } else {
                reportData->dataEntries[j].value.ival =
                         (m_reportData->dataEntries[j].value.ival)/m_reportCount;
            }
        }
    }
    return reportData;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn    void GPUMetricHandler::GenerateMetricsData(
 *                       uint8_t *rawBuffer, size_t rawDataSize, uint32_t mode) 
 *
 * @brief Calculate metric data from raw event data, the result will be aggrated into global data
 *
 * @param IN  rawBuffer     - buffer for raw event data
 * @param IN  rawDataSize   - the size of raw event data in the buffer
 * @param IN  mode          - report mode
 *
 */
void
GPUMetricHandler::GenerateMetricsData(
    uint8_t         *rawBuffer,
    size_t           rawDataSize,
    uint32_t         mode
)
{
    if (!rawDataSize) {
    return;
    }

    ze_result_t status = ZE_RESULT_SUCCESS;
    zet_metric_group_handle_t mGroup = m_groupList[m_groupId].handle;
    TMetricNode *metricList = m_groupList[m_groupId].metricList;
    uint32_t numEntries = m_groupList[m_groupId].props.metricCount;
    uint32_t metricDataSize = 0;
    status = zetMetricGroupCalculateMetricValuesFunc(mGroup,
            ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
            (size_t)rawDataSize, (uint8_t*)rawBuffer,
            &metricDataSize, nullptr);
    if ((status != ZE_RESULT_SUCCESS) || (metricDataSize == 0)) {
        DebugPrint("No metric data memory space allocated\n");
        return;
    }
    zet_typed_value_t* typedDataList = new zet_typed_value_t[metricDataSize];
    if (typedDataList == nullptr) {
        return;
    }
    status = zetMetricGroupCalculateMetricValuesFunc(mGroup,
            ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
            (size_t)rawDataSize, (uint8_t*)rawBuffer,
            &metricDataSize, typedDataList);
    if (status != ZE_RESULT_SUCCESS)  {
        DebugPrintError("Failed on metrics calculation\n");
        delete [] typedDataList;
        return;
    }
    if (!metricDataSize) {
        DebugPrintError("No metric calculated.\n");
        delete [] typedDataList;
        return;
    }

    uint32_t reportCounts = metricDataSize/numEntries;
    if (mode & METRIC_RESET) {
        for (uint32_t j=0; j<numEntries; j++) {
            m_reportData->dataEntries[j].value.ival = 0;
        }
        m_reportCount = 0;
    }

    DebugPrint("metricDataSize 0x%x, reportCounts %d\n", 
                    metricDataSize, reportCounts);

     // log metric names
     /*
    if (m_stdout) {
        for (uint32_t j=0; j<numEntries; j++) {
             printf("%s, ", metricList[j].props.name);
        }
        printf("\n");
    }
    */
    for (uint32_t i=0; i < reportCounts; i++) {
        for (uint32_t j=0; j<numEntries; j++) {
             int dtype = 0;
             long long iVal = 0; 
             double fpVal = 0.0; 

             typedValue2Value(typedDataList[i*numEntries + j], 
                             &m_dataDump, m_stdout, &dtype, &iVal, &fpVal, (j==(numEntries-1)));
             if (metricList[j].summaryOp != M_OP_STATIC) {
                 m_reportData->dataEntries[j].type = dtype;
                 if (!dtype) {
                     m_reportData->dataEntries[j].value.ival += iVal;
                 } else {
                     m_reportData->dataEntries[j].value.fpval += fpVal;
                 }
             } else {
                 // static value
                 if (i==0) {
                     m_reportData->dataEntries[j].type = dtype;
                     if (!dtype)  {
                         m_reportData->dataEntries[j].value.ival = iVal;
                     } else {
                         m_reportData->dataEntries[j].value.fpval = fpVal;
                     }
                 }
             }
        }
    }
    delete [] typedDataList;
    m_reportCount += reportCounts;
    return;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn        uint8_t * GPUMetricHandler::ReadStreamData(size_t *rawDataSize)
 *
 * @brief      read raw time based sampling data
 *
 *  @param OUT rawDataSize      - total byte size of raw data read.
 *
 *  @return                     - the buffer pointer which contains the raw data.
 *                                 nullptr will be return if failed.
 */
uint8_t *
GPUMetricHandler::ReadStreamData(size_t *rawDataSize)
{
    ze_result_t status  = ZE_RESULT_SUCCESS;
    size_t      rawSize = 0;

   *rawDataSize = 0;
    //read raw data
    status = zetMetricStreamerReadDataFunc(m_metricStreamer, UINT32_MAX,
                                          &rawSize, nullptr);
    if (status !=  ZE_RESULT_SUCCESS) {
        DebugPrintError("ReadStreamData failed, status 0x%x, rawSize %d\n", status, (int)rawSize);
        return nullptr;
    }

    uint8_t * rawBuffer = new uint8_t[rawSize];
    CHECK_N_RETURN_STATUS((rawBuffer==nullptr), nullptr);
    if (!rawSize) {
       *rawDataSize = rawSize;
        return rawBuffer;
    }
    status = zetMetricStreamerReadDataFunc(m_metricStreamer, UINT32_MAX,
                                   &rawSize, (uint8_t *)rawBuffer);
    if (status !=  ZE_RESULT_SUCCESS) {
        DebugPrintError("ReadStreamData failed, status 0x%x\n", status);
        delete [] rawBuffer;
        return nullptr;
    }
    if (!rawSize)  {
        // thi may not be an error, especially in multi-thread case.
        DebugPrint("No Raw Data available. This could be collection time too short or buffer overflow. Please increase the sampling period\n");
    }
   *rawDataSize = rawSize;
    return rawBuffer;
}

/*------------------------------------------------------------------------------*/
/*!
 *  @fn        uint8_t * GPUMetricHandler::ReadQueryData(
 *                            QueryData &data,
 *                            size_t *rawDataSize,
 *                            ze_result_t * status)
 *
 *  @brief     read raw query based sampling data
 *
 *  @param OUT rawDataSize      - total # of bytes of raw data read.
 *  @param OUT retStatus        - return status
 *
 *  @return                     - the buffer pointer which contains the raw data.
 *                                 nullptr will be return if failed.
 */
uint8_t *
GPUMetricHandler::ReadQueryData(QueryData &data, size_t *rawDataSize, ze_result_t *retStatus)
{

    size_t      rawSize = 0;
   *rawDataSize = 0;

    std::map<std::string, MetricData> dataMap;
    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zeEventHostSynchronizeFunc(data.event, 1000);
    *retStatus = status;
    CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);
    status = zeEventDestroyFunc(data.event);
    *retStatus = status;
    CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);

    status = zetMetricQueryGetDataFunc(data.metricQuery, &rawSize, nullptr);
    *retStatus = status;
    CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);

    uint8_t *rawBuffer = new uint8_t[rawSize];
    CHECK_N_RETURN_STATUS((rawBuffer==nullptr), nullptr);
    status = zetMetricQueryGetDataFunc(data.metricQuery, &rawSize, rawBuffer);
    *retStatus = status;
    CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);

    status = zetMetricQueryDestroyFunc(data.metricQuery);
    *retStatus = status;
    CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);

    *rawDataSize = rawSize;
    return rawBuffer;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn    int GPUMetricHandler::SetControl(uint32_t mode)
 * 
 * @brief Set control 
 *
 * @param IN mode  -- control mode to set
 */
int
GPUMetricHandler::SetControl(uint32_t mode) {

    int ret = 0;
    if (mode & METRIC_RESET) {
        if (m_reportData) {
            uint32_t numEntries = m_groupList[m_groupId].props.metricCount;
            for (uint32_t j=0; j<numEntries; j++) {
                m_reportData->dataEntries[j].value.ival = 0;
            }
            m_reportCount = 0;
        }
    }
    return ret;
}

