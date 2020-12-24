/*
 * GPUMetricHandler.h:  Intel® Graphics Component for PAPI
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

#ifndef _GPUMETRICHANDLER_H
#define _GPUMETRICHANDLER_H

#pragma once

#include <stdio.h>
#include <string.h>
#include <list>
#include <string>
#include <fstream>
#include <vector>
#include <atomic>
#include <map>
#include <mutex>

#include "level_zero/zet_api.h"

#include "GPUMetricInterface.h"

/* collection status */
#define COLLECTION_IDLE      0
#define COLLECTION_INIT      1
#define COLLECTION_ENABLED   2
#define COLLECTION_DISABLED  3
#define COLLECTION_COMPLETED 4


using namespace std;

typedef struct TMetricNode_S 
{
    uint32_t                             code;           // 0 mean invalid
    uint32_t                             metricId;
    uint32_t                             metricGroupId;
    uint32_t                             metricDomainId;
    int                                  summaryOp;
    zet_metric_properties_t              props;
    zet_metric_handle_t                  handle;
} TMetricNode;

typedef struct TMetricGroupNode_S
{
    uint32_t                               code;         // 0 mean invalid
    uint32_t                               numMetrics;
    TMetricNode                           *metricList;
    int                                   *opList;       // list of metric operation
    zet_metric_group_properties_t          props;
    zet_metric_group_handle_t              handle;
}TMetricGroupNode;

typedef struct QueryData_S {
    string                                  kernName;
    zet_metric_query_handle_t               metricQuery;
    ze_event_handle_t                       event;
} QueryData;


typedef struct QueryState_S {
    atomic<uint32_t>                        kernelId{0};
    std::mutex                              lock;
    std::map<ze_kernel_handle_t, string>    nameMap;
    zet_metric_query_pool_handle_t          queryPool;
    ze_event_pool_handle_t                  eventPool;
    vector<QueryData>                       queryList;
} QueryState;


class GPUMetricHandler
{
public:
    static GPUMetricHandler* GetInstance();
    ~GPUMetricHandler();
    int           InitMetricDevice(int *numDevice);
    void          DestroyMetricDevice();
    int           EnableMetricGroup(const char *metricGroupName,
                        uint32_t *metricSelected, uint32_t mtype);
    int           EnableTimeBasedStream(uint32_t timePeriod, uint32_t numReports);
    int           EnableEventBasedQuery();
    void          DisableMetricGroup();
    int           GetMetricsInfo(int type, MetricInfo *data);
    int           GetMetricsInfo(const char * name, int type, MetricInfo *data);
    int           GetMetricCode(const char *mGroupName, const char *metricName,    uint32_t mtype, 
                        uint32_t *mGroupCode, uint32_t *metricCode);
    MetricData   *GetMetricsData(uint32_t  mode, uint32_t *numReports, uint32_t *numMetrics);
    int           SetControl(uint32_t mode);

private:
    GPUMetricHandler();
    GPUMetricHandler(GPUMetricHandler const&);
    void      operator=(GPUMetricHandler const&);
    int       InitMetricGroups(ze_device_handle_t device);
    string    GetDeviceName(ze_device_handle_t device);
    uint8_t  *ReadStreamData(size_t *rawDataSize);
    uint8_t  *ReadQueryData(QueryData &data, size_t *rawDataSize, ze_result_t *retStatus);
    void      GenerateMetricsData(uint8_t *rawData, size_t rawDataSize, uint32_t  mode);

private: // Fields
    static GPUMetricHandler*   m_handlerInstance;

    string                     m_dataDumpFileName;
    string                     m_dataDumpFilePath;
    fstream                    m_dataDump;

    // current state
    ze_driver_handle_t         m_driver;
    ze_context_handle_t        m_context;
    ze_device_handle_t         m_device;
    TMetricGroupNode *         m_groupList;
    uint32_t                   m_numGroups;
    uint32_t                   m_numMetrics;
    uint32_t                   m_maxMetricsPerGroup;
    uint32_t                   m_domainId;
    int                        m_groupId;
    uint32_t                   m_groupType;
    uint32_t                  *m_metricsSelected;

    ze_event_pool_handle_t     m_eventPool;
    ze_event_handle_t          m_event;

    zet_metric_streamer_handle_t m_metricStreamer;

    zet_metric_query_pool_handle_t m_queryPool;
    zet_tracer_exp_handle_t    m_tracer;

    volatile int               m_status;
    int                        m_stdout;

    MetricData                *m_reportData;
    uint32_t                   m_reportCount;
};


#endif

