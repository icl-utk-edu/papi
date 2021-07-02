/*
 * GPUMetricInterface.h:  Intel® Graphics Component for PAPI
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

#ifndef _GPUMETRICINTERFACE_H
#define _GPUMETRICINTERFACE_H

#pragma once

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef MAX_STR_LEN
#define MAX_STR_LEN      256
#endif

//metric group type
#define TIME_BASED   0
#define EVENT_BASED  1

typedef int DEVICE_HANDLE;

typedef char NAMESTR[MAX_STR_LEN];

typedef struct MetricInfo_S MetricInfo;

struct MetricInfo_S {
    unsigned int code;
    unsigned int dataType;
    unsigned int op;        // 0 sum,  1 average, 2 static
    char name[MAX_STR_LEN];
    char desc[MAX_STR_LEN];
    int numEntries;
    MetricInfo *infoEntries;
};

typedef struct DataEntry_S {
    unsigned int code; 
    unsigned int type;
    union {
       long long ival;
       double    fpval;
    } value;
} DataEntry;

typedef struct MetricData_S {
    int id;
    unsigned int grpCode; 
    unsigned int numEntries;
    DataEntry *dataEntries;
}  MetricData;

typedef struct MetricNode_S {
    char     name[MAX_STR_LEN];
    int      code;
} MetricNode;


#define MAX_METRICS       128

#define MAX_NUM_REPORTS   20

/* collection modes */
#define METRIC_SAMPLE     0x1
#define METRIC_SUMMARY    0x2
#define METRIC_RESET      0x4

#define allocMetricInfo(info, num)        {info = (MetricInfo *)calloc(num, sizeof(MetricInfo)); }

#define freeMetricInfo(infoList)   {                                  \
        if (infoList)    {                                            \
            if (infoList.infoEntries)  {                              \
                free((MetricInfo *)(infoList.infoEntries));           \
            }                                                         \
            free((MetricInfo *)infoList);                             \
            }}


#define allocMetricData(data, num)       {data = (MetricData *)calloc(num, sizeof(MetricData)); }

#define allocMetricDataEntries(entries, num)   {entries = (DataEntry *)calloc(num, sizeof(DataEntry)); } 

#define freeMetricData(data, num)   {                                \
        if (data)    {                                               \
            for (unsigned int rid=0; rid<num; rid++) {               \
                if (data[rid].dataEntries)  {                        \
                    free(data[rid].dataEntries);                     \
                }                                                    \
            }                                                        \
            free(data);                                              \
        }}

#define freeMetricDataEntries(entries)     {free(entries);}

extern void
strncpy_se(char *dest, size_t destSize,  char *src, size_t count);


/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*!
 * @fn          int GPUDetectDevice(DEVICE_HANDLE *handle, int *numDevice);
 *
 * @brief       Detect the named device which has performance monitoring feature availale.
 *
 * @param       OUT  handle    - a array of handle, each for an instance of the device
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUDetectDevice(DEVICE_HANDLE *handle, int *numDevice);


/* ------------------------------------------------------------------------- */
/*!
 * @fn          int GPUGetDeviceInfo(DEVICE_HANDLE handle, MetricInfo *data);
 *
 * @brief       Get the device properties, which is mainly in <name, value> format
 *
 * @param       IN   handle - handle to the selected device
 * @param       OUT  data   - the property data
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUGetDeviceInfo(DEVICE_HANDLE handle, MetricInfo *data);

/* ------------------------------------------------------------------------- */
/*!
 * @fn          int GPUPrintDeviceInfo(DEVICE_HANDLE handle, FILE *stream);
 *
 * @brief       Get the device properties, which is mainly in <name, value> format
 *
 * @param       IN   handle - handle to the selected device
 * @param       IN   stream   - IO stream to print
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUPrintDeviceInfo(DEVICE_HANDLE handle, FILE *stream);


/* ------------------------------------------------------------------------- */
/*!
 * @fn          int GPUGetMetricGroups(DEVICE_HANDLE handle, unsigned int mtype, MetricInfo *data);
 *
 * @brief       list available metric groups for the selected type
 *
 * @param       IN   handle - handle to the selected device
 * @param       IN   mtype  - metric group type
 * @param       OUT  data   - metric data
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUGetMetricGroups(DEVICE_HANDLE handle, unsigned int mtype, MetricInfo *data);

/* ------------------------------------------------------------------------- */
/*!
 * @fn          int GPUGetMetricList(DEVICE_HANDLE handle, 
 *                                   char *groupName, unsigned int mtype, MetricInfo *data);
 *
 * @brief       list available metrics in the named group.
 *              If name is "", list all available metrics in all groups
 *
 * @param       IN   handle      - handle to the selected device
 * @param       IN   groupName   - metric group name. "" means all groups.
 * @param       In   mtype       - metric type <TIME_BASED, EVENT_BASED>
 * @param       OUT  data   - metric data
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUGetMetricList(DEVICE_HANDLE handle, char *groupName, unsigned int mtype, MetricInfo *data);

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUEnableMetrics(DEVICE_HANDLE handle, char *metricGroupName,
 *                          unsigned int stype,  unsigned int period, unsigned int numReports)
 *
 * @brief       enable named metricgroup to collection.
 *
 * @param       IN   handle             - handle to the selected device
 * @param       IN   metricGroupName    - metric group names
 * @param       IN   mtype              - metric type <TIME_BASED, EVENT_BASED>
 * @param       IN   period             - collection timer period
 * @param       IN   numReports         - number of reports. Default collect all available metrics
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUEnableMetricGroup(DEVICE_HANDLE handle, char *metricGroupName, unsigned int mtype,
                 unsigned int period, unsigned int numReports);

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUEnableMetrics(DEVICE_HANDLE handle, char **metricNameList,
 *                          unsigned numMetrics, unsigned int stype,
 *                          unsigned int period, unsigned int numReports)
 *
 * @brief       enable named metrics on a metric group to collection.
 *
 * @param       IN   handle             - handle to the selected device
 * @param       IN   metricNameList     - a list of metric names to be collected
 * @param       IN   numMetrics         - number of metrics to be collected
 * @param       IN   mtype              - metric type <TIME_BASED, EVENT_BASED>
 * @param       IN   period             - collection timer period
 * @param       IN   numReports         - number of reports. Default collect all available metrics
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUEnableMetrics(DEVICE_HANDLE handle, char **metricNameList,
                     unsigned int numMetrics, unsigned int mtype, 
                     unsigned int period, unsigned int numReports);

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUDisableMetrics(DEVICE_HANDLE handle, char *metricGroupName);
 *
 * @brief       disable current metric collection
 *
 * @param       IN   handle           - handle to the selected device
 * @param       IN   metricGroupName  - a metric group name
 * @param       IN   mtype            - a metric group type
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUDisableMetricGroup(DEVICE_HANDLE handle, char *metricGroupName, unsigned mtype);


/*------------------------------------------------------------------------- */
/*!
 * @fn int GPUStart(DEVICE_HANDLE handle);
 *
 * @brief       start collection
 *
 * @param       IN   handle - handle to the selected device
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUStart(DEVICE_HANDLE handle);


/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUStop(DEVICE_HANDLE handle,  MetricData **data, int *reportCounts);
 *
 * @brief       stop collection
 *
 * @param       IN   handle         - handle to the selected device
 * @param       OUT  data           - returned metric data array
 * @param       OUT  reportCounts   - returned metric data array size
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUStop(DEVICE_HANDLE handle);



/* ------------------------------------------------------------------------- */
/*!
 * @fn Metricdata *GPUReadMetricData(DEVICE_HANDLE handle, unsigned mode, unsigned int *reportCounts);
 *
 * @brief       read metric data
 *
 * @param       IN   handle         - handle to the selected device
 * @param       IN   mode           - collection mode (sample, summary, reset)
 * @param       OUT  reportCounts   - returned metric data array size
 *
 * @return      data                - returned metric data array
 */
MetricData *GPUReadMetricData(DEVICE_HANDLE handle, unsigned mode, unsigned int *reportCounts);

/* ------------------------------------------------------------------------- */
/*!
 * @fn Metricdata *GPUSetMetricControl(DEVICE_HANDLE handle, unsigned mode);
 *
 * @brief       set  control for metric data collection
 *
 * @param       IN   handle         - handle to the selected device
 * @param       IN   mode           - collection mode (sample, summary, reset)
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUSetMetricControl(DEVICE_HANDLE handle, unsigned mode);


/* ------------------------------------------------------------------------- */
/*!
 * @fn void GPUFreeDevice(DEVICE_HANDLE handle);
 *
 * @brief       free the resouce related this device handle
 *
 * @param       IN   handle       - handle to the selected device
 *
 * @return      0 -- success,  otherwise, error code
 */
void GPUFreeDevice(DEVICE_HANDLE handle);

#if defined(__cplusplus)
}
#endif

#endif
