/*
 * GPUMetricInterface.cpp:  Intel® Graphics Component for PAPI
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

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>

#include "GPUMetricInterface.h"
#include "GPUMetricHandler.h"

#ifdef __cplusplus
extern "C" {
#endif 

//#define _DEBUG 1

#define  DebugPrintError(format, args...)    fprintf(stderr, format, ## args)

#if defined(_DEBUG)
#define DebugPrint(format, args...)          fprintf(stderr, format, ## args)
#else
#define DebugPrint(format, args...)          {do { } while(0);}
#endif


#define UNINITED    0
#define INITED      1
#define ENABLED     2

#define MAX_ENTRY     256
#define MAX_HANDLES    16

/* current collection state */
//static char          curMetricGroupName[MAX_STR_LEN];
static char            *curMetricGroupName         = nullptr;
static unsigned int     curMetricGroupCode         = 0;
static int              numMetricsSelected         = 0;
static MetricNode       metrics_selected[MAX_ENTRY];
static unsigned int     curMetricList[MAX_ENTRY];

static int              curHandleTableIndex        = 0;
static GPUMetricHandler *handleTable[MAX_HANDLES];

static int              colState                   = UNINITED;
static int              runningSessions            = 0;

// lock to make sure API is thread-safe
static pthread_mutex_t  iflock                     = PTHREAD_MUTEX_INITIALIZER;

/*
 * wrapper function for safe strncpy
 */
void 
strncpy_se(char *dest, size_t destSize,  char *src, size_t count)  {
    if (dest && src) {
        size_t toCopy = (count <= strlen(src))? count:strlen(src);
        if (toCopy < destSize) {
            memcpy(dest, src, toCopy);
            dest[toCopy] = '\0';
        }
    }
}

/*
 * Check whether the metric code is already in the list
 */
static int
isInList(MetricNode *list,  int size,  int code) {
    int retError = 1;
    int ret      = 0;
    for (int i=0; i<size; i++) {
        if (list[i].code == code) {
            return retError;
    }
    }
    return ret;
}

static int
findMetricName(char *fullName, char *groupName, char *metricName, int maxNameSize) {
    int retError = 1;
    int ret      = 0;
    if (!fullName || !strlen(fullName)) {
        return retError;
    }
    int   fullNameLen =  strlen(fullName);

    char *pt = strchr(fullName, '.');
    if (!pt || (pt == &fullName[0]) || (pt == &fullName[fullNameLen-1])) {
        return retError;
    }
    // metric name in the format of  metricGroupName.metricNmae
    int groupNameSize = pt - fullName;
    int metricNameSize = fullNameLen - groupNameSize -1;
    if ((groupNameSize >= maxNameSize) || (metricNameSize >= maxNameSize)) {
        return retError;
    }
    strncpy_se(groupName, MAX_STR_LEN, fullName, groupNameSize);
    pt++;
    strncpy_se(metricName, MAX_STR_LEN, pt, metricNameSize);
    return ret;
}

/*============================================================================
 * Below are Wrapper interface functions
 *============================================================================*/

/* ------------------------------------------------------------------------- */
/*!
 * @fn          int GPUDetectDevice(DEVICE_HANDLE *handle, int *num_device);
 *
 * @brief       Detect and init GPU device which has performance metrics availale.
 *
 * @param       OUT  handle    - a array of handle, each for an instance of the device
 * @param       OUT  numDevice - total number of device detected
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUDetectDevice(DEVICE_HANDLE *handle, int *numDevice)
{
    int               ret             = 0;
    int               retError        = 1;
    GPUMetricHandler *curMetricHandle = nullptr;
    if ((colState != UNINITED) && curHandleTableIndex) {
        *numDevice = 1;
        *handle = curHandleTableIndex;
        return ret;
    }

    if (!curHandleTableIndex) { // first time
        memset((char *)&handleTable, 0, sizeof(GPUMetricHandler *)*16);
    }
    if (!curHandleTableIndex || !handleTable[curHandleTableIndex]) {
        curMetricHandle = GPUMetricHandler::GetInstance();
        ret =  curMetricHandle->InitMetricDevice(numDevice);
        if (ret || !numDevice) {
            curHandleTableIndex = 0;
            DebugPrintError("GPUDetectDevice failed, device does not exist or being used, abort\n");
            return retError;    
        }
        curHandleTableIndex = 1;
        handleTable[curHandleTableIndex] = curMetricHandle;
    }
    // only deal with one device.
    *handle = curHandleTableIndex;
    *numDevice = 1;
    curMetricGroupCode = 0;

    colState = INITED;
    return ret;    
}

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
void GPUFreeDevice(DEVICE_HANDLE handle)
{
    if (colState == UNINITED) {
        return;
    }
    if (!handle || (handle != curHandleTableIndex) || !handleTable[handle]) {
        DebugPrintError("GPUFreeDevice failed, handle is not valid or device is not initiated!\n");
        return;
    }

    handleTable[handle]->DestroyMetricDevice();
    handleTable[handle] = nullptr;
    curHandleTableIndex = 0;
    colState = UNINITED;
}


/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUEnableMetricGroup(DEVICE_HANDLE handle, char *metricGroupName
 *                 unsigned int mtype,  unsigned int period, unsigned int numReports)
 *
 * @brief       add named metric group to collection.
 *
 * @param       IN   handle              - handle to the selected device
 * @param       IN   metricGroupName     - a metric group name
 * @param       IN   mtype               - metric type <TIME_BASED,  EVENT_BASED>
 * @param       IN   period              - collection timer period
 * @param       IN   numReports          - number of reports. Default collect all available metrics
 *
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUEnableMetricGroup(DEVICE_HANDLE handle, char *metricGroupName,  unsigned int mtype,
                 unsigned int period, unsigned int numReports)
{
    int ret                 = 0;
    int retError            = 1;
    unsigned int groupCode  = 0;

    if (!handle || (handle != curHandleTableIndex) || !handleTable[handle]) {
        DebugPrintError("GPUEnableMetricGroup: device handle is not initiated!\n");
        return retError;
    }

    if (!metricGroupName || !strlen(metricGroupName) || strlen(metricGroupName) >= MAX_STR_LEN) {
        DebugPrintError("GPUEnableMetricGroup: no metric group selected\n");
        return retError;
    }
    pthread_mutex_lock(&iflock);
    runningSessions++; 
    if (colState == ENABLED) {
        pthread_mutex_unlock(&iflock);
        return ret;
    }

    GPUMetricHandler *curMetricHandle = handleTable[handle];

    if (!curMetricGroupCode) {
        if (!curMetricGroupName) {
            curMetricGroupName = (char *)calloc(MAX_STR_LEN, sizeof(char));
            if (!curMetricGroupName) {
                DebugPrintError("GPUEnableMetricGroup: memory alloc failed, abort!\n");
                ret = retError;
                goto cleanup;
            }
        }
        strncpy_se(curMetricGroupName, MAX_STR_LEN,  metricGroupName, strlen(metricGroupName));
        ret=curMetricHandle->GetMetricCode(metricGroupName, "", mtype, &groupCode, nullptr);
        if (ret) {
            DebugPrintError("GPUEnableMetricGroup: metric group %s is not supported, abort!\n",
                 metricGroupName);
            goto cleanup;
        }
        curMetricGroupCode = groupCode;
        numMetricsSelected = 0;   // select all
    } else {
        if (strncmp(curMetricGroupName, metricGroupName, strlen(curMetricGroupName)) != 0) {
            DebugPrintError("GPUEnableMetricGroup: metric group %s cannot be collected with metric group %s\n",
                metricGroupName, curMetricGroupName);
            ret = retError;
            goto cleanup;
        }
    }

    if (!numMetricsSelected) {  // selected all
        for (unsigned int i=0; i<MAX_ENTRY; i++) {
            curMetricList[i] = 1;   
        }
    }
    ret = curMetricHandle->EnableMetricGroup(metricGroupName, curMetricList, mtype);
    if (ret) {
        DebugPrintError("GPUEnableMetricGroup %s failed, return %d\n", metricGroupName, ret);
        goto cleanup;
    }
    if (mtype == TIME_BASED) {
        ret = curMetricHandle->EnableTimeBasedStream(period, numReports);
    } else {
        ret = curMetricHandle->EnableEventBasedQuery();
    }

cleanup:
    if (!ret) {
        colState = ENABLED;
    } else {
        runningSessions--;
    }
    pthread_mutex_unlock(&iflock);

    return ret;
}



/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUEnableMetrics(DEVICE_HANDLE handle, char **metricNameList, 
 *                    unsigned numMetrics, unsigned int mtype,
 *                        unsigned int period, unsigned int numReports)
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
             unsigned int period, unsigned int numReports)
{
    int          ret        = 0;
    int          retError   = 1;
    unsigned int groupCode  = 0;
    unsigned int metricCode = 0;

    if (!handle || (handle != curHandleTableIndex) || !handleTable[handle]) {
        DebugPrintError("GPUEnableMetrics: device handle is not initiated!\n");
        return retError;
    }
   
    if (!metricNameList || !numMetrics) {
        DebugPrintError("GPUEnableMetrics: no metric added\n");
        return retError;
    }

    pthread_mutex_lock(&iflock);
    runningSessions++; 
    if (colState == ENABLED) {
        pthread_mutex_unlock(&iflock);
        return ret;
    }

    GPUMetricHandler *curMetricHandle = handleTable[handle];
    char  groupName[MAX_STR_LEN]; 
    char  metricName[MAX_STR_LEN]; 

    for (unsigned int i=0; i<numMetrics; i++) {
        if (findMetricName(metricNameList[i], groupName, metricName, MAX_STR_LEN)) {
            DebugPrintError("GPUEnableMetrics: metric %s is not supported, expect format as <groupName.metricName>, abort!\n", metricNameList[i]);
           ret = retError;
           goto cleanup;
        }
        ret = curMetricHandle->GetMetricCode(groupName, metricName,  mtype, &groupCode, &metricCode);
        if (ret || !groupCode || !metricCode) {
            DebugPrintError("GPUEnableMetrics: metric %s is not supported, abort!\n", 
                             metricNameList[i]);
            ret = retError;
            goto cleanup;
        }

        if (!curMetricGroupCode) {  // first event with metric group
            curMetricGroupCode = groupCode;
            if (!curMetricGroupName)  {
                curMetricGroupName = (char *)calloc(MAX_STR_LEN, sizeof(char));
                if (!curMetricGroupName) {
                     DebugPrintError("GPUEnableMetrics: memory alloc failed, abort!\n");
                     ret = retError;
                     goto cleanup;
                }
            }
            strncpy_se(curMetricGroupName, MAX_STR_LEN, groupName, strlen(groupName));
        } else {
            if (groupCode != curMetricGroupCode) {
                DebugPrintError("GPUEnableMetrics: cannot enable metrics in multiple groups <%s, %s> at the same time, abort!\n", curMetricGroupName, groupName);
                 ret = retError;
                 goto cleanup;
            }
        }
        if  (!isInList(metrics_selected, numMetricsSelected, metricCode)) {
            strncpy_se(metrics_selected[numMetricsSelected].name, MAX_STR_LEN,
               metricNameList[i], strlen(metricNameList[i]));
            unsigned int mcode = ((metricCode & 0xff));  //  mcode start from 1.
            curMetricList[mcode-1] = 1;
            metrics_selected[numMetricsSelected].code =  metricCode;
            numMetricsSelected++;
        }
    }
    ret = curMetricHandle->EnableMetricGroup(curMetricGroupName, curMetricList, mtype);
    if (ret) {
        DebugPrintError("EnableMetricGroup %s failed, return %d\n", curMetricGroupName, ret);
        goto cleanup;
    }

    if (mtype == TIME_BASED) {
        ret = curMetricHandle->EnableTimeBasedStream(period, numReports);
    } else {
        ret = curMetricHandle->EnableEventBasedQuery();
    }
cleanup:
    if (!ret) {
        colState = ENABLED;
    } else {
        DebugPrintError("Enabling metrics %s failed, return %d\n", curMetricGroupName, ret);
        runningSessions--; 
    }
    pthread_mutex_unlock(&iflock);
    return ret;
}

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUDisableMetricGroup(DEVICE_HANDLE handle, char *metricGroupName, unsigned int mtype);
 *
 * @brief       add named metric to collection.
 *
 * @param       IN   handle            - handle to the selected device
 * @param       IN   metricGroupName   - a metric group name
 * @param       IN   mtype             - a metric group type
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUDisableMetricGroup(DEVICE_HANDLE handle, char *metricGroupName, unsigned mtype)
{

    (void)metricGroupName;
    int ret               = 0;
    int retError          = 1;

    if (!handle || (handle != curHandleTableIndex) || !handleTable[handle]) {
        DebugPrintError("GPUDisableMetricGroup: device handle is not initiated!\n");
        return retError;
    }
    pthread_mutex_lock(&iflock);
    runningSessions--;
    if ((runningSessions == 0) && (colState == ENABLED)) {
       if (mtype == TIME_BASED) {
           handleTable[handle]->DisableMetricGroup();
        }
        colState = INITED;
    }
    pthread_mutex_unlock(&iflock);
    return ret;

}

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUSetMetricControl(DEVICE_HANDLE handle, unsigned int mode);
 *
 * @brief       set metroc collection control
 *
 * @param       IN   handle            - handle to the selected device
 * @param       IN   mode              - a metric collection control mode
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUSetMetricControl(DEVICE_HANDLE handle, unsigned int mode)
{
    int ret               = 0;
    int retError          = 1;
    if (!handle || (handle != curHandleTableIndex) || !handleTable[handle]) {
        DebugPrintError("GPUStop: device handle is not initiated!\n");
        return retError;
    }
    pthread_mutex_lock(&iflock);
    handleTable[handle]->SetControl(mode);
    pthread_mutex_unlock(&iflock);
    return ret;
}


/* ----------------------------------------------------------------------------------------- */
/*!
 * @fn         MetricData *GPUReadMetricData(DEVICE_HANDLE handle, unsigned int *reportCounts);
 *
 * @brief       read metric data
 *
 * @param       IN   handle         - handle to the selected device
 * @param       IN   mode           - reprot data mode,  METRIC_SUMMARY, METIC_SAMPLE
 * @param       OUT  reportCounts   - returned metric data array size
 *
 * @return      data                - returned metric data array
 */
MetricData *GPUReadMetricData(DEVICE_HANDLE handle, unsigned int mode, unsigned int *reportCounts)
{
    MetricData   *reportData = nullptr;

    if (!handle || (handle != curHandleTableIndex) || !handleTable[handle]) {
        DebugPrintError("GPUReadMetricData: device handle is not initiated!\n");
        return nullptr;
    }
    DebugPrint("GPUReadMetricData:  numMtricsSelected %d\n", numMetricsSelected);

    unsigned int numReports = 0;
    unsigned int numMetrics = 0;

    pthread_mutex_lock(&iflock);
    reportData = handleTable[handle]->GetMetricsData(mode, &numReports, &numMetrics);
    pthread_mutex_unlock(&iflock);
    if (!reportData) {
        DebugPrintError("Failed on GPUReadMetricData\n");
       *reportCounts = 0;
        return nullptr;
    }

    if (!numReports || !numMetrics) {
        DebugPrintError("GPUReadMetricData: No metric is collected\n");
       *reportCounts = 0;
        freeMetricData(reportData, numReports);
        return nullptr;
    }
    DebugPrint("GPUReadMetricData:  GetMetricsData numreport %d, numMetrics %d, numSelected %d\n", 
                numReports, numMetrics, numMetricsSelected);

#if defined(_DEBUG)
    for (int j=0; j<(int)numReports; j++) {
        DebugPrint("reportData[%d], numEntries %d\n", j, reportData[j].numEntries);
        for (int i=0; i<reportData[j].numEntries; i++) {
             DebugPrint("record[%d], metric [%d]: code 0x%x, ", j, i, reportData[j].dataEntries[i].code);
            if (reportData[j].dataEntries[i].type) {
                 DebugPrint("value %lf \n", reportData[j].dataEntries[i].value.fpval);
            } else {
                 DebugPrint("value %llu\n", reportData[j].dataEntries[i].value.ival);
            }
        }
    }
#endif

    if (!numMetricsSelected) {
        *reportCounts = numReports;
        return reportData;
    }

    // only return selected metrics
    int nums = reportData[0].numEntries;
    for (unsigned int j=0; j<numReports; j++) {
        DataEntry *entries = reportData[j].dataEntries; 
        nums = reportData[j].numEntries;
        reportData[j].numEntries = numMetricsSelected; 
        allocMetricDataEntries(reportData[j].dataEntries, numMetricsSelected);
        int index = 0;
        for (int i=0; i<nums; i++) {
            if (curMetricList[i]) {
                reportData[j].dataEntries[index].code = entries[i].code;
                reportData[j].dataEntries[index].type = entries[i].type;
                DebugPrint("return entry[%d], reportData[%d].entries[%d], code 0x%x, type %d, ", 
                            i, j, index, 
                reportData[j].dataEntries[index].code,
                reportData[j].dataEntries[index].type);
                if (!reportData[j].dataEntries[index].type) {
                    reportData[j].dataEntries[index].value.ival = entries[i].value.ival;
                    DebugPrint("ival %llu\n", reportData[j].dataEntries[index].value.ival);
                } else  {
                    reportData[j].dataEntries[index].value.fpval = entries[i].value.fpval;
                    DebugPrint("pfval %lf\n", reportData[j].dataEntries[index].value.fpval);
                }
                index++;
            }
        }
        freeMetricDataEntries(entries);
    }

    *reportCounts = (int)numReports;
    return reportData;
}


/* ------------------------------------------------------------------------- */
/*!
 * @fn          int GPUGetMetricGroups(DEVICE_HANDLE handle, unsigned int mtype, MetricInfo *data);
 *
 * @brief       list all available metric groups for the selected type 
 * @param       IN   handle - handle to the selected device
 * @param       IN   mtype  - metric group type
 * @param       OUT  data   - metric data
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUGetMetricGroups(DEVICE_HANDLE handle, unsigned int mtype, MetricInfo *data)
{
    int ret               = 0;
    int retError          = 1;

    if (!handle || (handle != curHandleTableIndex) || !handleTable[handle]) {
        DebugPrintError("GPUGetMetricGroups: device handle is not initiated!\n");
        return retError;
    }
    pthread_mutex_lock(&iflock);
    ret = handleTable[handle]->GetMetricsInfo(mtype, data);
    pthread_mutex_unlock(&iflock);
    if (ret) { 
        DebugPrintError("GPUGetMetricGroups failed, return %d\n", ret);
    }
#if defined(_DEBUG)
    for (int i=0; i<data->numEntries; i++) {
        DebugPrint("GPUGetMetricGroups: metric group[%d]: %s, code 0x%x, numEntries %d\n",
            i, data->infoEntries[i].name, data->infoEntries[i].code,
            data->infoEntries[i].numEntries);
    }
#endif

    return ret;
}



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
 * @param       OUT  data        - metric data
 *
 * @return      0 -- success,  otherwise, error code
 */
int GPUGetMetricList(DEVICE_HANDLE handle, char *groupName, unsigned mtype, MetricInfo *data)
{
    int ret      = 0;
    int retError = 1;

    if (!handle || (handle != curHandleTableIndex) || !handleTable[handle]) {
        DebugPrintError("GPUGetMetricList: device handle is not initiated!\n");
        return retError;
    }
    if (groupName == nullptr) {
        return retError;
    }
    if (strlen(groupName) > 0) {
        ret = handleTable[handle]->GetMetricsInfo(groupName, mtype, data);
    } else {
        ret = handleTable[handle]->GetMetricsInfo("", mtype, data);
    }
    if (ret) {
        DebugPrintError("GPUGetMetrics [%s] failed, return %d\n", groupName, ret);
    }
#if defined(_DEBUG)
    for (int i=0; i<data->numEntries; i++) {
        if (strlen(groupName) == 0) {
            DebugPrint("GPUGetMetrics: get all metric groups\n");
        } else {
            DebugPrint("GPUGetMetrics: get metrics in metric group %s\n", groupName);
        }
        DebugPrint("data[%d]: %s, code 0x%x,  numEntries %d\n", 
            i, data->infoEntries[i].name, data->infoEntries[i].code, 
            data->infoEntries[i].numEntries);
    }
#endif

    return ret;
}


#if defined(__cplusplus)
}
#endif
