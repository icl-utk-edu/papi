/* 
* File:    p4_events.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#include "papi.h"

#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
  #define SUBSTRATE "linux-perfctr.h"
#endif

#include SUBSTRATE

#ifdef PAPI3
#include "papi_internal.h"
#include "papi_protos.h"
#endif

/* Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example. 

   The third field is meant to be the 'read selector' but it is not implemented. */

const P4_search_t _papi_hwd_pentium4_mlt2_preset_map[] = {
  { PAPI_TOT_INS, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0x0C) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(4) | ENABLE, EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, (1<<0xC)|(1<<0x8), 1<<0xC, 2,
    {{{COUNTER(0x0C) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      {COUNTER(0x08) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(1) | ENABLE, EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1<<0xD, 1<<0xD, 1,
    {{{COUNTER(0xD) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(4) | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), EVENT(0x3f) | CPL(1)} }}},
  { PAPI_L1_LDM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L1_STM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE} }}},
  { PAPI_L1_DCM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L2_LDM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L2_STM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE} }}},
  { PAPI_L2_DCM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { 0, NULL, }
};

const P4_search_t _papi_hwd_pentium4_mge2_preset_map[] = {
  { PAPI_TOT_INS, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0x0C) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(4) | ENABLE, EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, (1<<0xC)|(1<<0x8), 1<<0xC, 2,
    {{{COUNTER(0x0C) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      {COUNTER(0x08) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(1) | ENABLE, EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1<<0xD, 1<<0xD, 1,
    {{{COUNTER(0xD) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(4) | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), EVENT(0x3f) | CPL(1)} }}},
  { PAPI_L1_LDM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L1_STM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE} }}},
  { PAPI_L1_DCM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L2_LDM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L2_STM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE} }}},
  { PAPI_L2_DCM, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0xC) | FAST_RDPMC, HYPERTHREAD_ANY | ESCR(5) | ENABLE, EVENT(9) | EVENTMASK(0x1) | CPL(1), PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { 0, NULL, }
};

