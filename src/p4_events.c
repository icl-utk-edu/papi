/* 
* File:    p4_events.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#include "papi.h"

#ifndef _WIN32
  #include SUBSTRATE
#else
  #include "win32.h"
#endif

#ifdef PAPI3
#include "papi_internal.h"
#include "papi_protos.h"
#endif

/* Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example. */

/* You requested all the ESCR/CCCR/Counter triplets that allow one to
count cycles.  Well, this is a special case in that an ESCR is not
needed at all. By configuring the threshold comparison appropriately
in a CCCR, you can get the counter to count every cycle, independent
of whatever ESCR the CCCR happens to be listening to.  To do this, set
the COMPARE and COMPLEMENT bits in the CCCR and set the THRESHOLD
value to "1111" (binary).  This works because the setting the
COMPLEMENT bit makes the threshold comparison to be "less than or
equal" and, with THRESHOLD set to its maximum value, the comparison
will always succeed and the counter will increment by one on every
clock cycle. */

const P4_search_t _papi_hwd_pentium4_mlt2_preset_map[] = {
  { PAPI_RES_STL, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x9) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_BR_INS, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_BR_TKN, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xc) | CPL(1)} }}},
  { PAPI_BR_NTK, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_BR_MSP, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xa) | CPL(1)} }}},
  { PAPI_BR_PRC, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x5) | CPL(1)} }}},
  { PAPI_TLB_DM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_TLB_IM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x2) | CPL(1)} }}},
  { PAPI_TLB_TL, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_TOT_INS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, 2,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      { COUNTER(8) | COUNTER(9), 
	HYPERTHREAD_ANY | ESCR(1) | ENABLE, 
	EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1,
    {{{ COUNTER(4), 
	HYPERTHREAD_ANY | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), 
	CPL(1)} }}},
  { PAPI_L1_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L1_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE} }}},
  { PAPI_L1_DCM, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L2_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L2_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE} }}},
  { PAPI_L2_DCM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { 0, NULL, }
};

const P4_search_t _papi_hwd_pentium4_mge2_preset_map[] = {
  { PAPI_RES_STL, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x9) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_BR_INS, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_BR_TKN, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xc) | CPL(1)} }}},
  { PAPI_BR_NTK, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_BR_MSP, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xa) | CPL(1)} }}},
  { PAPI_BR_PRC, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x5) | CPL(1)} }}},
  { PAPI_TLB_DM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_TLB_IM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x2) | CPL(1)} }}},
  { PAPI_TLB_TL, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_TOT_INS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_TOT_IIS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, 2,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      { COUNTER(8) | COUNTER(9), 
	HYPERTHREAD_ANY | ESCR(1) | ENABLE, 
	EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1,
    {{{ COUNTER(4), 
	HYPERTHREAD_ANY | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), 
	CPL(1)} }}},
  { PAPI_L1_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16),  
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L1_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16),  
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE} }}},
  { PAPI_L1_DCM, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L1_DCA, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG , PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L2_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L2_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE} }}},
  { PAPI_L2_DCM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { 0, NULL, }
};

