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

#include "papi_internal.h"

#include "papi_protos.h"

/* Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example. 

   The third field is meant to be the 'read selector' but it is not implemented. */

const P4_search_t _papi_hwd_pentium4_preset_map[] = {
  { PAPI_TOT_INS, NULL, 1<<0xC, 1<<0xC, 1,
    {{{COUNTER(0x0C) | FAST_RDPMC, REQUIRED | ESCR(4) | ENABLE, EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, (1<<0xC)|(1<<0x8), 1<<0xC, 2,
    {{{COUNTER(0x0C) | FAST_RDPMC, REQUIRED | ESCR(5) | ENABLE, EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      {COUNTER(0x08) | FAST_RDPMC, REQUIRED | ESCR(1) | ENABLE, EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1<<0xD, 1<<0xD, 1,
    {{{COUNTER(0xD) | FAST_RDPMC, REQUIRED | ESCR(4) | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), EVENT(0x3f) | CPL(1)} }}},
  { 0, NULL, }
};

