/* 
* File:    papi_preset.h
* CVS:     
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#ifndef _PAPI_PRESET            /* _PAPI_PRESET */
#define _PAPI_PRESET


#define OPS MAX_COUNTERS*5

typedef struct hwi_preset_data {
   int derived;                 /* Derived code */
   int native[MAX_COUNTER_TERMS];       /* array of native event code(s) for this preset event */
   char operation[OPS];         /* operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
} hwi_preset_data_t;

typedef struct hwi_search {
   unsigned int event_code;     /* Preset code that keys back to sparse preset array */
   hwi_preset_data_t data;      /* Event data for this preset event */
} hwi_search_t;

extern hwi_search_t *preset_search_map;
extern int _papi_hwi_setup_all_presets(hwi_search_t * preset_search_map);

#endif                          /* _PAPI_PRESET */
