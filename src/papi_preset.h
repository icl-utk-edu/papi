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


#define OPS MAX_COUNTER_TERMS*5

typedef struct hwi_preset_info { /* descriptive text information for each preset */
   char *symbol;      /* name of the preset event; i.e. PAPI_TOT_INS, etc. */
   char *short_descr; /* short description of the event for labels, etc. */
   char *long_descr;  /* long description (full sentence) */
} hwi_preset_info_t;

typedef struct hwi_preset_data {  /* preset event data for each defined preset */
   int derived;                   /* Derived type code */
   int native[MAX_COUNTER_TERMS]; /* array of native event code(s) for this preset event */
   char operation[OPS];           /* operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
} hwi_preset_data_t;

typedef struct hwi_search {   /* search element for preset events defined for each platform */
   unsigned int event_code;   /* Preset code that keys back to sparse preset array */
   hwi_preset_data_t data;    /* Event data for this preset event */
} hwi_search_t;

typedef struct hwi_dev_notes {
   unsigned int event_code;   /* Preset code that keys back to sparse preset array */
   char *dev_note;          /* optional developer notes for this event */
} hwi_dev_notes_t;

typedef struct hwi_presets {  /* collected text and data info for all preset events */
   unsigned int *count;       /* array of number of terms in this event. 0 = no event */
   hwi_preset_info_t *info;   /* array of descriptive text for all events */
   hwi_preset_data_t **data;  /* sparse array of pointers to event data including native terms, etc. */
   char **dev_note;           /* sparse array of pointers to optional developer note strings */
} hwi_presets_t;

typedef struct hwi_derived_info {
   int type;                  /* derived type (from papi.h) */
   char *name;                /* name of the derived type */
   char *descr;               /* description of the derived type */
} hwi_derived_info_t;

extern hwi_search_t *preset_search_map;
extern int _papi_hwi_setup_all_presets(hwi_search_t * preset_search_map, hwi_dev_notes_t *notes);
#ifdef XML
extern int _xml_papi_hwi_setup_all_presets(char *arch, hwi_dev_notes_t *notes);
#endif

#endif                          /* _PAPI_PRESET */
