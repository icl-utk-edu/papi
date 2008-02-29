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

typedef struct hwi_preset_info { /* descriptive text information for each preset */
   char *symbol;      /* name of the preset event; i.e. PAPI_TOT_INS, etc. */
   char *short_descr; /* short description of the event for labels, etc. */
   char *long_descr;  /* long description (full sentence) */
} hwi_preset_info_t;

typedef struct hwi_preset_data {  /* preset event data for each defined preset */
   int derived;                   /* Derived type code */
/* Unused but should be to prevent checking native against PAPI_NULL:
   unsigned int mask;
   unsigned int count; */
   int native[PAPI_MAX_COUNTER_TERMS];    /* array of native event code(s) for this preset event */
   char operation[PAPI_MIN_STR_LEN]; /* operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
} hwi_preset_data_t;

typedef struct hwi_search {   /* search element for preset events defined for each platform */
  /* eventcode should have a more specific name, like papi_preset! -pjm */
   unsigned int event_code;   /* Preset code that keys back to sparse preset array */
   hwi_preset_data_t data;    /* Event data for this preset event */
} hwi_search_t;

typedef struct hwi_dev_notes {
   unsigned int event_code;   /* Preset code that keys back to sparse preset array */
   char *dev_note;          /* optional developer notes for this event */
} hwi_dev_notes_t;

typedef struct hwi_presets {  /* collected text and data info for all preset events */
   unsigned int count[PAPI_MAX_PRESET_EVENTS];       /* array of number of terms in this event. 0 = no event */
   const hwi_preset_info_t *info;   /* array of descriptive text for all events */
   const unsigned int *type;						/* array of event types for all events */
   hwi_preset_data_t *data[PAPI_MAX_PRESET_EVENTS];  /* sparse array of pointers to event data including native terms, etc. */
   char *dev_note[PAPI_MAX_PRESET_EVENTS];           /* sparse array of pointers to optional developer note strings */
} hwi_presets_t;

/* This is a general description structure definition for various parameter lists */   
typedef struct hwi_describe {
   int value;                 /* numeric value (from papi.h) */
   char *name;                /* name of the element */
   char *descr;               /* description of the element */
} hwi_describe_t;

extern hwi_search_t *preset_search_map;

#endif                          /* _PAPI_PRESET */
