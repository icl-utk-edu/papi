/** 
* @file    papi_preset.h
* @author  Haihang You
*          you@cs.utk.edu
*/

#ifndef _PAPI_PRESET		 /* _PAPI_PRESET */
#define _PAPI_PRESET

/** @struct hwi_preset_info 
    @brief descriptive text information for each preset */
typedef struct hwi_preset_info { 
   char *symbol;      /**< name of the preset event; i.e. PAPI_TOT_INS, etc. */
   char *short_descr; /**< short description of the event for labels, etc. */
   char *long_descr;  /**< long description (full sentence) */
} hwi_preset_info_t;

/** @struct hwi_preset_data 
    @brief preset event data for each defined preset */
typedef struct hwi_preset_data { 
   int derived;                   /**< Derived type code */
/* Unused but should be to prevent checking native against PAPI_NULL:
   unsigned int mask;
   unsigned int count; */
   int native[PAPI_MAX_COUNTER_TERMS];    /**< array of native event code(s) for this preset event */
#ifdef _BGP
   char operation[PAPI_2MAX_STR_LEN]; /**< operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
#else
   char operation[PAPI_MIN_STR_LEN]; /**< operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
#endif
} hwi_preset_data_t;

/** @struct hwi_search 
  @brief search element for preset events defined for each platform */
typedef struct hwi_search {   
  /* eventcode should have a more specific name, like papi_preset! -pjm */
   unsigned int event_code;   /**< Preset code that keys back to sparse preset array */
   hwi_preset_data_t data;    /**< Event data for this preset event */
} hwi_search_t;

/** @struct hwi_dev_notes */
typedef struct hwi_dev_notes {
   unsigned int event_code;   /**< Preset code that keys back to sparse preset array */
   char *dev_note;          /**< optional developer notes for this event */
} hwi_dev_notes_t;

/** @struct hwi_presets i
  @brief collected text and data info for all preset events */
typedef struct hwi_presets {  
   unsigned int count[PAPI_MAX_PRESET_EVENTS];       /**< array of number of terms in this event. 0 = no event */
   const hwi_preset_info_t *info;   /**< array of descriptive text for all events */
   const unsigned int *type;						/**< array of event types for all events */
   hwi_preset_data_t *data[PAPI_MAX_PRESET_EVENTS];  /**< sparse array of pointers to event data including native terms, etc. */
   char *dev_note[PAPI_MAX_PRESET_EVENTS];           /**< sparse array of pointers to optional developer note strings */
} hwi_presets_t;

/** @struct hwi_describe 
    @brief This is a general description structure definition for various parameter lists */   
typedef struct hwi_describe {
   int value;                 /**< numeric value (from papi.h) */
   char *name;                /**< name of the element */
   char *descr;               /**< description of the element */
} hwi_describe_t;

extern hwi_search_t *preset_search_map;

#endif /* _PAPI_PRESET */
