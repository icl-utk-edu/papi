/** 
* @file    papi_preset.h
* @author  Haihang You
*          you@cs.utk.edu
*/

#ifndef _PAPI_PRESET		 /* _PAPI_PRESET */
#define _PAPI_PRESET

/** descriptive text information for each preset 
 *	@internal */
typedef struct hwi_preset_info { 
   char *symbol;      /**< name of the preset event; i.e. PAPI_TOT_INS, etc. */
   char *short_descr; /**< short description of the event for labels, etc. */
   char *long_descr;  /**< long description (full sentence) */
} hwi_preset_info_t;

/** preset event data for each defined preset 
 *	@internal */
typedef struct hwi_preset_data { 
   int derived;                   /**< Derived type code */
/* Unused but should be to prevent checking native against PAPI_NULL:
   unsigned int mask;
   unsigned int count; */
   int native[PAPI_MAX_COUNTER_TERMS];    /**< array of native event code(s) for this preset event */
#ifdef __bgp__
   char operation[PAPI_2MAX_STR_LEN]; /**< operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
#else
   char operation[PAPI_MIN_STR_LEN]; /**< operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
#endif
} hwi_preset_data_t;

/** search element for preset events defined for each platform 
 *	@internal */
typedef struct hwi_search {   
  /* eventcode should have a more specific name, like papi_preset! -pjm */
   unsigned int event_code;   /**< Preset code that keys back to sparse preset array */
   hwi_preset_data_t data;    /**< Event data for this preset event */
} hwi_search_t;

/** @internal */
typedef struct hwi_dev_notes {
   unsigned int event_code;   /**< Preset code that keys back to sparse preset array */
   char *dev_note;          /**< optional developer notes for this event */
} hwi_dev_notes_t;

/** collected text and data info for all preset events 
 *	@internal */
typedef struct hwi_presets {  
   unsigned int count[PAPI_MAX_PRESET_EVENTS];       /**< array of number of terms in this event. 0 = no event */
   const hwi_preset_info_t *info;   /**< array of descriptive text for all events */
   const unsigned int *type;						/**< array of event types for all events */
   hwi_preset_data_t *data[PAPI_MAX_PRESET_EVENTS];  /**< sparse array of pointers to event data including native terms, etc. */
   char *dev_note[PAPI_MAX_PRESET_EVENTS];           /**< sparse array of pointers to optional developer note strings */
} hwi_presets_t;

/** This is a general description structure definition for various parameter lists 
 *	@internal */   
typedef struct hwi_describe {
   int value;                 /**< numeric value (from papi.h) */
   char *name;                /**< name of the element */
   char *descr;               /**< description of the element */
} hwi_describe_t;

extern hwi_search_t *preset_search_map;

int _papi_hwi_setup_all_presets( hwi_search_t * findem,
                                 hwi_dev_notes_t * notes,
                                 int cidx);
int _papi_hwi_cleanup_all_presets( void );
int _xml_papi_hwi_setup_all_presets( char *arch, hwi_dev_notes_t * notes );

#endif /* _PAPI_PRESET */
