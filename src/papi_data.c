/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi_data.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
* Mods:    Kevin London
*	   london@cs.utk.edu
* Mods:    Per Ekman
*          pek@pdc.kth.se
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_protos.h"
#include "papi_data.h"

/********************/
/*  BEGIN GLOBALS   */
/********************/

#include "papi_data.h"

EventSetInfo_t *default_master_eventset = NULL;
int init_retval = DEADBEEF;
int init_level = PAPI_NOT_INITED;
int papi_num_substrates = 0;
#ifdef DEBUG
int _papi_hwi_debug = 0;
#endif

/* Machine dependent info structure */
papi_mdi_t _papi_hwi_system_info;
papi_substrate_mdi_t * _papi_hwi_substrate_info=NULL;

unsigned int _papi_hwi_preset_count[PAPI_MAX_PRESET_EVENTS] = {0};
hwi_preset_data_t *_papi_hwi_preset_data[PAPI_MAX_PRESET_EVENTS] = {NULL};
char *_papi_hwi_dev_notes[PAPI_MAX_PRESET_EVENTS] = {0};
hwi_presets_t _papi_hwi_presets = {
   _papi_hwi_preset_count,
   _papi_hwi_preset_info,
   _papi_hwi_preset_data,
   _papi_hwi_dev_notes
};


/* table matching derived types to derived strings.
   used by get_info, encode_event, xml translator
*/
const hwi_describe_t _papi_hwi_derived[] = {
   {NOT_DERIVED,     "NOT_DERIVED",    "Do nothing"},
   {DERIVED_ADD,     "DERIVED_ADD",    "Add counters"},
   {DERIVED_PS,      "DERIVED_PS",     "Divide by the cycle counter and convert to seconds"},
   {DERIVED_ADD_PS,  "DERIVED_ADD_PS", "Add 2 counters then divide by the cycle counter and xl8 to secs."},
   {DERIVED_CMPD,    "DERIVED_CMPD",   "Event lives in first counter but takes 2 or more codes"},
   {DERIVED_SUB,     "DERIVED_SUB",    "Sub all counters from first counter"},
   {DERIVED_POSTFIX, "DERIVED_POSTFIX", "Process counters based on specified postfix string"},
   {-1, "", ""}
};

/* _papi_hwi_derived_type:
   Helper routine to extract a derived type from a derived string
   returns type value if found, otherwise returns -1
*/
int _papi_hwi_derived_type(char *derived) {
   int j;

   for(j = 0; _papi_hwi_derived[j].value != -1; j++)
      if (!strcmp (derived, _papi_hwi_derived[j].name)) break; /* match */
   return(_papi_hwi_derived[j].value);
}

/* _papi_hwi_derived_string:
   Helper routine to extract a derived string from a derived type
   copies derived type string into derived if found,
   otherwise returns PAPI_EINVAL
*/
int _papi_hwi_derived_string(int type, char *derived, int len) {
   int j;

   for(j = 0; _papi_hwi_derived[j].value != -1; j++) {
      if (_papi_hwi_derived[j].value == type) {
         strncpy(derived, _papi_hwi_derived[j].name, len);
         return(PAPI_OK);
      }
   }
   return(PAPI_EINVAL);
}

/* papi_sizeof:
   Helper routine to return the size of hardware dependent data structures.
   These sizes are stored into the substrate info structure by the substrate
   at initialization.
*/
int papi_sizeof(int type, int idx){
   switch(type){
     case HWD_CONTEXT:
       return(_papi_hwi_substrate_info[idx].context_size);
     case HWD_REGISTER:
       return(_papi_hwi_substrate_info[idx].register_size); 
     case HWD_REG_ALLOC:
       return(_papi_hwi_substrate_info[idx].reg_alloc_size);
     case HWD_CONTROL_STATE:
       return(_papi_hwi_substrate_info[idx].control_state_size);
     default:
       return(0);
   }
   return(0);
}


/********************/
/*    END GLOBALS   */
/********************/
