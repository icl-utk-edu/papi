/* 
* File:    papi_preset.c
* CVS:     
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#include "papi.h"
#include SUBSTRATE
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"

hwi_preset_t _papi_hwi_preset_map[PAPI_MAX_PRESET_EVENTS] = {{ 0 }};

int _papi_hwi_setup_all_presets(preset_search_t *findem)
{
  int pnum,did_something = 0,pmc;
  int preset_index;
  char *name;

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++){

      /* dense array of events is terminated with a 0 preset */
    if (findem[pnum].preset == 0)
	  break;

    preset_index = findem[pnum].preset & PRESET_AND_MASK; 
      
      /* will change for derived event in near future */
      /* _papi_hwi_preset_map[preset_index].operation=NULL;*/

      /* The same block (below) is executed for derived and non-derived events.
	 Typically non-derived events will only have a single term and 
	 derived events will have multiple terms.
	 In some cases, i.e. Pentium 4 floating point, it may be possible
	 to have non-derived events with multiple terms, where one term acts
	 as a preconditioner for the term that's actually counted.
      */
    pmc=0;
    _papi_hwi_preset_map[preset_index].derived=findem[pnum].derived;
    while((findem[pnum].natEvent[pmc] > 0) && (pmc < MAX_COUNTER_TERMS)){
	  _papi_hwi_preset_map[preset_index].metric_count++;
	  _papi_hwi_preset_map[preset_index].natIndex[pmc]=findem[pnum].natEvent[pmc] ^ NATIVE_MASK;
	  name = _papi_hwd_ntv_code_to_name(findem[pnum].natEvent[pmc]);
	  if (strlen(_papi_hwi_preset_map[preset_index].note)+strlen(name)+1 < PAPI_MAX_STR_LEN){
	    if (pmc) strcat(_papi_hwi_preset_map[preset_index].note,", ");
	    strcat(_papi_hwi_preset_map[preset_index].note,name);
	  }
	  pmc++;
    }
    did_something++;
  }
  return(did_something ? 0 : PAPI_ESBSTR);
}

int _papi_hwi_preset_query(int preset_index, int *flags, char **note)
{
  int events;

  events = _papi_hwi_preset_map[preset_index].metric_count;

  if (events == 0)
    return(0);
  if (_papi_hwi_preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (_papi_hwi_preset_map[preset_index].note)
    *note = _papi_hwi_preset_map[preset_index].note;
  return(1);
}




