#include "preset.h"

hwi_preset_t _papi_hwi_preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };

int setup_all_presets(preset_search_t *findem)
{
  int pnum,did_something = 0,pmc,derived;
  int preset_index, nix;

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++){
      /* dense array of events is terminated with a 0 preset */
      if (findem[pnum].preset == 0)
	  	break;

      preset_index = findem[pnum].preset & PRESET_AND_MASK; 
      
	  /* will change for derived event in near future */
	 /* _papi_hwi_preset_map[preset_index].operation=NULL;*/
	  
	  /* If it's not derived */
      if (findem[pnum].derived == 0){
	  	_papi_hwi_preset_map[preset_index].derived=findem[pnum].derived;
	  	_papi_hwi_preset_map[preset_index].metric_count=1;
	  	nix = findem[pnum].natEvent[0] ^ NATIVE_MASK;
	  	_papi_hwi_preset_map[preset_index].natIndex[0]=nix;
	  	strncpy(_papi_hwi_preset_map[preset_index].note,native_table[nix].name, PAPI_MAX_STR_LEN);
	  	did_something++;
	  }
      else { /* derived event */
	  	_papi_hwi_preset_map[preset_index].derived=findem[pnum].derived;
	  	pmc=0;
		while(findem[pnum].natEvent[pmc]){
			_papi_hwi_preset_map[preset_index].metric_count++;
			nix = findem[pnum].natEvent[pmc] ^ NATIVE_MASK;
			_papi_hwi_preset_map[preset_index].natIndex[pmc]=nix;
	      	if (strlen(_papi_hwi_preset_map[preset_index].note)+strlen(native_table[nix].name)+1 < PAPI_MAX_STR_LEN){
		  		strcat(_papi_hwi_preset_map[preset_index].note,native_table[nix].name);
		  		strcat(_papi_hwi_preset_map[preset_index].note,",");
			}
			pmc++;
		}
	  	did_something++;
	  }
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




