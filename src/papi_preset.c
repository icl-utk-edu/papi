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

/* Defined in papi_data.c */
extern PAPI_event_info_t _papi_hwi_presets[];

hwi_preset_data_t _papi_hwi_preset_data[PAPI_MAX_PRESET_EVENTS] = { {0} };

int _papi_hwi_setup_all_presets(hwi_search_t * findem)
{
   int pnum, did_something = 0, pmc;
   int preset_index;
   char *str;
   hwi_preset_data_t *data;

   /* make sure every native event array is terminated */
   for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++) {
      _papi_hwi_preset_data[pnum].native[0] = PAPI_NULL;
   }

   /* dense array of events is terminated with a 0 preset */
   for (pnum = 0; (pnum < PAPI_MAX_PRESET_EVENTS) && (findem[pnum].event_code != 0);
        pnum++) {

      preset_index = (findem[pnum].event_code & PAPI_PRESET_AND_MASK);
      _papi_hwi_preset_data[preset_index] = findem[pnum].data;

      /* The same block (below) is executed for derived and non-derived events.
         Typically non-derived events will only have a single term and 
         derived events will have multiple terms.
         In some cases, i.e. Pentium 4 floating point, it may be possible
         to have non-derived events with multiple terms, where one term acts
         as a preconditioner for the term that's actually counted.
       */
      pmc = 0;
      data = &findem[pnum].data;
      _papi_hwi_presets[preset_index].vendor_name[0] = '\0';
      _papi_hwi_presets[preset_index].vendor_descr[0] = '\0';
      while ((data->native[pmc] != PAPI_NULL) && (pmc < MAX_COUNTER_TERMS)) {
         str = _papi_hwd_ntv_code_to_name(data->native[pmc]);
         if (strlen(_papi_hwi_presets[preset_index].vendor_name) + strlen(str) + 1 <
             PAPI_MAX_STR_LEN) {
            if (pmc)
               strcat(_papi_hwi_presets[preset_index].vendor_name, ", ");
            strcat(_papi_hwi_presets[preset_index].vendor_name, str);
         }
         str = _papi_hwd_ntv_code_to_descr(data->native[pmc]);
         if (strlen(_papi_hwi_presets[preset_index].vendor_descr) + strlen(str) + 1 <
             PAPI_HUGE_STR_LEN) {
            if (pmc)
               strcat(_papi_hwi_presets[preset_index].vendor_descr, ", ");
            strcat(_papi_hwi_presets[preset_index].vendor_descr, str);
         }
         pmc++;
      }
      did_something++;
   }
   return (did_something ? 0 : PAPI_ESBSTR);
}
