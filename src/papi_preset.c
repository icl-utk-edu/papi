/* 
* File:    papi_preset.c
* CVS:     
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

/* Defined in papi_data.c */
extern hwi_presets_t _papi_hwi_presets;

int _papi_hwi_setup_all_presets(hwi_search_t * findem, hwi_dev_notes_t *notes)
{
   int i, pnum, preset_index, did_something = 0;

   /* dense array of events is terminated with a 0 preset */
   for (pnum = 0; (pnum < PAPI_MAX_PRESET_EVENTS) && (findem[pnum].event_code != 0);
        pnum++) {

      /* copy a pointer to the data into the sparse preset data array */
      /* NOTE: this assumes the data is *static* inside the substrate! */
      preset_index = (findem[pnum].event_code & PAPI_PRESET_AND_MASK);
      _papi_hwi_presets.data[preset_index] = &findem[pnum].data;

      /* count and set the number of native terms in this event */
      for (i = 0; (i < MAX_COUNTER_TERMS) && (findem[pnum].data.native[i] != PAPI_NULL); i++);
      _papi_hwi_presets.count[preset_index] = i;

      did_something++;
   }

   /* optional dense array of event notes is terminated with a 0 preset */
   if (notes != NULL) {
      for (pnum = 0; (pnum < PAPI_MAX_PRESET_EVENTS) && (notes[pnum].event_code != 0);
         pnum++) {

         /* copy a pointer to the note string into the sparse preset data array */
         /* NOTE: this assumes the note is *static* inside the substrate! */
         preset_index = (notes[pnum].event_code & PAPI_PRESET_AND_MASK);
         _papi_hwi_presets.dev_note[preset_index] = notes[pnum].dev_note;
      }
   }
   return (did_something ? 0 : PAPI_ESBSTR);
}
