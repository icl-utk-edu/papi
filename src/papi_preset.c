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

/* This routine copies values from a dense 'findem' array of events into the sparse
   global _papi_hwi_presets array, which is assumed to be empty at initialization. 
   Multiple dense arrays can be copied into the sparse array, allowing event overloading
   at run-time, or allowing a baseline table to be augmented by a model specific table
   at init time. This method supports adding new events; overriding existing events, or
   deleting deprecated events.
*/
int _papi_hwi_setup_all_presets(hwi_search_t * findem, hwi_dev_notes_t *notes)
{
   int i, pnum, preset_index, did_something = 0;

   /* dense array of events is terminated with a 0 preset.
      don't do anything if NULL pointer. This allows just notes to be loaded.
      It's also good defensive programming. 
   */
   if (findem != NULL) {
      for (pnum = 0; (pnum < PAPI_MAX_PRESET_EVENTS) && (findem[pnum].event_code != 0);
         pnum++) {

         /* copy a pointer to the data into the sparse preset data array */
         /* NOTE: this assumes the data is *static* inside the substrate! */
         preset_index = (findem[pnum].event_code & PAPI_PRESET_AND_MASK);
         _papi_hwi_presets.data[preset_index] = &findem[pnum].data;

         /* count and set the number of native terms in this event */
         for (i = 0; (i < MAX_COUNTER_TERMS) && (findem[pnum].data.native[i] != PAPI_NULL); i++);
         _papi_hwi_presets.count[preset_index] = i;

         /* if the native event array is empty, clear the data pointer.
            this allows existing events to be 'undefined' by overloading with nulls */
         if (i == 0) _papi_hwi_presets.data[preset_index] = NULL;
         did_something++;
      }
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
