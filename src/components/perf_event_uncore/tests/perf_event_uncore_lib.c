#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"

char *get_uncore_event(char *event, int size) {

   const PAPI_hw_info_t *hwinfo;

   hwinfo = PAPI_get_hardware_info();
   if ( hwinfo == NULL ) {
	return NULL;
   }

   if (hwinfo->vendor == PAPI_VENDOR_INTEL) {

      if ( hwinfo->cpuid_family == 6) {
	switch(hwinfo->cpuid_model) {
	   case 26:
           case 30:
	   case 31: /* Nehalem */
	   case 46: /* Nehalem EX */
	            strncpy(event,"nhm_unc::UNC_CLK_UNHALTED",size);
		    return event;
	            break;
	   case 37:
	   case 44: /* Westmere */
	   case 47: /* Westmere EX */
	            strncpy(event,"wsm_unc::UNC_CLK_UNHALTED",size);
		    return event;
	            break;

	   case 62: /* Ivy Trail */
	   case 45: /* SandyBridge EP */
	            strncpy(event,"snbep_unc_imc0::UNC_M_CLOCKTICKS",size);
		    return event;
		    break;
	   case 42: /* SandyBridge */
	            strncpy(event,"snb_unc_cbo0::UNC_CLOCKTICKS",size);
		    return event;
		    break;
	   case 58: /* IvyBridge */
	            strncpy(event,"ivb_unc_cbo0::UNC_CLOCKTICKS",size);
		    return event;
		    break;
	   case 63: /*haswell EP*/
		    strncpy(event,"hswep_unc_cbo0::UNC_C_CLOCKTICKS",size);
		    return event;
		    break;

	   case 87: /*Knights Landing*/
			strncpy(event,"knl_unc_imc0::UNC_M_D_CLOCKTICKS",size);
			return event;
			break;
	}
      }
      return NULL;
   }
   else if (hwinfo->vendor == PAPI_VENDOR_AMD) {
      if ( hwinfo->cpuid_family == 21) {
         /* For kernel 3.9 at least */
	 strncpy(event,"DRAM_ACCESSES:ALL",size);
         return event;
      }
      return NULL;
   }

   return NULL;
}
