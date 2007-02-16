/* 
* File:    p3_events.c
* CVS:     $Id$
* Author:  Joseph Thomas
*          jthomas@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

native_event_entry_t *native_table;
hwi_search_t *preset_search_map;

/*  This file serves as a collection point for architecture specific
    information and for code common to all related architectures in the
    pentium family of processors. Architecture-specific information can
    be found in one of the files included below. These files include
    enumeration lists of native events, PAPI preset event maps, and native
    event tables taken from vendor documentation. All of these files are
    included at compile time to allow run-time selection of the required
    architecture.
*/

//#include "p3_core_event_tables.h"
//#include "p3_opt_event_tables.h"
//#include "p3_pm_event_tables.h"
//#include "p3_p3_event_tables.h"
#include "p3_ath_event_tables.h"
#include "p3_p2_event_tables.h"


/* Note:  MESI (Intel) and MOESI (AMD) bits are programmatically defined
          for those events that can support them. You can find those
          events in the appropriate architecture-specific file by
	  searching for HAS_MESI or HAS_MOESI. Events containing all
	  possible combinations of these bits can be formed by appending
	  the proper letters to the end of the event name, e.g. L2_LD_MESI
	  or L2_ST_MI. The user can access MOESI conditioned native events
          using _papi_hwd_name_to_code() with the proper bit characters
          appended to the event name.
*/


/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

/* **NOT THREAD SAFE STATIC!!**
   The name and description strings below are both declared static. 
   This is NOT thread-safe, because these values are returned 
     for use by the calling program, and could be trashed by another thread
     before they are used. To prevent this, any call to routines using these
     variables (_papi_hwd_code_to_{name,descr}) should be wrapped in 
     _papi_hwi_{lock,unlock} calls.
   They are declared static to reserve non-volatile space for constructed strings.
*/
static char name[128];
static char description[1024];

static void mask2hex(int umask, char *hex)
{
    char digits[] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};

    strcpy(hex,"0x");
    hex[2] = digits[(umask >> 12) & 0xF];
    hex[3] = digits[(umask >> 8) & 0xF];
    hex[4] = '\0';
}

inline_static void internal_decode_event(unsigned int EventCode, int *event, int *umask)
{
   /* mask off the native event flag and the MOESI or unit mask bits */
   *event = (EventCode & PAPI_NATIVE_AND_MASK) & (~UNIT_MASK_ALL);
   /* find the unit mask bits (if any) */
   *umask = (EventCode & UNIT_MASK_ALL);
}

/* Called by _papi_hwd_ntv_code_to_{name,descr}() to build the return strings.
   See above for discussion of name and description strings.
*/
static char *internal_translate_code(int event, int umask, char *str, char *separator)
{
    int selector = native_table[event].resources.selector;
    char hex[8];

   if (*separator == '_') /* implied flag for name */
      strcpy(str, native_table[event].name);
   else
      strcpy(str, native_table[event].description);

   // if no mask bits, we're done
   if (!umask)
      return (str);

   // go here if the event supports unit mask bits
   if (selector & HAS_UMASK) {
      if ((umask & selector & UNIT_MASK_ALL) != umask)
	 return (NULL);

       if (*separator == '_') strcat(str, ":");
       else strcat(str, separator);
       mask2hex(umask, hex);
       strcat(str, hex);
   }

   // end up here if it's a MOESI event
   else if ((selector & HAS_MESI) || (selector & HAS_MOESI)) {
      // do a sanity check for valid mask bits
      if ((umask & MOESI_ALL) != umask)
	 return (NULL);

       strcat(str, separator);
       if (*separator == '_') { /* implied flag for name */
	  if (umask & MOESI_M) strcat(str, "M");
	  if (umask & MOESI_O) {
	     if (native_table[event].resources.selector & HAS_MOESI) strcat(str, "O");
	     else strcat(str, "M");
	  }
	  if (umask & MOESI_E) strcat(str, "E");
	  if (umask & MOESI_S) strcat(str, "S");
	  if (umask & MOESI_I) strcat(str, "I");
       }
       else {
	  if (umask & MOESI_M) strcat(str, " Modified");
	  if (umask & MOESI_O) {
	     if (native_table[event].resources.selector & HAS_MOESI) strcat(str, " Owner");
	     else strcat(str, " Modified");
	  }
	  if (umask & MOESI_E) strcat(str, " Exclusive");
	  if (umask & MOESI_S) strcat(str, " Shared");
	  if (umask & MOESI_I) strcat(str, " Invalid");
       }
   }
   return (str);
}


/* Given a native event code, returns the short text label. */
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   int event, umask;

   internal_decode_event(EventCode, &event, &umask);
   if (event > _papi_hwi_system_info.sub_info.num_native_events) {
       return ('\0'); // return a null string for invalid events
   }

   if (!umask)
      return (native_table[event].name);
   else {
      return (internal_translate_code(event, umask, name, "_"));
   }
}

/* Given a native event code, returns the longer native event
   description. */
char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   int event, umask;

   internal_decode_event(EventCode, &event, &umask);
   if (event > _papi_hwi_system_info.sub_info.num_native_events)
       return ('\0'); // return a null string for invalid events

   if (!umask)
      return (native_table[event].description);
   else {
      if (native_table[event].resources.selector & HAS_UMASK)
	 return (internal_translate_code(event, umask, description, ". Unit Mask bits: "));
      else
	 return (internal_translate_code(event, umask, description, ". Cache bits:"));
   }
}

/* Given a native event code, assigns the native event's 
   information to a given pointer.
   NOTE: the info must be COPIED to the provided pointer,
   not just referenced!
*/
int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   int event, umask;

   internal_decode_event(EventCode, &event, &umask);
   if (event > _papi_hwi_system_info.sub_info.num_native_events)
       return (PAPI_ENOEVNT);

   if(native_table[event].resources.selector == 0) {
      return (PAPI_ENOEVNT);
   }
   *bits = native_table[event].resources;
   bits->counter_cmd |= umask; /* OR unit mask bits into command */
   return (PAPI_OK);
}

/* Given a native event code, looks for next MOESI or umask bit if applicable.
   If not, looks for the next event in the table if the next one exists. 
   If not, returns the proper error code. */
int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   int event, umask, selector;

   internal_decode_event(*EventCode, &event, &umask);
   if (event > _papi_hwi_system_info.sub_info.num_native_events)
       return (PAPI_ENOEVNT);

   /* increment by smallest step size (same for unit mask or MOESI */
   umask += MOESI_I;
   selector = native_table[event].resources.selector;

   /* Check for unit mask bits */
   if (selector & HAS_UMASK) {
       while (umask <= (selector & UNIT_MASK_ALL)) {
	  if (umask == (umask & selector & UNIT_MASK_ALL)) {
	     *EventCode = (event | PAPI_NATIVE_MASK) + umask;
	     return (PAPI_OK);
	  }
       }
   }

    /* Check for MOESI bits */
   /* for AMD processors, 5 bits are valid */
   if (selector & HAS_MOESI) {
      if (umask <= MOESI_ALL) {
         *EventCode = (event | PAPI_NATIVE_MASK) + umask;
         return (PAPI_OK);
      }
   }
   /* for Intel processors, only 4 bits are valid */
   else if (selector & HAS_MESI) {
      if (!(umask & MOESI_M)) { /* never set top bit */
         *EventCode = (event | PAPI_NATIVE_MASK) + umask;
         return (PAPI_OK);
      }
   }
   if (native_table[event + 1].resources.selector) {
      *EventCode = (event | PAPI_NATIVE_MASK) + 1;
      return (PAPI_OK);
   } else {
      return (PAPI_ENOEVNT);
   }
}

/* Reports the elements of the hwd_register_t struct as an array of names and a matching array of values.
   Maximum string length is name_len; Maximum number of values is count.
*/
static void copy_value(unsigned int val, char *nam, char *names, unsigned int *values, int len)
{
   *values = val;
   strncpy(names, nam, len);
   names[len-1] = 0;
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
   int i = 0;
   copy_value(bits->selector, "Event Mask", &names[i*name_len], &values[i], name_len);
   if (++i == count) return(i);
   copy_value(bits->counter_cmd, "Event Code", &names[i*name_len], &values[i], name_len);
   return(++i);
}

papi_svector_t _papi_p3_event_vectors[] = {
  {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
  {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
  {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
  {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
  {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {NULL, VEC_PAPI_END}
};


/*****************************************************/
/* CODE TO SUPPORT CUSTOMIZABLE FP COUNTS ON OPTERON */
/*****************************************************/

#if defined(PAPI_OPTERON_FP_RETIRED)
   #define AMD_FPU "RETIRED"
#elif defined(PAPI_OPTERON_FP_SSE_SP)
   #define AMD_FPU "SSE_SP"
#elif defined(PAPI_OPTERON_FP_SSE_DP)
   #define AMD_FPU "SSE_DP"
#else
   #define AMD_FPU "SPECULATIVE"
#endif

extern int _papi_pfm_setup_presets(char *name, int type);

int _papi_hwd_fixup_fp(void)
{
   char table_name[PAPI_MIN_STR_LEN] = "AMD64 FPU ";
   char *str = getenv("PAPI_OPTERON_FP");

   /* if the env variable isn't set, return the defaults */
   if ((str == NULL) || (strlen(str) == 0)) {
      strcat(table_name, AMD_FPU);
   } else {
      strcat(table_name, str);
   }

   if((_papi_pfm_setup_presets(table_name, 0)) != PAPI_OK) {
      PAPIERROR("Improper usage of PAPI_OPTERON_FP environment variable.\nUse one of RETIRED, SPECULATIVE, SSE_SP, SSE_DP");
      return(PAPI_ESBSTR);
   }
      return(PAPI_OK);
}

