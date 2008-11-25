/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi_fwrappers.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Nils Smeds
*          smeds@pdc.kth.se
*          Anders Nilsson
*          anni@pdc.kth.se
*	   Kevin London
*	   london@cs.utk.edu
*	   dan terpstra
*	   terpstra@cs.utk.edu
*          Min Zhou
*	   min@cs.utk.edu
*/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#if defined ( _CRAYT3E )
#include  <stdlib.h>
#include  <fortran.h>
#endif
#include "papi.h"

/* Lets use defines to rename all the files */

#ifdef FORTRANUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##_ args
#elif FORTRANDOUBLEUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##__ args
#elif FORTRANCAPS
#define PAPI_FCALL(function,caps,args) void caps args
#else
#define PAPI_FCALL(function,caps,args) void function args
#endif

/* Many Unix systems passes Fortran string lengths as extra arguments */
/* Compaq Visual Fortran on Windows also supports this convention */
#if defined ( __crayx1 ) || defined(_AIX) || defined(sun) || defined(mips) || defined(_WIN32) || defined(linux) || ( defined(__ALPHA) && defined(__osf__)) 
#define _FORTRAN_STRLEN_AT_END
#endif
/* The Low Level Wrappers */

PAPI_FCALL(papif_accum, PAPIF_ACCUM, (int *EventSet, long long * values, int *check))
{
   *check = PAPI_accum(*EventSet, values);
}

PAPI_FCALL(papif_add_event, PAPIF_ADD_EVENT, (int *EventSet, int *Event, int *check))
{
   *check = PAPI_add_event(*EventSet, *Event);
}

PAPI_FCALL(papif_add_events, PAPIF_ADD_EVENTS,
           (int *EventSet, int *Events, int *number, int *check))
{
   *check = PAPI_add_events(*EventSet, Events, *number);
}

PAPI_FCALL(papif_cleanup_eventset, PAPIF_CLEANUP_EVENTSET, (int *EventSet, int *check))
{
   *check = PAPI_cleanup_eventset(*EventSet);
}

PAPI_FCALL(papif_create_eventset, PAPIF_CREATE_EVENTSET, (int *EventSet, int *check))
{
   *check = PAPI_create_eventset(EventSet);
}

PAPI_FCALL(papif_assign_eventset_component, PAPIF_ASSIGN_EVENTSET_COMPONENT, (int *EventSet, int *cidx, int *check))
{
   *check = PAPI_assign_eventset_component(*EventSet, *cidx);
}

PAPI_FCALL(papif_destroy_eventset, PAPIF_DESTROY_EVENTSET, (int *EventSet, int *check))
{
   *check = PAPI_destroy_eventset(EventSet);
}

PAPI_FCALL(papif_get_dmem_info, PAPIF_GET_DMEM_INFO, (long long *dest, int *check))
{
  *check = PAPI_get_dmem_info((PAPI_dmem_info_t *)dest);
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_get_exe_info, PAPIF_GET_EXE_INFO,
           (_fcd fullname_fcd, _fcd name_fcd, long long * text_start,
            long long * text_end, long long * data_start, long long * data_end,
            long long * bss_start, long long * bss_end, _fcd lib_preload_env_fcd,
            int *check))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_get_exe_info, PAPIF_GET_EXE_INFO,
           (char *fullname, char *name, long long * text_start, long long * text_end,
            long long * data_start, long long * data_end, long long * bss_start,
            long long * bss_end, char *lib_preload_env, int *check, int fullname_len,
            int name_len, int lib_preload_env_len))
#else
PAPI_FCALL(papif_get_exe_info, PAPIF_GET_EXE_INFO,
           (char *fullname, char *name, long long * text_start, long long * text_end,
            long long * data_start, long long * data_end, long long * bss_start,
            long long * bss_end, int *check))
#endif
{
   PAPI_option_t e;
/* WARNING: The casts from caddr_t to long below WILL BREAK on systems with
    64-bit addresses. I did it here because I was lazy. And because I wanted
    to get rid of those pesky gcc warnings. If you find a 64-bit system,
    conditionalize the cast with (yet another) #ifdef...
*/
#if defined( _CRAYT3E ) || defined(_FORTRAN_STRLEN_AT_END)
#if defined( _CRAYT3E )
   int fullname_len = _fcdlen(fullname_fcd);
   char *fullname = _fcdtocp(fullname_fcd);
   int name_len = _fcdlen(name_fcd);
   char *name = _fcdtocp(name_fcd);
   int lib_preload_env_len = _fcdlen(lib_preload_env_fcd);
   char *lib_preload_env = _fcdtocp(lib_preload_env_fcd);
#endif
   int i;
   if ((*check = PAPI_get_opt(PAPI_EXEINFO, &e)) == PAPI_OK) {
      strncpy(fullname, e.exe_info->fullname, fullname_len);
      for (i = strlen(e.exe_info->fullname); i < fullname_len; fullname[i++] = ' ');
      strncpy(name, e.exe_info->address_info.name, name_len);
      for (i = strlen(e.exe_info->address_info.name); i < name_len; name[i++] = ' ');
      *text_start = (long) e.exe_info->address_info.text_start;
      *text_end = (long) e.exe_info->address_info.text_end;
      *data_start = (long) e.exe_info->address_info.data_start;
      *data_end = (long) e.exe_info->address_info.data_end;
      *bss_start = (long) e.exe_info->address_info.bss_start;
      *bss_end = (long) e.exe_info->address_info.bss_end;
   }
#else
   if ((*check = PAPI_get_opt(PAPI_EXEINFO, &e)) == PAPI_OK) {
      strncpy(fullname, e.exe_info->fullname, PAPI_MAX_STR_LEN);
      strncpy(name, e.exe_info->address_info.name, PAPI_MAX_STR_LEN);
      *text_start = (long) (e.exe_info->address_info.text_start);
      *text_end = (long) e.exe_info->address_info.text_end;
      *data_start = (long) e.exe_info->address_info.data_start;
      *data_end = (long) e.exe_info->address_info.data_end;
      *bss_start = (long) e.exe_info->address_info.bss_start;
      *bss_end = (long) e.exe_info->address_info.bss_end;
   }
#endif
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_get_hardware_info, PAPIF_GET_HARDWARE_INFO, (int *ncpu,
                                                          int *nnodes, int *totalcpus,
                                                          int *vendor, _fcd vendor_fcd,
                                                          int *model, _fcd model_fcd,
                                                          double *revision, double *mhz))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_get_hardware_info, PAPIF_GET_HARDWARE_INFO, (int *ncpu,
                                                          int *nnodes, int *totalcpus,
                                                          int *vendor, char *vendor_str,
                                                          int *model, char *model_str,
                                                          float *revision, float *mhz,
                                                          int vendor_len, int model_len))
#else
PAPI_FCALL(papif_get_hardware_info, PAPIF_GET_HARDWARE_INFO, (int *ncpu,
                                                          int *nnodes, int *totalcpus,
                                                          int *vendor,
                                                          char *vendor_string, int *model,
                                                          char *model_string,
                                                          float *revision, float *mhz))
#endif
{
   const PAPI_hw_info_t *hwinfo;
#if defined(_FORTRAN_STRLEN_AT_END)
   int i;
#elif defined(_CRAYT3E)
   int i;
   char *vendor_str = _fcdtocp(vendor_fcd), *model_str = _fcdtocp(model_fcd);
   int vendor_len = _fcdlen(vendor_fcd), model_len = _fcdlen(model_fcd);
#endif
   hwinfo = PAPI_get_hardware_info();
   if (hwinfo == NULL) {
      *ncpu = 0;
      *nnodes = 0;
      *totalcpus = 0;
      *vendor = 0;
      *model = 0;
      *revision = 0;
      *mhz = 0;
   } else {
      *ncpu = hwinfo->ncpu;
      *nnodes = hwinfo->nnodes;
      *totalcpus = hwinfo->totalcpus;
      *vendor = hwinfo->vendor;
      *model = hwinfo->model;
      *revision = hwinfo->revision;
      *mhz = hwinfo->mhz;
#if defined ( _CRAYT3E ) ||  defined(_FORTRAN_STRLEN_AT_END)
      strncpy(vendor_str, hwinfo->vendor_string, vendor_len);
      for (i = strlen(hwinfo->vendor_string); i < vendor_len; vendor_str[i++] = ' ');
      strncpy(model_str, hwinfo->model_string, model_len);
      for (i = strlen(hwinfo->model_string); i < model_len; model_str[i++] = ' ');
#else
      /* This case needs the passed strings to be of sufficient size *
       * and will include the NULL character in the target string    */
      strcpy(vendor_string, hwinfo->vendor_string);
      strcpy(model_string, hwinfo->model_string);
#endif
   }
   return;
}

PAPI_FCALL(papif_num_hwctrs, PAPIF_num_hwctrs, (int *num))
{
   *num = PAPI_num_hwctrs();
}

PAPI_FCALL(papif_num_cmp_hwctrs, PAPIF_num_cmp_hwctrs, (int *cidx, int *num))
{
   *num = PAPI_num_cmp_hwctrs(*cidx);
}

PAPI_FCALL(papif_get_real_cyc, PAPIF_GET_REAL_CYC, (long long * real_cyc))
{
   *real_cyc = PAPI_get_real_cyc();
}

PAPI_FCALL(papif_get_real_usec, PAPIF_GET_REAL_USEC, (long long * time))
{
   *time = PAPI_get_real_usec();
}

PAPI_FCALL(papif_get_virt_cyc, PAPIF_GET_VIRT_CYC, (long long * virt_cyc))
{
   *virt_cyc = PAPI_get_virt_cyc();
}

PAPI_FCALL(papif_get_virt_usec, PAPIF_GET_VIRT_USEC, (long long * time))
{
   *time = PAPI_get_virt_usec();
}

PAPI_FCALL(papif_is_initialized, PAPIF_IS_INITIALIZED, (int *level))
{
   *level = PAPI_is_initialized();
}

PAPI_FCALL(papif_library_init, PAPIF_LIBRARY_INIT, (int *check))
{
   *check = PAPI_library_init(*check);
}

PAPI_FCALL(papif_thread_id, PAPIF_THREAD_ID, (unsigned long *id))
{
   *id = PAPI_thread_id();
}

PAPI_FCALL(papif_register_thread, PAPIF_REGISTER_THREAD, (int *check))
{
  *check = PAPI_register_thread();
}

PAPI_FCALL(papif_unregster_thread, PAPIF_UNREGSTER_THREAD, (int *check))
{
  *check = PAPI_unregister_thread();
}

/* This must be passed an EXTERNAL or INTRINISIC FUNCTION not a SUBROUTINE */

PAPI_FCALL(papif_thread_init, PAPIF_THREAD_INIT,
           (unsigned long int (*handle) (void), int *check))
{
   *check = PAPI_thread_init(handle);
}

PAPI_FCALL(papif_list_events, PAPIF_LIST_EVENTS,
           (int *EventSet, int *Events, int *number, int *check))
{
   *check = PAPI_list_events(*EventSet, Events, number);
}

PAPI_FCALL(papif_multiplex_init, PAPIF_MULTIPLEX_INIT, (int *check))
{
   *check = PAPI_multiplex_init();
}

PAPI_FCALL(papif_get_multiplex, PAPIF_GET_MULTIPLEX, (int *EventSet, int *check))
{
   *check = PAPI_get_multiplex(*EventSet);
}

PAPI_FCALL(papif_set_multiplex, PAPIF_SET_MULTIPLEX, (int *EventSet, int *check))
{
   *check = PAPI_set_multiplex(*EventSet);
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_perror, PAPIF_PERROR, (int *code, _fcd destination_fcd, int *check))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_perror, PAPIF_PERROR, (int *code, char *destination_str, int *check,
                                        int destination_len))
#else
PAPI_FCALL(papif_perror, PAPIF_PERROR, (int *code, char *destination, int *check))
#endif
{
#if defined( _CRAYT3E ) || defined(_FORTRAN_STRLEN_AT_END)
#if defined( _CRAYT3E )
   int destination_len = _fcdlen(destination_fcd);
   char *destination_str = _fcdtocp(destination_fcd);
#endif
   int i;
   char tmp[PAPI_MAX_STR_LEN];

   *check = PAPI_perror(*code, tmp, PAPI_MAX_STR_LEN);
   /* tmp has \0 within PAPI_MAX_STR_LEN chars so strncpy is safe */
   strncpy(destination_str, tmp, destination_len);
   /* overwrite any NULLs and trailing garbage in destination_str */
   for (i = strlen(tmp); i < destination_len; destination_str[i++] = ' ');
#else
   /* Assume that the underlying Fortran implementation 
      can handle \0 terminated strings and that the 
      passed array is of sufficient size */
   *check = PAPI_perror(*code, destination, PAPI_MAX_STR_LEN);
#endif
}

/* This will not work until Fortran2000 :)
 * PAPI_FCALL(papif_profil, PAPIF_PROFIL, (unsigned short *buf, unsigned *bufsiz, unsigned long *offset, unsigned *scale, unsigned *eventset, 
 *            unsigned *eventcode, unsigned *threshold, unsigned *flags, unsigned *check))
 * {
 * *check = PAPI_profil(buf, *bufsiz, *offset, *scale, *eventset, *eventcode, *threshold, *flags);
 * }
 */

PAPI_FCALL(papif_query_event, PAPIF_QUERY_EVENT, (int *EventCode, int *check))
{
   *check = PAPI_query_event(*EventCode);
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_get_event_info, PAPIF_GET_EVENT_INFO,
           (unsigned int *EventCode, _fcd symbol_fcd, _fcd long_descr_fcd,
            _fcd short_descr_fcd, int *count, _fcd event_note_fcd, int *flags,
            unsigned *check))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_get_event_info, PAPIF_GET_EVENT_INFO,
           (int *EventCode, char *symbol, char *long_descr, char *short_descr, int *count,
            char *event_note, int *flags, int *check, int symbol_len, int long_descr_len,
            int short_descr_len, int event_note_len))
#else
PAPI_FCALL(papif_get_event_info, PAPIF_GET_EVENT_INFO,
           (int *EventCode, char *symbol, char *long_descr, char *short_descr, int *count,
            char *event_note, int *flags, int *check))
#endif
{
   PAPI_event_info_t info;

#if defined( _CRAYT3E )
   int symbol_len = _fcdlen(symbol_fcd);
   char *symbol = _fcdtocp(symbol_fcd);
   int long_descr_len = _fcdlen(long_descr_fcd);
   char *long_descr = _fcdtocp(long_descr_fcd);
   int short_descr_len = _fcdlen(short_descr_fcd);
   char *short_descr = _fcdtocp(short_descr_fcd);
   int event_note_len = _fcdlen(event_note_fcd);
   char *event_note = _fcdtocp(event_note_fcd);
#endif
#if defined( _CRAYT3E ) || defined(_FORTRAN_STRLEN_AT_END)
   int i;
#endif

#if defined( _CRAYT3E ) || defined(_FORTRAN_STRLEN_AT_END)
   if ((*check = PAPI_get_event_info(*EventCode, &info)) == PAPI_OK) {
      strncpy(symbol, info.symbol, symbol_len);
      for (i = strlen(info.symbol); i < symbol_len; symbol[i++] = ' ');
      strncpy(long_descr, info.long_descr, long_descr_len);
      for (i = strlen(info.long_descr); i < long_descr_len; long_descr[i++] = ' ');
      strncpy(short_descr, info.short_descr, short_descr_len);
      for (i = strlen(info.short_descr); i < short_descr_len; short_descr[i++] = ' ');
      *count = info.count;
      strncpy(event_note, info.note, event_note_len);
      for(i=strlen(info.note);i<event_note_len;event_note[i++]=' ');
   }
#else
/* printf("EventCode: %d\n", *EventCode ); -KSL */
   if ((*check = PAPI_get_event_info(*EventCode, &info)) == PAPI_OK) {
      strncpy(symbol, info.symbol, PAPI_MAX_STR_LEN);
      strncpy(long_descr, info.long_descr, PAPI_MAX_STR_LEN);
      strncpy(short_descr, info.short_descr, PAPI_MAX_STR_LEN);
      *count = info.count;
      strncpy(event_note, info.note, PAPI_MAX_STR_LEN);
   }
/*  printf("Check: %d\n", *check); -KSL */
#endif
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_event_code_to_name, PAPIF_EVENT_CODE_TO_NAME,
           (int *EventCode, _fcd out_fcd, int *check))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_event_code_to_name, PAPIF_EVENT_CODE_TO_NAME,
           (int *EventCode, char *out_str, int *check, int out_len))
#else
PAPI_FCALL(papif_event_code_to_name, PAPIF_EVENT_CODE_TO_NAME,
           (int *EventCode, char *out, int *check))
#endif
{
#if defined( _CRAYT3E ) || defined( _FORTRAN_STRLEN_AT_END )
#if defined( _CRAYT3E )
   char *out_str = _fcdtocp(out_fcd);
   int out_len = _fcdlen(out_fcd);
#endif
   char tmp[PAPI_MAX_STR_LEN];
   int i;
   *check = PAPI_event_code_to_name(*EventCode, tmp);
   /* tmp has \0 within PAPI_MAX_STR_LEN chars so strncpy is safe */
   strncpy(out_str, tmp, out_len);
   /* overwrite any NULLs and trailing garbage in out_str */
   for (i = strlen(tmp); i < out_len; out_str[i++] = ' ');
#else
   /* The array "out" passed by the user must be sufficiently long */
   *check = PAPI_event_code_to_name(*EventCode, out);
#endif
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_event_name_to_code, PAPIF_EVENT_NAME_TO_CODE,
           (_fcd in_fcd, int *out, int *check))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_event_name_to_code, PAPIF_EVENT_NAME_TO_CODE,
           (char *in_str, int *out, int *check, int in_len))
#else
PAPI_FCALL(papif_event_name_to_code, PAPIF_EVENT_NAME_TO_CODE,
           (char *in, int *out, int *check))
#endif
{
#if defined( _CRAYT3E ) || defined( _FORTRAN_STRLEN_AT_END )
#if defined( _CRAYT3E )
   int in_len = _fcdlen(in_fcd);        /* Get the string and length */
   char *in_str = _fcdtocp(in_fcd);
#endif
   int slen, i;
   char tmpin[PAPI_MAX_STR_LEN];

   /* What is the maximum number of chars to copy ? */
   slen = in_len < PAPI_MAX_STR_LEN ? in_len : PAPI_MAX_STR_LEN;
   strncpy(tmpin, in_str, slen);

   /* Remove trailing blanks from initial Fortran string */
   for (i = slen - 1; i > -1 && tmpin[i] == ' '; tmpin[i--] = '\0');

   /* Make sure string is NULL terminated before call */
   tmpin[PAPI_MAX_STR_LEN - 1] = '\0';
   if (slen < PAPI_MAX_STR_LEN)
      tmpin[slen] = '\0';

   *check = PAPI_event_name_to_code(tmpin, out);
#else
   /* This will have trouble if argument in is not null terminated */
   *check = PAPI_event_name_to_code(in, out);
#endif
}

PAPI_FCALL(papif_num_events, PAPIF_NUM_EVENTS, (int *EventCode, int *count))
{
   *count = PAPI_num_events(*EventCode);
}

PAPI_FCALL(papif_enum_event, PAPIF_ENUM_EVENT, (int *EventCode, int *modifier, int *check))
{
   *check = PAPI_enum_event(EventCode, *modifier);
}

PAPI_FCALL(papif_read, PAPIF_READ, (int *EventSet, long long * values, int *check))
{
   *check = PAPI_read(*EventSet, values);
}

PAPI_FCALL(papif_remove_event, PAPIF_REMOVE_EVENT,
           (int *EventSet, int *Event, int *check))
{
   *check = PAPI_remove_event(*EventSet, *Event);
}

PAPI_FCALL(papif_remove_events, PAPIF_REMOVE_EVENTS,
           (int *EventSet, int *Events, int *number, int *check))
{
   *check = PAPI_remove_events(*EventSet, Events, *number);
}

PAPI_FCALL(papif_reset, PAPIF_RESET, (int *EventSet, int *check))
{
   *check = PAPI_reset(*EventSet);
}

PAPI_FCALL(papif_set_debug, PAPIF_SET_DEBUG, (int *debug, int *check))
{
   *check = PAPI_set_debug(*debug);
}

PAPI_FCALL(papif_set_domain, PAPIF_SET_DOMAIN, (int *domain, int *check))
{
   *check = PAPI_set_domain(*domain);
}

PAPI_FCALL(papif_set_cmp_domain, PAPIF_SET_CMP_DOMAIN, (int *domain, int *cidx, int *check))
{
   *check = PAPI_set_cmp_domain(*domain, *cidx);
}

PAPI_FCALL(papif_set_granularity, PAPIF_SET_GRANULARITY, (int *granularity, int *check))
{
   *check = PAPI_set_granularity(*granularity);
}

PAPI_FCALL(papif_set_cmp_granularity, PAPIF_SET_CMP_GRANULARITY, (int *granularity, int *cidx, int *check))
{
   *check = PAPI_set_cmp_granularity(*granularity, *cidx);
}

PAPI_FCALL(papif_shutdown, PAPIF_SHUTDOWN, (void))
{
   PAPI_shutdown();
}

PAPI_FCALL(papif_start, PAPIF_START, (int *EventSet, int *check))
{
   *check = PAPI_start(*EventSet);
}

PAPI_FCALL(papif_state, PAPIF_STATE, (int *EventSet, int *status, int *check))
{
   *check = PAPI_state(*EventSet, status);
}

PAPI_FCALL(papif_stop, PAPIF_STOP, (int *EventSet, long long * values, int *check))
{
   *check = PAPI_stop(*EventSet, values);
}

PAPI_FCALL(papif_write, PAPIF_WRITE, (int *EventSet, long long * values, int *check))
{
   *check = PAPI_write(*EventSet, values);
}

/* The High Level API Wrappers */

PAPI_FCALL(papif_start_counters, PAPIF_START_COUNTERS,
           (int *events, int *array_len, int *check))
{
   *check = PAPI_start_counters(events, *array_len);
}

PAPI_FCALL(papif_read_counters, PAPIF_READ_COUNTERS,
           (long long * values, int *array_len, int *check))
{
   *check = PAPI_read_counters(values, *array_len);
}

PAPI_FCALL(papif_stop_counters, PAPIF_STOP_COUNTERS,
           (long long * values, int *array_len, int *check))
{
   *check = PAPI_stop_counters(values, *array_len);
}

PAPI_FCALL(papif_accum_counters, PAPIF_ACCUM_COUNTERS,
           (long long * values, int *array_len, int *check))
{
   *check = PAPI_accum_counters(values, *array_len);
}

PAPI_FCALL(papif_num_counters, PAPIF_NUM_COUNTERS, (int *numevents))
{
   *numevents = PAPI_num_counters();
}

PAPI_FCALL(papif_ipc, PAPIF_IPC,
           (float *rtime, float *ptime, long long * ins, float *ipc, int *check))
{
   *check = PAPI_ipc(rtime, ptime, ins, ipc);
}

PAPI_FCALL(papif_flips, PAPIF_FLIPS,
           (float *real_time, float *proc_time, long long * flpins, float *mflips,
            int *check))
{
   *check = PAPI_flips(real_time, proc_time, flpins, mflips);
}

PAPI_FCALL(papif_flops, PAPIF_FLOPS,
           (float *real_time, float *proc_time, long long * flpops, float *mflops,
            int *check))
{
   *check = PAPI_flops(real_time, proc_time, flpops, mflops);
}


/* Fortran only APIs for get_opt and set_opt functionality */

PAPI_FCALL(papif_get_clockrate, PAPIF_GET_CLOCKRATE, (int *cr))
{
   *cr = PAPI_get_opt(PAPI_CLOCKRATE, NULL);
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_get_preload, PAPIF_GET_PRELOAD, (_fcd lib_preload_env_fcd, int *check))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_get_preload, PAPIF_GET_PRELOAD,
           (char *lib_preload_env, int *check, int lib_preload_env_len))
#else
PAPI_FCALL(papif_get_preload, PAPIF_GET_PRELOAD, (char *lib_preload_env, int *check))
#endif
{
   PAPI_option_t p;
#if defined( _CRAYT3E ) || defined(_FORTRAN_STRLEN_AT_END)
#if defined( _CRAYT3E )
   int lib_preload_env_len = _fcdlen(lib_preload_env_fcd);
   char *lib_preload_env = _fcdtocp(lib_preload_env_fcd);
#endif
   int i;

   if ((*check = PAPI_get_opt(PAPI_PRELOAD, &p)) == PAPI_OK) {
      strncpy(lib_preload_env, p.preload.lib_preload_env, lib_preload_env_len);
      for (i = strlen(p.preload.lib_preload_env); i < lib_preload_env_len;
           lib_preload_env[i++] = ' ');
   }
#else
   if ((*check = PAPI_get_opt(PAPI_PRELOAD, &p)) == PAPI_OK) {
      strncpy(lib_preload_env, p.preload.lib_preload_env, PAPI_MAX_STR_LEN);
   }
#endif
}

PAPI_FCALL(papif_get_granularity, PAPIF_GET_GRANULARITY,
           (int *eventset, int *granularity, int *mode, int *check))
{
   PAPI_option_t g;

   if (*mode == PAPI_DEFGRN) {
      *granularity = PAPI_get_opt(*mode, &g);
      *check = PAPI_OK;
   } else if (*mode == PAPI_GRANUL) {
      g.granularity.eventset = *eventset;
      if ((*check = PAPI_get_opt(*mode, &g)) == PAPI_OK) {
         *granularity = g.granularity.granularity;
      }
   } else {
      *check = PAPI_EINVAL;
   }
}

PAPI_FCALL(papif_get_domain, PAPIF_GET_DOMAIN,
           (int *eventset, int *domain, int *mode, int *check))
{
   PAPI_option_t d;

   if (*mode == PAPI_DEFDOM) {
      *domain = PAPI_get_opt(*mode, NULL);
      *check = PAPI_OK;
   } else if (*mode == PAPI_DOMAIN) {
      d.domain.eventset = *eventset;
      if ((*check = PAPI_get_opt(*mode, &d)) == PAPI_OK) {
         *domain = d.domain.domain;
      }
   } else {
      *check = PAPI_EINVAL;
   }
}

#if 0
PAPI_FCALL(papif_get_inherit, PAPIF_GET_INHERIT, (int *inherit, int *check))
{
   PAPI_option_t i;

   if ((*check = PAPI_get_opt(PAPI_INHERIT, &i)) == PAPI_OK) {
      *inherit = i.inherit.inherit;
   }
}
#endif

PAPI_FCALL(papif_set_event_domain, PAPIF_SET_EVENT_DOMAIN,
           (int *es, int *domain, int *check))
{
   PAPI_option_t d;

   d.domain.domain = *domain;
   d.domain.eventset = *es;
   *check = PAPI_set_opt(PAPI_DOMAIN, &d);
}

#if 0
PAPI_FCALL(papif_set_inherit, PAPIF_SET_INHERIT, (int *inherit, int *check))
{
   PAPI_option_t i;

   i.inherit.inherit = *inherit;
   *check = PAPI_set_opt(PAPI_INHERIT, &i);
}
#endif
