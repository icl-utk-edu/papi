/* 
* File:    papi_fwrappers.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Nils Smeds
*          smeds@pdc.kth.se
*          Anders Nilsson
*          anni@pdc.kth.se
*	       Kevin London
*	       london@cs.utk.edu
*	       dan terpstra
*	       terpstra@cs.utk.edu
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
#define PAPI_FCALL(function,caps,args) void function##_##args
#elif FORTRANDOUBLEUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##__##args
#elif FORTRANCAPS
#define PAPI_FCALL(function,caps,args) void caps##args
#else
#define PAPI_FCALL(function,caps,args) void function##args
#endif

/* Many Unix systems passes Fortran string lengths as extra arguments */
/* Compaq Visual Fortran on Windows also supports this convention */
#if defined(__i386__) || defined(_AIX) || defined(sun) || defined(mips) || defined(_WIN32)
#define _FORTRAN_STRLEN_AT_END
#endif
/* The Low Level Wrappers */

PAPI_FCALL(papif_accum,PAPIF_ACCUM,(int *EventSet, long_long *values, int *check))
{
  *check = PAPI_accum(*EventSet, values);
}

PAPI_FCALL(papif_add_event,PAPIF_ADD_EVENT,(int *EventSet, int *Event, int *check))
{
  *check = PAPI_add_event(EventSet, *Event);
}

PAPI_FCALL(papif_add_events,PAPIF_ADD_EVENTS,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_add_events(EventSet, Events, *number);
}

PAPI_FCALL(papif_cleanup_eventset,PAPIF_CLEANUP_EVENTSET,(int *EventSet, int *check))
{
  *check = PAPI_cleanup_eventset(EventSet);
}

PAPI_FCALL(papif_library_init,PAPIF_LIBRARY_INIT,(int *check))
{
  *check = PAPI_library_init(*check);
}

/* This must be passed an EXTERNAL or INTRINISIC FUNCTION not a SUBROUTINE */

PAPI_FCALL(papif_thread_init,PAPIF_THREAD_INIT,(unsigned long int (*handle)(void), int *flag, int *check))
{
  *check = PAPI_thread_init(handle, *flag);
}

PAPI_FCALL(papif_list_events,PAPIF_LIST_EVENTS,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_list_events(*EventSet, Events, number);
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_perror,PAPIF_PERROR,(int *code, _fcd destination_fcd, int *check))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_perror,PAPIF_PERROR,(int *code, char *destination_str, int *check, 
				      int destination_len))
#else
PAPI_FCALL(papif_perror,PAPIF_PERROR,(int *code, char *destination, int *check))
#endif
{
#if defined( _CRAYT3E ) || defined(_FORTRAN_STRLEN_AT_END)
#if defined( _CRAYT3E )
  int destination_len=_fcdlen(destination_fcd);
  char *destination_str=_fcdtocp(destination_fcd);
#endif
  int i;
  char tmp[PAPI_MAX_STR_LEN];

  *check = PAPI_perror(*code, tmp, PAPI_MAX_STR_LEN);
  /* tmp has \0 within PAPI_MAX_STR_LEN chars so strncpy is safe */
  strncpy(destination_str,tmp,destination_len);
  /* overwrite any NULLs and trailing garbage in destinaion_str */
  for(i=strlen(tmp);i<destination_len;destination_str[i++]=' ');
#else
  /* Assume that the underlying Fortran implementation 
     can handle \0 terminated strings and that the 
     passed array is of sufficient size */
  *check = PAPI_perror(*code, destination, PAPI_MAX_STR_LEN);
#endif
}

PAPI_FCALL(papif_query_event,PAPIF_QUERY_EVENT,(int *EventCode, int *check))
{
  *check = PAPI_query_event(*EventCode);
}
 
#if defined ( _CRAYT3E )
PAPI_FCALL(papif_event_code_to_name,PAPIF_EVENT_CODE_TO_NAME,(int *EventCode, _fcd out_fcd, int *check))
#elif defined(_FORTRAN_STRLEN_AT_END)  
PAPI_FCALL(papif_event_code_to_name,PAPIF_EVENT_CODE_TO_NAME,(int *EventCode, char *out_str, int *check,
							      int out_len))
#else
PAPI_FCALL(papif_event_code_to_name,PAPIF_EVENT_CODE_TO_NAME,(int *EventCode, char *out, int *check))
#endif
{
#if defined( _CRAYT3E ) || defined( _FORTRAN_STRLEN_AT_END )
#if defined( _CRAYT3E )
  char *out_str=_fcdtocp(out_fcd);
  int  out_len=_fcdlen(out_fcd);
#endif
  char tmp[PAPI_MAX_STR_LEN];
  int i;
  *check = PAPI_event_code_to_name(*EventCode, tmp);
  /* tmp has \0 within PAPI_MAX_STR_LEN chars so strncpy is safe */
  strncpy(out_str,tmp,out_len);
  /* overwrite any NULLs and trailing garbage in out_str */
  for(i=strlen(tmp);i<out_len;out_str[i++]=' ');
#else
  /* The array "out" passed by the user must be sufficiently long */
  *check = PAPI_event_code_to_name(*EventCode, out);
#endif
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_event_name_to_code,PAPIF_EVENT_NAME_TO_CODE,(_fcd in_fcd, int *out, int *check))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_event_name_to_code,PAPIF_EVENT_NAME_TO_CODE,(char *in_str, int *out, int *check,
							      int in_len))
#else
PAPI_FCALL(papif_event_name_to_code,PAPIF_EVENT_NAME_TO_CODE,(char *in, int *out, int *check))
#endif
{
#if defined( _CRAYT3E ) || defined( _FORTRAN_STRLEN_AT_END )
#if defined( _CRAYT3E )
  int in_len=_fcdlen(in_fcd);    /* Get the string and length */
  char *in_str=_fcdtocp(in_fcd);
#endif
  int slen,i;
  char tmpin[PAPI_MAX_STR_LEN];

  /* What is the maximum number of chars to copy ? */
  slen = in_len < PAPI_MAX_STR_LEN ? in_len : PAPI_MAX_STR_LEN ;
  strncpy( tmpin, in_str, slen );

  /* Remove trailing blanks from initial Fortran string */
  for(i=slen-1;i>-1 && tmpin[i]==' ';tmpin[i--]='\0');

  /* Make sure string is NULL terminated before call*/
  tmpin[PAPI_MAX_STR_LEN-1]='\0';   
  if(slen<PAPI_MAX_STR_LEN) tmpin[slen]='\0';
  
  *check= PAPI_event_name_to_code(tmpin, out);
#else
  /* This will have trouble if argument in is not null terminated */
  *check= PAPI_event_name_to_code(in, out);
#endif
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_describe_event,PAPIF_DESCRIBE_EVENT,(_fcd name_fcd, int *EventCode, _fcd descr_fcd, int *check, int name_len))
#elif defined(_FORTRAN_STRLEN_AT_END)  
PAPI_FCALL(papif_describe_event,PAPIF_DESCRIBE_EVENT,(char *name_str, int *EventCode, char *descr_str, int *check,
		                int name_len, int descr_len))
#else
PAPI_FCALL(papif_describe_event,PAPIF_DESCRIBE_EVENT,(char *name, int *EventCode, char *descr, int *check))
#endif
{
#if defined( _CRAYT3E ) || defined( _FORTRAN_STRLEN_AT_END )
#if defined( _CRAYT3E )
  char *name_str=_fcdtocp(name_fcd), *descr_str=_fcdtocp(descr_fcd);
  int   out_len=_fcdlen(descr_fcd), descr_len=_fcdlen(descr_fcd);
#endif
  char tmpname[PAPI_MAX_STR_LEN], tmpdescr[PAPI_MAX_STR_LEN];
  int i,slen;

  /* What is the maximum number of chars to copy ? */
  slen = name_len < PAPI_MAX_STR_LEN ? name_len : PAPI_MAX_STR_LEN ;
  strncpy( tmpname, name_str, slen );
  /* Remove trailing blanks from initial Fortran string */
  for(i=slen-1;i>-1 && tmpname[i]==' ';tmpname[i--]='\0');
  /* Make sure string is NULL terminated before call*/
  tmpname[PAPI_MAX_STR_LEN-1]='\0';   
  if(slen<PAPI_MAX_STR_LEN) tmpname[slen]='\0';

  *check = PAPI_describe_event(tmpname,EventCode,tmpdescr);
  /* tmp has \0 within PAPI_MAX_STR_LEN chars so strncpy is safe */
  strncpy(name_str,tmpname,name_len);
  strncpy(descr_str,tmpdescr,descr_len);
  /* overwrite any NULLs and trailing garbage in out_str */
  for(i=strlen(tmpname);i<name_len;name_str[i++]=' ');
  for(i=strlen(tmpdescr);i<descr_len;descr_str[i++]=' ');
#else
  /* The arrays passed by the user must be sufficiently long */
  *check = PAPI_describe_event(name,EventCode,descr);
#endif
}

PAPI_FCALL(papif_read,PAPIF_READ,(int *EventSet, long_long *values, int *check))
{
  *check = PAPI_read(*EventSet, values);
}

PAPI_FCALL(papif_rem_event,PAPIF_REM_EVENT,(int *EventSet, int *Event, int *check))
{
  *check = PAPI_rem_event(EventSet, *Event);
}

PAPI_FCALL(papif_rem_events,PAPIF_REM_EVENTS,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_rem_events(EventSet, Events, *number);
}

PAPI_FCALL(papif_reset,PAPIF_RESET,(int *EventSet, int *check))
{
  *check = PAPI_reset(*EventSet);
}

PAPI_FCALL(papif_set_debug,PAPIF_SET_DEBUG,(int *debug, int *check))
{
  *check = PAPI_set_debug(*debug);
}

PAPI_FCALL(papif_set_domain,PAPIF_SET_DOMAIN,(int *domain, int *check))
{
  *check = PAPI_set_domain(*domain);
}

PAPI_FCALL(papif_set_granularity,PAPIF_SET_GRANULARITY,(int *granularity, int *check))
{
  *check = PAPI_set_granularity(*granularity);
}

PAPI_FCALL(papif_start,PAPIF_START,(int *EventSet, int *check))
{
  *check = PAPI_start(*EventSet);
}

PAPI_FCALL(papif_state,PAPIF_STATE,(int *EventSet, int *status, int *check))
{
  *check = PAPI_state(*EventSet, status);
}

PAPI_FCALL(papif_stop,PAPIF_STOP,(int *EventSet, long_long *values, int *check))
{
  *check = PAPI_stop(*EventSet, values);
}

PAPI_FCALL(papif_write,PAPIF_WRITE,(int *EventSet, long_long *values, int *check))
{
  *check = PAPI_write(*EventSet, values);
}

PAPI_FCALL(papif_shutdown,PAPIF_SHUTDOWN,(void))
{
  PAPI_shutdown();
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_get_hardware_info,PAPIF_GET_HARDWARE_INFO,(int *ncpu, 
	   int *nnodes, int *totalcpus, int *vendor, _fcd vendor_fcd, 
	   int *model, _fcd model_fcd, double *revision, double *mhz))
#elif defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL(papif_get_hardware_info,PAPIF_GET_HARDWARE_INFO,(int *ncpu, 
	   int *nnodes, int *totalcpus, int *vendor, char *vendor_str, 
	   int *model, char *model_str, float *revision, float *mhz,
	   int vendor_len, int model_len))
#else
PAPI_FCALL(papif_get_hardware_info,PAPIF_GET_HARDWARE_INFO,(int *ncpu, 
	   int *nnodes, int *totalcpus, int *vendor, char *vendor_string, 
	   int *model, char *model_string, float *revision, float *mhz))
#endif
{
  const PAPI_hw_info_t *hwinfo;
#if defined(_FORTRAN_STRLEN_AT_END)
  int i;
#elif defined(_CRAYT3E)
  int i;
  char *vendor_str=_fcdtocp(vendor_fcd), *model_str=_fcdtocp(model_fcd);
  int vendor_len=_fcdlen(vendor_fcd), model_len=_fcdlen(model_fcd);
#endif
  hwinfo = PAPI_get_hardware_info();
  if ( hwinfo == NULL ){
    *ncpu = 0;
    *nnodes = 0;
    *totalcpus = 0;
    *vendor = 0;
    *model = 0;
    *revision=0;
    *mhz=0;
  }
  else {
    *ncpu = hwinfo->ncpu;
    *nnodes = hwinfo->nnodes;
    *totalcpus = hwinfo->totalcpus;
    *vendor = hwinfo->vendor;
    *model = hwinfo->model;
    *revision = hwinfo->revision;
    *mhz = hwinfo->mhz;
#if defined ( _CRAYT3E ) ||  defined(_FORTRAN_STRLEN_AT_END)
    strncpy( vendor_str, hwinfo->vendor_string,vendor_len);
    for(i=strlen(hwinfo->vendor_string);i<vendor_len;vendor_str[i++]=' ') ;
    strncpy( model_str, hwinfo->model_string, model_len);
    for(i=strlen(hwinfo->model_string);i<model_len;model_str[i++]=' ') ;
#else
    /* This case needs the passed strings to be of sufficient size *
     * and will include the NULL character in the target string    */
    strcpy( vendor_string, hwinfo->vendor_string );
    strcpy( model_string, hwinfo->model_string );
#endif    
  }
  return;
}

PAPI_FCALL(papif_create_eventset,PAPIF_CREATE_EVENTSET,(int *EventSet, int *check))
{
  *check = PAPI_create_eventset(EventSet);
}

PAPI_FCALL(papif_destroy_eventset,PAPIF_DESTROY_EVENTSET,(int *EventSet, int *check))
{
  *check = PAPI_destroy_eventset(EventSet);
}

/* The High Level API Wrappers */

PAPI_FCALL(papif_num_counters,PAPIF_NUM_COUNTERS,(int *numevents))
{
  *numevents = PAPI_num_counters();
}
 
PAPI_FCALL(papif_flops, PAPIF_FLOPS, ( float *real_time, float *proc_time, long_long *flpins, float *mflops, int *check )) 
{
  *check = PAPI_flops( real_time, proc_time, flpins, mflops);
}

PAPI_FCALL(papif_start_counters,PAPIF_START_COUNTERS,(int *events, int *array_len, int *check))
{
  *check = PAPI_start_counters(events, *array_len);
}

PAPI_FCALL(papif_read_counters,PAPIF_READ_COUNTERS,(long_long *values, int *array_len, int *check))
{
  *check = PAPI_read_counters(values, *array_len);
}

PAPI_FCALL(papif_accum_counters,PAPIF_ACCUM_COUNTERS,(long_long *values, int *array_len, int *check))
{
  *check = PAPI_accum_counters(values, *array_len);
}

PAPI_FCALL(papif_stop_counters,PAPIF_STOP_COUNTERS,(long_long *values, int *array_len, int *check))
{
  *check = PAPI_stop_counters(values, *array_len);
}

PAPI_FCALL(papif_get_real_usec,PAPIF_GET_REAL_USEC,( long_long *time))
{
  *time = PAPI_get_real_usec();
}

PAPI_FCALL(papif_get_real_cyc,PAPIF_GET_REAL_CYC,(long_long *real_cyc))
{
  *real_cyc = PAPI_get_real_cyc();
}

PAPI_FCALL(papif_get_virt_usec,PAPIF_GET_VIRT_USEC,( long_long *time))
{
  *time = PAPI_get_virt_usec();
}

PAPI_FCALL(papif_get_virt_cyc,PAPIF_GET_VIRT_CYC,(long_long *virt_cyc))
{
  *virt_cyc = PAPI_get_virt_cyc();
}
