#include "papi.h"
#include <stdio.h>

#ifdef FORTRANUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##_##args
#elif FORTRANDOUBLEUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##__##args
#elif FORTRANCAPS
#define PAPI_FCALL(function,caps,args) void caps##args
#else
#define PAPI_FCALL(function,caps,args) void function##args
#endif

PAPI_FCALL(papif_get_preload, PAPIF_GET_PRELOAD, (char *lib_preload_env, int *check))
{
  PAPI_option_t p;

  if ((*check = PAPI_get_opt(PAPI_GET_PRELOAD, &p))==PAPI_OK){
    strncpy(lib_preload_env, p.preload.lib_preload_env, PAPI_MAX_STR_LEN);
  }
}

#if 0
PAPI_FCALL(papif_get_inherit, PAPIF_GET_INHERIT, (int *inherit, int *check))
{
  PAPI_option_t i;

  if ((*check = PAPI_get_opt(PAPI_GET_INHERIT, &i))==PAPI_OK){
    *inherit = i.inherit.inherit;
  }
}
#endif

PAPI_FCALL(papif_get_granularity, PAPIF_GET_GRANULARITY, 
           (int *eventset, int *granularity, int *mode, int *check))
{
  PAPI_option_t g;

  if (*mode == PAPI_GET_DEFGRN){
    *granularity = PAPI_get_opt(*mode, &g);
    *check = PAPI_OK;
  }
  else if (*mode == PAPI_GET_GRANUL){
    g.granularity.eventset = *eventset;
    if ((*check = PAPI_get_opt(*mode, &g))==PAPI_OK){
      *granularity = g.granularity.granularity;
    }
  }
  else{
    *check = PAPI_EINVAL;
  }
}

PAPI_FCALL(papif_get_domain, PAPIF_GET_DOMAIN, (int *eventset, int *domain, int *mode, int *check))
{
  PAPI_option_t d;

  if (*mode == PAPI_GET_DEFDOM){
    *domain = PAPI_get_opt(*mode, NULL);
    *check = PAPI_OK;
  }
  else if(*mode == PAPI_GET_DOMAIN){
    d.domain.eventset = *eventset;
    if ((*check = PAPI_get_opt(*mode, &d))==PAPI_OK){
      *domain = d.domain.domain;
    }
  }
  else{
    *check = PAPI_EINVAL;
  }
}

PAPI_FCALL(papif_get_exe_info, PAPIF_GET_EXE_INFO, (char *fullname, char *name, int *text_start, int *text_end, 
           int *data_start, int *data_end, int *bss_start, int *bss_end, char *lib_preload_env, int *check))
{
  PAPI_option_t e;
  
  if ((*check = PAPI_get_opt(PAPI_GET_EXEINFO, &e))==PAPI_OK){
    strncpy(fullname, e.exe_info->fullname, PAPI_MAX_STR_LEN);
    strncpy(name, e.exe_info->name, PAPI_MAX_STR_LEN);
    *text_start = (int)e.exe_info->text_start;
    *text_end = (int)e.exe_info->text_end;
    *data_start = (int)e.exe_info->data_start;
    *data_end = (int)e.exe_info->data_end;
    *bss_start = (int)e.exe_info->bss_start;
    *bss_end = (int)e.exe_info->bss_end;
    strncpy(lib_preload_env, e.exe_info->lib_preload_env, PAPI_MAX_STR_LEN);
  }
}

#if 0
PAPI_FCALL(papif_set_inherit, PAPIF_SET_INHERIT, (int *inherit, int *check))
{
  PAPI_option_t i;

  i.inherit.inherit = *inherit;
  *check = PAPI_set_opt(PAPI_SET_INHERIT, &i);
}
#endif

PAPI_FCALL(papif_set_domain1, PAPIF_SET_DOMAIN1, (int *es, int *domain, int *check))
{
  PAPI_option_t d;

  d.domain.domain = *domain;
  d.domain.eventset = *es;
  *check = PAPI_set_opt(PAPI_SET_DOMAIN, &d);
}

PAPI_FCALL(papif_profil, PAPIF_PROFIL, (unsigned short *buf, unsigned *bufsiz, unsigned long *offset, unsigned *scale, unsigned *eventset, 
           unsigned *eventcode, unsigned *threshold, unsigned *flags, unsigned *check))
{
  *check = PAPI_profil(buf, *bufsiz, *offset, *scale, *eventset, *eventcode, *threshold, *flags);
}

PAPI_FCALL(papif_get_clockrate, PAPIF_GET_CLOCKRATE, (int *cr))
{
  *cr = PAPI_get_opt(PAPI_GET_CLOCKRATE, NULL);
}

PAPI_FCALL(papif_query_event_verbose, PAPIF_QUERY_EVENT_VERBOSE, (unsigned int *EventCode, char* event_name, char *event_descr, char *event_label, int *avail, char *event_note, int *flags, unsigned *check))
{
  PAPI_preset_info_t info;
  
  if ((*check = PAPI_query_event_verbose(*EventCode, &info))==PAPI_OK){
    strncpy(event_name, info.event_name, PAPI_MAX_STR_LEN);
    strncpy(event_descr, info.event_descr, PAPI_MAX_STR_LEN);
    strncpy(event_label, info.event_label, PAPI_MAX_STR_LEN);
    *avail = info.avail;
    strncpy(event_note, info.event_note, PAPI_MAX_STR_LEN);
    *flags = info.flags;
  }
}

PAPI_FCALL(papif_query_all_event_verbose, PAPIF_QUERY_ALL_EVENT_VERBOSE, (unsigned int *EventCode, char* event_name, char *event_descr, char *event_label, int *avail, char *event_note, int *flags, unsigned *check))
{
  const PAPI_preset_info_t *info;
  int index;
  
  if((info = PAPI_query_all_events_verbose())==NULL){
    *check = PAPI_EINVAL;
    return;
  }

  index = *EventCode - 0x80000000;

  if (!info[index].event_name) *check = PAPI_ENOEVNT;
  else {
    strncpy(event_name, info[index].event_name, PAPI_MAX_STR_LEN);
    if(info[index].event_descr)
      strncpy(event_descr, info[index].event_descr, PAPI_MAX_STR_LEN);
    else
      strncpy(event_descr, "",  PAPI_MAX_STR_LEN);
    if(info[index].event_label)
      strncpy(event_label, info[index].event_label, PAPI_MAX_STR_LEN);
    else
      strncpy(event_label, "",  PAPI_MAX_STR_LEN);
    *avail = info[index].avail;
    if(info[index].event_note)
      strncpy(event_note, info[index].event_note, PAPI_MAX_STR_LEN);
    else
      strncpy(event_note, "",  PAPI_MAX_STR_LEN);
    *flags = info[index].flags;
    *check = PAPI_OK;
  }
}
