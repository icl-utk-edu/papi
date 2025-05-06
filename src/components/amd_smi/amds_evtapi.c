/**
 * @file    amds_evtapi.c
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#include "amds.h"
#include "amds_priv.h"
#include "htable.h"
#include "papi.h"
#include <string.h>
#include <stdio.h>

/* Event enumeration: iterate over native events */
int amds_evt_enum(unsigned int *EventCode, int modifier) {
  if (!EventCode) {
    return PAPI_EINVAL;
  }
  if (!ntv_table_p) {
    return PAPI_ECMP;
  }

  if (modifier == PAPI_ENUM_FIRST) {
    if (ntv_table_p->count == 0) {
      return PAPI_ENOEVNT;
    }
    *EventCode = 0;
    return PAPI_OK;
  } else if (modifier == PAPI_ENUM_EVENTS) {
    if (*EventCode >= (unsigned int)ntv_table_p->count) {
      return PAPI_ENOEVNT;
    }
    if (*EventCode + 1 < (unsigned int)ntv_table_p->count) {
      *EventCode = *EventCode + 1;
      return PAPI_OK;
    } else {
      return PAPI_ENOEVNT;
    }
  }
  return PAPI_EINVAL;
}

int amds_evt_code_to_name(unsigned int EventCode, char *name, int len) {
  if (!name || len <= 0) {
    return PAPI_EINVAL;
  }
  if (!ntv_table_p) {
    return PAPI_ECMP;
  }
  if (EventCode >= (unsigned int)ntv_table_p->count) {
    return PAPI_EINVAL;
  }
  snprintf(name, (size_t)len, "%s", ntv_table_p->events[EventCode].name);
  return PAPI_OK;
}

int amds_evt_name_to_code(const char *name, unsigned int *EventCode) {
  if (!name || !EventCode) {
    return PAPI_EINVAL;
  }
  if (!htable) {
    return PAPI_ECMP;
  }
  native_event_t *event = NULL;
  int hret = htable_find(htable, name, (void **)&event);
  if (hret != HTABLE_SUCCESS) {
    return (hret == HTABLE_ENOVAL) ? PAPI_ENOEVNT : PAPI_ECMP;
  }
  *EventCode = event->id;
  return PAPI_OK;
}

int amds_evt_code_to_descr(unsigned int EventCode, char *descr, int len) {
  if (!descr || len <= 0) {
    return PAPI_EINVAL;
  }
  if (!ntv_table_p) {
    return PAPI_ECMP;
  }
  if (EventCode >= (unsigned int)ntv_table_p->count) {
    return PAPI_EINVAL;
  }
  snprintf(descr, (size_t)len, "%s", ntv_table_p->events[EventCode].descr);
  return PAPI_OK;
}
