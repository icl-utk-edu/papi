#ifndef _PAPI_USER_EVENTS_H
#define _PAPI_USER_EVENTS_H

#include "papi_internal.h"

#define PAPI_UE_AND_MASK	0x3FFFFFFF
#define PAPI_UE_MASK		((int)0xC0000000)

#define USER_EVENT_OPERATION_LEN 512

extern  int _papi_user_defined_events_setup(char *);
extern void _papi_cleanup_user_events();

typedef struct {
  unsigned int count;
  int events[PAPI_EVENTS_IN_DERIVED_EVENT];
   char operation[USER_EVENT_OPERATION_LEN]; /**< operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
   char symbol[PAPI_MIN_STR_LEN];
   char *short_desc;
   char *long_desc;
} user_defined_event_t;

extern user_defined_event_t *	_papi_user_events;
extern unsigned int				_papi_user_events_count;

#endif // _PAPI_USER_EVENTS_H
