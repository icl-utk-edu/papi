/*
* File:    pe_libpfm4_events.c
* Author:  Vince Weaver vincent.weaver @ maine.edu
* Mods:    Gary Mohr
*          gary.mohr@bull.com
*          Modified the perf_event component to use PFM_OS_PERF_EVENT_EXT mode in libpfm4.
*          This adds several new event masks, including cpu=, u=, and k= which give the user
*          the ability to set cpu number to use or control the domain (user, kernel, or both)
*          in which the counter should be incremented.  These are event masks so it is now 
*          possible to have multiple events in the same event set that count activity from 
*          differennt cpu's or count activity in different domains.
*
* Handle the libpfm4 event interface for the perf_event component
*/

#include <string.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include "papi_libpfm4_events.h"
#include "pe_libpfm4_events.h"
#include "perf_event_lib.h"

#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"
#include "libpfm4/lib/pfmlib_priv.h"

#define NATIVE_EVENT_CHUNK 1024

// used to step through the attributes when enumerating events
static int attr_idx;

/** @class  find_existing_event
 *  @brief  looks up an event, returns it if it exists
 *
 *  @param[in] name 
 *             -- name of the event
 *  @param[in] event_table
 *             -- native_event_table structure
 *
 *  @returns returns offset in array
 *
 */

static int find_existing_event(char *name, 
                               struct native_event_table_t *event_table) {
  SUBDBG("Entry: name: %s, event_table: %p, num_native_events: %d\n", name, event_table, event_table->num_native_events);

  int i,event=PAPI_ENOEVNT;

  _papi_hwi_lock( NAMELIB_LOCK );

  for(i=0;i<event_table->num_native_events;i++) {

    if (!strcmp(name,event_table->native_events[i].allocated_name)) {
      SUBDBG("Found allocated_name: %s, libpfm4_idx: %#x, papi_event_code: %#x\n", 
         event_table->native_events[i].allocated_name, event_table->native_events[i].libpfm4_idx, event_table->native_events[i].papi_event_code);
      event=i;
      break;
    }
  }
  _papi_hwi_unlock( NAMELIB_LOCK );

  SUBDBG("EXIT: returned: %#x\n", event);
  return event;
}


static int pmu_is_present_and_right_type(pfm_pmu_info_t *pinfo, int type) {
//   SUBDBG("ENTER: pinfo: %p, pinfo->is_present: %d, pinfo->type: %#x, type: %#x\n", pinfo, pinfo->is_present, pinfo->type, type);
  if (!pinfo->is_present) {
//	  SUBDBG("EXIT: not present\n");
	  return 0;
  }

  if ((pinfo->type==PFM_PMU_TYPE_UNCORE) && (type&PMU_TYPE_UNCORE)) {
//	  SUBDBG("EXIT: found PFM_PMU_TYPE_UNCORE\n");
	  return 1;
  }
  if ((pinfo->type==PFM_PMU_TYPE_CORE) && (type&PMU_TYPE_CORE)) {
//	  SUBDBG("EXIT: found PFM_PMU_TYPE_CORE\n");
	  return 1;
  }
  if ((pinfo->type==PFM_PMU_TYPE_OS_GENERIC) && (type&PMU_TYPE_OS)) {
//	  SUBDBG("EXIT: found PFM_PMU_TYPE_OS_GENERIC\n");
	  return 1;
  }

//  SUBDBG("EXIT: not right type\n");
  return 0;
}


/** @class  allocate_native_event
 *  @brief  Allocates a native event
 *
 *  @param[in] name
 *             -- name of the event
 *  @param[in] event_idx
 *             -- libpfm4 identifier for the event
 *  @param[in] event_table
 *             -- native event table struct
 *
 *  @returns returns a native_event_t or NULL
 *
 */

static struct native_event_t *allocate_native_event(char *name, int libpfm4_index,
			  struct native_event_table_t *event_table) {

  SUBDBG("ENTER: name: %s, libpfm4_index: %#x, event_table: %p, event_table->pmu_type: %d\n", name, libpfm4_index, event_table, event_table->pmu_type);

  static int last_libpfm4_index=0;
  int nevt_idx;
  int event_num;
  int encode_failed=0;

  pfm_err_t ret;
  char *event_string=NULL;
  char *pmu_name;
  char *event;
  char *masks;
  char fullname[BUFSIZ];
  struct native_event_t *ntv_evt;

  pfm_perf_encode_arg_t perf_arg;
  pfm_event_info_t einfo;
  pfm_pmu_info_t pinfo;

  // if no place to put native events, report that allocate failed
  if (event_table->native_events==NULL) {
     SUBDBG("EXIT: no place to put native events\n");
     return NULL;
  }

  // find out if this event is already known
  event_num=find_existing_event(name, event_table);

  /* add the event to our event table */
  _papi_hwi_lock( NAMELIB_LOCK );

  // if we already know this event name, it was created as part of setting up the preset tables
  // we need to use the event table which is already created
  if (event_num >= 0) {
     nevt_idx = event_num;
     ntv_evt = &(event_table->native_events[event_num]);
  } else {
	  // set to use a new event table (count of used events not bumped until we are sure setting it up does not get an errro)
     nevt_idx = event_table->num_native_events;
     ntv_evt = &(event_table->native_events[nevt_idx]);
  }

  SUBDBG("event_num: %d, nevt_idx: %d, ntv_evt: %p, ntv_evt->attr_idx: %d\n", event_num, nevt_idx, ntv_evt, ntv_evt->attr_idx);

  /* clear the argument and attribute structures */
  memset(&perf_arg,0,sizeof(pfm_perf_encode_arg_t));
  memset(&(ntv_evt->attr),0,sizeof(struct perf_event_attr));

  // set argument structure fields so the encode function can give us what we need
  perf_arg.attr=&ntv_evt->attr;
  perf_arg.fstr=&event_string;

  /* use user provided name of the event to get the perf_event encoding and a fully qualified event string */
  ret = pfm_get_os_event_encoding(name, 
  				  PFM_PLM0 | PFM_PLM3, 
  				  PFM_OS_PERF_EVENT_EXT,
  				  &perf_arg);

  // If the encode function failed, skip processing of the event_string
  if ((ret != PFM_SUCCESS) || (event_string == NULL)) {
	  SUBDBG("encode failed for event: %s, returned: %d\n", name, ret);

	  // we need to remember that this event encoding failed but still create the native event table
	  // the event table is used by the list so we put what we can get into it
	  // but the failure doing the encode causes us to return null to our caller
	  encode_failed = 1;
  }

  // get a copy of the event name and break it up into its parts
  event_string = strdup(name);

  SUBDBG("event_string: %s\n", event_string);

  // get the pmu name, event name and mask list pointers from the event string
  event = strstr (event_string, "::");
  if (event != NULL) {
	*event = 0;     // null terminate pmu name
	event += 2;     // event name follows '::'
	pmu_name = strdup(event_string);
  } else {
	// no pmu name in event string
	pmu_name = malloc(2);
	pmu_name[0] = 0;
	event = event_string;
  }
  masks = strstr (event, ":");
  if (masks != NULL) {
	  *masks = 0;     // null terminate event name
	  masks += 1;     // masks follow :
  } else {
	  masks = "";
  }

  // build event name to find, put a pmu name on it if we have one
  if(strlen(pmu_name) == 0) {
	  sprintf(fullname,"%s", event);
  } else {
	  sprintf(fullname,"%s::%s", pmu_name, event);
  }
  SUBDBG("pmu_name: %s, event: %s, masks: %s, fullname: %s\n", pmu_name, event, masks, fullname);

  // if the libpfm4 index was not provided, try to get one based on the event name passed in.
  if (libpfm4_index == -1) {
	  libpfm4_index = pfm_find_event(fullname);
	  if (libpfm4_index < 0) {
		  free(event_string);
		  free(pmu_name);
		  _papi_hwi_unlock( NAMELIB_LOCK );
		  SUBDBG("EXIT: error from libpfm4 find event\n");
		  return NULL;
	  }
	  SUBDBG("libpfm4_index: %#x\n", libpfm4_index);
  }

	/* libpfm requires the structure be zeroed */
	memset( &einfo, 0, sizeof( pfm_event_info_t ));
	einfo.size = sizeof(pfm_event_info_t);

	// get this events information from libpfm4, if unavailable return event not found
	if ((ret = pfm_get_event_info(libpfm4_index, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
		free(event_string);
		free(pmu_name);
		_papi_hwi_unlock( NAMELIB_LOCK );
		SUBDBG("EXIT: pfm_get_event_info failed with %d\n", ret);
		return NULL;
	}

	// if pmu type is not one supported by this component, return event not found
	memset(&pinfo,0,sizeof(pfm_pmu_info_t));
	pfm_get_pmu_info(einfo.pmu, &pinfo);
	if (pmu_is_present_and_right_type(&pinfo, event_table->pmu_type) == 0) {
		free(event_string);
		free(pmu_name);
		_papi_hwi_unlock( NAMELIB_LOCK );
		SUBDBG("EXIT: PMU not supported by this component: einfo.pmu: %d, PFM_PMU_TYPE_CORE: %d\n", einfo.pmu, PFM_PMU_TYPE_CORE);
		return NULL;
	}

	// if we have a new event code, start back at beginning of attributes
	if (libpfm4_index != last_libpfm4_index) {
		SUBDBG("new native event code: libpfm4_index: %#x, last_libpfm4_index: %#x, attr_idx: %d\n", libpfm4_index, last_libpfm4_index, attr_idx);
		attr_idx = 0;
	}
	last_libpfm4_index = libpfm4_index;

  ntv_evt->allocated_name=strdup(name);
  ntv_evt->component=_pe_libpfm4_get_cidx();
  ntv_evt->pmu=pmu_name;
  ntv_evt->base_name=strdup(event);
  ntv_evt->pmu_plus_name=strdup(fullname);
  ntv_evt->libpfm4_idx=libpfm4_index;
  ntv_evt->attr_idx=attr_idx;
  ntv_evt->users=0;      /* is this needed? */

  free(event_string);

  // the event string returned from the encode call contains all of the legal masks for this event
  // we really want to save just the masks provided in the event string passed to this function
  // so get the pmu name, event name and mask list pointers for the event string passed to us
  event_string = strdup(name);
  SUBDBG("event_string: %s\n", event_string);

  event = strstr (event_string, "::");
  if (event != NULL) {
    *event = 0;     // null terminate pmu name
    event += 2;     // event name follows '::'
  } else {
    event = event_string;
  }
  masks = strstr (event, ":");
  if (masks != NULL) {
	  *masks = 0;     // null terminate event name
	  masks += 1;     // masks follow :
  } else {
	  masks = "";
  }
  SUBDBG("pmu: %s, event: %s, masks: %s\n", ntv_evt->pmu, event, masks);

  // if we have no masks, create native event using native event index of -1 (says no mask)
  int new_event_code;
  if(strlen(masks) == 0) {
	new_event_code = _papi_hwi_native_to_eventcode(_pe_libpfm4_get_cidx(), libpfm4_index, -1, ntv_evt->allocated_name);
  } else {
	// we have a mask, use the correct index into the native events array for the index
	new_event_code = _papi_hwi_native_to_eventcode(_pe_libpfm4_get_cidx(), libpfm4_index, nevt_idx, ntv_evt->allocated_name);
  }
	_papi_hwi_set_papi_event_string((const char *)ntv_evt->allocated_name);
	_papi_hwi_set_papi_event_code(new_event_code, 1);

  ntv_evt->papi_event_code=new_event_code;
  ntv_evt->mask_string=strdup(masks);

  free(event_string);

  SUBDBG("Using %#x as index for %s\n", ntv_evt->libpfm4_idx, fullname);

  ntv_evt->cpu=perf_arg.cpu;
  SUBDBG("event_table->native_events[%d]: %p, cpu: %d, attr.config: 0x%"PRIx64", attr.config1: 0x%"PRIx64", attr.config2: 0x%"PRIx64", attr.type: 0x%"PRIx32", attr.exclude_user: %d, attr.exclude_kernel: %d, attr.exclude_guest: %d\n",
		nevt_idx, &(event_table->native_events[nevt_idx]), ntv_evt->cpu, ntv_evt->attr.config,
		ntv_evt->attr.config1, ntv_evt->attr.config2, ntv_evt->attr.type,
		ntv_evt->attr.exclude_user, ntv_evt->attr.exclude_kernel, ntv_evt->attr.exclude_guest);

  SUBDBG("Event: base_name: %s, mask_string: %s, libpfm4_idx: %#x, papi_event_code: %#x, num_native_events: %d, allocated_native_events: %d\n",
		  ntv_evt->base_name, ntv_evt->mask_string, ntv_evt->libpfm4_idx,
		  ntv_evt->papi_event_code, event_table->num_native_events, event_table->allocated_native_events);

  /* If we've used all of the allocated native events, then allocate more room */
  if (event_table->num_native_events >=
      event_table->allocated_native_events) {

      SUBDBG("Allocating more room for native events (%d %ld)\n",
	    (event_table->allocated_native_events+NATIVE_EVENT_CHUNK),
	    (long)sizeof(struct native_event_t) *
	    (event_table->allocated_native_events+NATIVE_EVENT_CHUNK));

      event_table->native_events=realloc(event_table->native_events,
			   sizeof(struct native_event_t) *
			   (event_table->allocated_native_events+NATIVE_EVENT_CHUNK));
      event_table->allocated_native_events+=NATIVE_EVENT_CHUNK;

      // we got new space so we need to reset the pointer to the correct native event in the new space
      ntv_evt = &(event_table->native_events[nevt_idx]);
   }

  _papi_hwi_unlock( NAMELIB_LOCK );

  // if getting more space for native events failed, report that allocate failed
  if (event_table->native_events==NULL) {
     SUBDBG("EXIT: attempt to get more space for native events failed\n");
     return NULL;
  }

  // if we created a new event, bump the number used
  if (event_num < 0) {
     event_table->num_native_events++;
  }

  if (encode_failed != 0) {
	  SUBDBG("EXIT: encoding event failed\n");
	  return NULL;
  }

  SUBDBG("EXIT: new_event: %p\n", ntv_evt);
  return ntv_evt;
}


/** @class  get_first_event_next_pmu
 *  @brief  return the first available event that's on an active PMU
 *
 *  @returns returns a libpfm event number
 *  @retval PAPI_ENOEVENT  Could not find an event
 *
 */

static int
get_first_event_next_pmu(int pmu_idx, int pmu_type)
{
	SUBDBG("ENTER: pmu_idx: %d, pmu_type: %d\n", pmu_idx, pmu_type);
  int pidx, ret;

  pfm_pmu_info_t pinfo;

  // start looking at the next pmu in the list
  pmu_idx++;

  while(pmu_idx<PFM_PMU_MAX) {

    /* clear the PMU structure (required by libpfm4) */
    memset(&pinfo,0,sizeof(pfm_pmu_info_t));
    ret=pfm_get_pmu_info(pmu_idx, &pinfo);

    if ((ret==PFM_SUCCESS) && pmu_is_present_and_right_type(&pinfo,pmu_type)) {

      pidx=pinfo.first_event;
      SUBDBG("First event in pmu: %s is %#x\n", pinfo.name, pidx);

      if (pidx<0) {
	/* For some reason no events available */
	/* despite the PMU being active.       */
        /* This can happen, for example with ix86arch */
	/* inside of VMware                           */
      }
      else {
         SUBDBG("EXIT: pidx: %#x\n", pidx);
         return pidx;
      }
    }

    pmu_idx++;

  }

  SUBDBG("EXIT: PAPI_ENOEVNT\n");
  return PAPI_ENOEVNT;
  
}


/***********************************************************/
/* Exported functions                                      */
/***********************************************************/


/** @class  _pe_libpfm4_ntv_name_to_code
 *  @brief  Take an event name and convert it to an event code.
 *
 *  @param[in] *name
 *        -- name of event to convert
 *  @param[out] *event_code
 *        -- pointer to an integer to hold the event code
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK event was found and an event assigned
 *  @retval PAPI_ENOEVENT event was not found
 */

int
_pe_libpfm4_ntv_name_to_code( char *name, unsigned int *event_code,
				struct native_event_table_t *event_table)
{
  SUBDBG( "ENTER: name: %s, event_code: %p, *event_code: %#x, event_table: %p\n", name, event_code, *event_code, event_table);

  struct native_event_t *our_event;
  int event_num;

  // if we already know this event name, just return its native code
  event_num=find_existing_event(name, event_table);
  if (event_num >= 0) {
	     *event_code=event_table->native_events[event_num].libpfm4_idx;
	     SUBDBG("EXIT: Found code: %#x\n",*event_code);
	     return PAPI_OK;
  }

     // Try to allocate this event to see if it is known by libpfm4, if allocate fails tell the caller it is not valid
     our_event=allocate_native_event(name, -1, event_table);
     if (our_event==NULL) {
    	 SUBDBG("EXIT: Allocating event: '%s' failed\n", name);
    	 return PAPI_ENOEVNT;
     }

     *event_code = our_event->libpfm4_idx;
     SUBDBG("EXIT: Found code: %#x\n",*event_code);
     return PAPI_OK;
}


/** @class  _pe_libpfm4_ntv_code_to_name
 *  @brief  Take an event code and convert it to a name
 *
 *  @param[in] EventCode
 *        -- PAPI event code
 *  @param[out] *ntv_name
 *        -- pointer to a string to hold the name
 *  @param[in] len
 *        -- length of ntv_name string
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       The event was found and converted to a name
 *  @retval PAPI_ENOEVENT The event does not exist
 *  @retval PAPI_EBUF     The event name was too big for ntv_name
 */

int
_pe_libpfm4_ntv_code_to_name(unsigned int EventCode,
			       char *ntv_name, int len,
			       struct native_event_table_t *event_table)
{
	SUBDBG("ENTER: EventCode: %#x, ntv_name: %p, len: %d, event_table: %p\n", EventCode, ntv_name, len, event_table);

	int eidx;
	int ret;
	int papi_ret;
	int papi_event_code;
	pfm_pmu_info_t pinfo;
	pfm_event_info_t einfo;

	// get the attribute index for this papi event
	papi_event_code = _papi_hwi_get_papi_event_code();

	// an attribute index less than 0 is invalid, return error
	if (papi_event_code <= 0) {
		SUBDBG("EXIT: PAPI_ENOEVNT\n");
		return PAPI_ENOEVNT;
	}

	// find our native event table for this papi event code
	for (eidx=0 ; eidx<event_table->num_native_events ; eidx++) {
		if ((papi_event_code == event_table->native_events[eidx].papi_event_code) && (EventCode == ((unsigned)event_table->native_events[eidx].libpfm4_idx))) {
			break;
		}
	}

	if (eidx < event_table->num_native_events) {
		char *ename = event_table->native_events[eidx].pmu_plus_name;

		// if it will not fit, return error
		if (strlen (ename) >= (unsigned)len) {
			SUBDBG("EXIT: event name %s will not fit in buffer provided\n", ename);
			return PAPI_EBUF;
		}
		strcpy (ntv_name, ename);

		SUBDBG("EXIT: event name: %s\n", ename);
		return PAPI_OK;
	}

	// if we did not find a match in our native event table, then the code passed in has not been allocated yet
	// try to use libpfm4 to see if we can get a name (not sure if this can happen)

	// We must be listing events so try to get event name from libpfm4
	SUBDBG("Use libpfm4 to get event name\n");

	// get this events information from libpfm4 (structure be zeroed)
	memset( &einfo, 0, sizeof( pfm_event_info_t ));
	einfo.size = sizeof(pfm_event_info_t);
	if ((ret = pfm_get_event_info(EventCode, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
		papi_ret = _papi_libpfm4_error(ret);
		SUBDBG("EXIT: return: %d (%s)\n", papi_ret, PAPI_strerror(papi_ret));
		return papi_ret;
	}

	// get the pmu name
	memset( &pinfo, 0, sizeof(pfm_pmu_info_t) );
	pinfo.size = sizeof(pfm_pmu_info_t);
	ret=pfm_get_pmu_info(einfo.pmu, &pinfo);
	if (ret!=PFM_SUCCESS) {
		SUBDBG("EXIT: pfm_get_pmu_info returned: %d\n", ret);
		return ret;
	}

	// start with empty name string
	ntv_name[0] = 0;

	/* Only prepend PMU name if not on the "default" PMU */
	if (event_table->default_pmu.pmu != pinfo.pmu) {
		// if pmu name will not fit, return an error
		if (strlen(pinfo.name)+2 >= (unsigned)len) {
			SUBDBG("EXIT: PAPI_EBUF (%s), pmu name does not fit\n", PAPI_strerror(PAPI_EBUF));
			return PAPI_EBUF;
		}
		strcat(ntv_name, pinfo.name);
		strcat(ntv_name, "::");
		len -= (strlen(pinfo.name)+2);
	}

	// now add the event name
	if (strlen(einfo.name) < (unsigned)len) {
		strcat(ntv_name, einfo.name);
		len -= strlen(einfo.name);

		SUBDBG("EXIT: name: %s\n", ntv_name);
		return PAPI_OK;
	}

	SUBDBG("EXIT: PAPI_EBUF (%s), event name does not fit\n", PAPI_strerror(PAPI_EBUF));
	return PAPI_EBUF;
}


/** @class  _pe_libpfm4_ntv_code_to_descr
 *  @brief  Take an event code and convert it to a description
 *
 *  @param[in] EventCode
 *        -- PAPI event code
 *  @param[out] *ntv_descr
 *        -- pointer to a string to hold the description
 *  @param[in] len
 *        -- length of ntv_descr string
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       The event was found and converted to a description
 *  @retval PAPI_ENOEVENT The event does not exist
 *  @retval PAPI_EBUF     The event name was too big for ntv_descr
 *
 *  Return the event description.
 *  If the event has umasks, then include ", masks" and the
 *  umask descriptions follow, separated by commas.
 */


int
_pe_libpfm4_ntv_code_to_descr( unsigned int EventCode,
				 char *ntv_descr, int len,
			         struct native_event_table_t *event_table)
{
	SUBDBG("ENTER: EventCode: %#x, ntv_descr: %p, len: %d: event_table: %p\n", EventCode, ntv_descr, len, event_table);

	int ret;
	int papi_ret;
	pfm_event_info_t einfo;

	if (event_table->pmu_type) {
		;   // avoid compiler warning
	}

	/* libpfm requires the structure be zeroed */
	memset( &einfo, 0, sizeof( pfm_event_info_t ));
	einfo.size = sizeof(pfm_event_info_t);

	// get this events information from libpfm4
	ret = pfm_get_event_info(EventCode, PFM_OS_PERF_EVENT_EXT, &einfo);
	SUBDBG("GOT BACK, ret: %d\n", ret);

	if (ret == PFM_SUCCESS) {
		// if name will not fit, return an error
		if (strlen(einfo.desc) > (unsigned)len) {
			SUBDBG("EXIT: PAPI_EBUF (%s)\n", PAPI_strerror(PAPI_EBUF));
			return PAPI_EBUF;
		}

		strncpy(ntv_descr, einfo.desc, len);
		SUBDBG("EXIT: description: %s\n", ntv_descr);
		return PAPI_OK;
	}

	papi_ret = _papi_libpfm4_error(ret);
	SUBDBG("EXIT: return: %d (%s)\n", papi_ret, PAPI_strerror(papi_ret));
	return papi_ret;
}


int
_pe_libpfm4_ntv_code_to_info(unsigned int EventCode,
			       PAPI_event_info_t *info,
			       struct native_event_table_t *event_table)
{
	SUBDBG("ENTER: EventCode: %#x, info: %p, event_table: %p\n", EventCode, info, event_table);

	int ret;
	struct native_event_t *ntv_evt;
	char *temp;
	pfm_event_attr_info_t ainfo;

	// get the native event index used for this papi event, if we do not know about this event yet try to get info from libpfm4
	int ntv_idx = _papi_hwi_get_ntv_idx(_papi_hwi_get_papi_event_code());
	if (ntv_idx < 0) {
		// collect libpfm4 information about this event
		if ((ret = _pe_libpfm4_ntv_code_to_name(EventCode, info->symbol, sizeof(info->symbol), event_table)) != PAPI_OK) {
			SUBDBG("EXIT: _pe_libpfm4_ntv_code_to_name returned: %d\n", ret);
			return PAPI_ENOEVNT;
		}
		if ((ret = _pe_libpfm4_ntv_code_to_descr(EventCode, info->long_descr, sizeof(info->long_descr), event_table)) != PAPI_OK) {
			SUBDBG("EXIT: _pe_libpfm4_ntv_code_to_descr returned: %d\n", ret);
			return PAPI_ENOEVNT;
		}

		SUBDBG("EXIT: EventCode: %#x, name: %s, desc: %s\n", EventCode, info->symbol, info->long_descr);
		return PAPI_OK;
	}
	ntv_evt = &(event_table->native_events[ntv_idx]);
//	SUBDBG("event_table->native_events[%d]: %p, ntv_evt: %p, ntv_evt->attr_idx: %d\n", ntv_idx, &(event_table->native_events[ntv_idx]), ntv_evt, ntv_evt->attr_idx);

	// make sure there is enough room for event name
	if (strlen(ntv_evt->pmu_plus_name) >= sizeof(info->symbol)) {
		SUBDBG("EXIT: event name too long\n");
		return PAPI_ENOEVNT;
	}

	// put the pmu name and event name into the callers info structure
	strcpy(info->symbol, ntv_evt->pmu_plus_name);

	// put the event description into the info structure
	if ((ret = _pe_libpfm4_ntv_code_to_descr(EventCode, info->long_descr, sizeof(info->long_descr), event_table)) != PAPI_OK) {
		SUBDBG("EXIT: _pe_libpfm4_ntv_code_to_descr returned: %d\n", ret);
		return PAPI_ENOEVNT;
	}

	// if native event attribute index is negative, we are listing an event which does not have a mask
	// just return what was built so far
	if (ntv_evt->attr_idx < 0) {
		SUBDBG("EXIT: EventCode: %#x, name: %s, desc: %s\n", EventCode, info->symbol, info->long_descr);
		return PAPI_OK;
	}

	// there should be a mask for this event code so add its information to end of event name and description

	// go get the attribute information for this event
	// libpfm4 likes the table cleared
	memset (&ainfo, 0, sizeof(pfm_event_attr_info_t));
	ainfo.size = sizeof(pfm_event_attr_info_t);
	ret = pfm_get_event_attr_info(EventCode, ntv_evt->attr_idx, PFM_OS_PERF_EVENT_EXT, &ainfo);
	if (ret != PFM_SUCCESS) {
		SUBDBG("EXIT: Attribute info not found, EventCode: %#x, attr_idx: %d, ret: %d\n", EventCode, ntv_evt->attr_idx, _papi_libpfm4_error(ret));
		return _papi_libpfm4_error(ret);
	}
	SUBDBG("EventCode: %#x, attr_idx: %d, type: %d, name: %s, description: %s\n", EventCode, ntv_evt->attr_idx, ainfo.type, ainfo.name, ainfo.desc);

	// format the attribute name so we can put it on the end of the event name
	temp = malloc(strlen(ainfo.name) + 35);
	switch (ainfo.type) {
	case PFM_ATTR_UMASK:
		sprintf(temp,":%s",ainfo.name);
		break;
	case PFM_ATTR_MOD_BOOL:
	case PFM_ATTR_MOD_INTEGER:
		sprintf(temp, ":%s", ainfo.name);
		// a few attributes require a non-zero value to encode correctly (most would accept zero here)
		strcat(temp,"=0");
		break;
	default:
		free(temp);
		SUBDBG("EXIT: Unsupported attribute type: %d", ainfo.type);
		return PAPI_EATTR;
	}

	if ((strlen(info->symbol) + strlen(temp)) >= sizeof(info->symbol)) {
		SUBDBG("EXIT: Not enough room in info->symbol for event string: need: %u, have: %u", (unsigned)(strlen(info->symbol) + strlen(temp)), (unsigned)sizeof(info->symbol));
		free(temp);
		return PAPI_EBUF;
	}
	strcat(info->symbol, temp);
	free(temp);


	if ((strlen(info->long_descr) + 8 + strlen(ainfo.desc)) >= sizeof(info->long_descr)) {
		SUBDBG("EXIT: Not enough room in info->long_descr for event description: need: %u, have: %u", (unsigned)(strlen(info->long_descr) + 8 +strlen(ainfo.desc)), (unsigned)sizeof(info->long_descr));
		return PAPI_EBUF;
	}
	strcat (info->long_descr, ", masks:");
	strcat (info->long_descr, ainfo.desc);

	SUBDBG("EXIT: EventCode: %#x, name: %s, desc: %s\n",
			EventCode, info->symbol, info->long_descr);
	return PAPI_OK;
}


/** @class  _pe_libpfm4_ntv_enum_events
 *  @brief  Walk through all events in a pre-defined order
 *
 *  @param[in,out] *PapiEventCode
 *        -- PAPI event code to start with
 *  @param[in] modifier
 *        -- describe how to enumerate
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       The event was found and converted to a description
 *  @retval PAPI_ENOEVENT The event does not exist
 *  @retval PAPI_ENOIMPL  The enumeration method requested in not implemented
 *
 */

int
_pe_libpfm4_ntv_enum_events( unsigned int *PapiEventCode,
			       int modifier,
			       struct native_event_table_t *event_table) {

	SUBDBG("ENTER: PapiEventCode: %p, *PapiEventCode: %#x, modifier: %d, event_table: %p\n", PapiEventCode, *PapiEventCode, modifier, event_table);

	int code,ret, pnum;
	int max_umasks;
	char event_string[BUFSIZ];
	pfm_event_info_t einfo;
	struct native_event_t *our_event;

	/* return first event if so specified */
	if ( modifier == PAPI_ENUM_FIRST ) {
		attr_idx = 0;   // set so if they want attribute information, it will start with the first attribute
		code=get_first_event_next_pmu(-1, event_table->pmu_type);
	   if (code < 0 ) {
	       SUBDBG("EXIT: Invalid component first event code: %d\n", code);
	      return code;
	   }

		// get this events name
		// libpfm requires the structure be zeroed
		memset( &einfo, 0, sizeof( pfm_event_info_t ));
		einfo.size = sizeof(pfm_event_info_t);

		// get this events information from libpfm4
		if ((ret = pfm_get_event_info(code, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
			SUBDBG("EXIT: pfm_get_event_info returned: %d\n", ret);
			return PAPI_ENOIMPL;
		}

	    /* clear the PMU structure (required by libpfm4) */
		pfm_pmu_info_t pinfo;
		memset(&pinfo,0,sizeof(pfm_pmu_info_t));
		if ((ret=pfm_get_pmu_info(einfo.pmu, &pinfo)) != PFM_SUCCESS) {
			SUBDBG("EXIT: pfm_get_pmu_info returned: %d\n", ret);
			return PAPI_ENOIMPL;
		}

		sprintf(event_string, "%s::%s", pinfo.name, einfo.name);
		SUBDBG("code: %#x, event: %s\n", code, einfo.name);

		int new_event_code = _papi_hwi_native_to_eventcode(_pe_libpfm4_get_cidx(), code, -1, event_string);
		_papi_hwi_set_papi_event_string(event_string);
		_papi_hwi_set_papi_event_code(new_event_code, 1);
		*PapiEventCode = code;

		SUBDBG("EXIT: *PapiEventCode: %#x\n", *PapiEventCode);
		return PAPI_OK;
	}

	/* Handle looking for the next event */
	if ( modifier == PAPI_ENUM_EVENTS ) {
		attr_idx = 0;   // set so if they want attribute information, it will start with the first attribute
		// get the next event code from libpfm4, if there are no more in this pmu find first event in next pmu
		if ((code = pfm_get_event_next(*PapiEventCode)) < 0) {

			/* libpfm requires the structure be zeroed */
			memset( &einfo, 0, sizeof( pfm_event_info_t ));
			einfo.size = sizeof(pfm_event_info_t);

			// get this events information from libpfm4, we need the pmu number of the last event we processed
			if ((ret = pfm_get_event_info(*PapiEventCode, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
				SUBDBG("EXIT: pfm_get_event_info returned: %d\n", ret);
				return PAPI_ENOIMPL;
			}
			SUBDBG("*PapiEventCode: %#x, event: %s\n", *PapiEventCode, einfo.name);

			// get the pmu number of the last event
			pnum = einfo.pmu;

			while ( pnum<PFM_PMU_MAX) {
				SUBDBG("pnum: %d\n", pnum);
				code=get_first_event_next_pmu(pnum, event_table->pmu_type);
				if (code < 0) {
					SUBDBG("EXIT: No more pmu's to list, returning: %d\n", code);
					return code;
				}
				break;
			}
		}

		// get the event name for this event code
		_pe_libpfm4_ntv_code_to_name(code, event_string, sizeof(event_string), event_table);

		// go allocate this event, need to create tables used by the get event info call that will follow
		if ((our_event = allocate_native_event(event_string, code, event_table)) == NULL) {
			// allocate may have created the event table but returned NULL to tell the caller the event string was invalid (attempt to encode it failed).
			// if the caller wants to use this event to count something, it will report an error
			// but if the caller is just interested in listing the event, then we need an event table with an event name and libpfm4 index
			int evt_idx;
			if ((evt_idx = find_existing_event(event_string, event_table)) < 0) {
				SUBDBG("EXIT: Allocating event: '%s' failed\n", event_string);
				return PAPI_ENOEVNT;
			}

			// give back the new event code
			*PapiEventCode = event_table->native_events[evt_idx].libpfm4_idx;
			SUBDBG("EXIT: event code: %#x\n", *PapiEventCode);
			return PAPI_OK;
		}

		// the event table may have been created as part of processing the papi preset events
		// if that happened, it may not have a valid attribute index yet, so put correct index into event table
		our_event->attr_idx = attr_idx;
//      SUBDBG("our_event: %p, papi_event_code: %#x, libpfm4_idx: %#x, attr_idx: %d\n", our_event, our_event->papi_event_code, our_event->libpfm4_idx, our_event->attr_idx);

		// give back the new event code
		*PapiEventCode = our_event->libpfm4_idx;

		SUBDBG("EXIT: *PapiEventCode: %#x\n", *PapiEventCode);
		return PAPI_OK;
	}

	/* We don't handle PAPI_NTV_ENUM_UMASK_COMBOS */
	if ( modifier == PAPI_NTV_ENUM_UMASK_COMBOS ) {
		SUBDBG("EXIT: do not support umask combos yet\n");
		return PAPI_ENOIMPL;
	} 

	/* Enumerate PAPI_NTV_ENUM_UMASKS (umasks on an event) */
	if ( modifier == PAPI_NTV_ENUM_UMASKS ) {
		/* libpfm requires the structure be zeroed */
		memset( &einfo, 0, sizeof( pfm_event_info_t ));
		einfo.size = sizeof(pfm_event_info_t);

		// get this events information from libpfm4, we need the number of masks this event knows about
		if ((ret = pfm_get_event_info(*PapiEventCode, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
			SUBDBG("EXIT: pfm_get_event_info returned: %d\n", ret);
			return PAPI_ENOIMPL;
		}
//      SUBDBG("*PapiEventCode: %#x, einfo.name: %s, einfo.code: %#x, einfo.nattrs: %d\n", *PapiEventCode, einfo.name, einfo.code, einfo.nattrs);

		// set max number of masks
		max_umasks = einfo.nattrs;

		// if we reached last attribute, return error to show we are done with this events masks
		if (attr_idx == max_umasks) {
			SUBDBG("EXIT: already processed all umasks: attr_idx: %d\n", attr_idx);
			return PAPI_ENOEVNT;
		}

		// get the event name for this event code
		_pe_libpfm4_ntv_code_to_name(*PapiEventCode, event_string, sizeof(event_string), event_table);

		// go get the attribute information for this event
		// libpfm4 likes the table cleared
		pfm_event_attr_info_t ainfo;
		memset (&ainfo, 0, sizeof(pfm_event_attr_info_t));
		ainfo.size = sizeof(pfm_event_attr_info_t);
		ret = pfm_get_event_attr_info(*PapiEventCode, attr_idx, PFM_OS_PERF_EVENT_EXT, &ainfo);
		if (ret != PFM_SUCCESS) {
			SUBDBG("EXIT: Attribute info not found, EventCode: %#x, attr_idx: %d, ret: %d\n", *PapiEventCode, attr_idx, _papi_libpfm4_error(ret));
			return _papi_libpfm4_error(ret);
		}
		SUBDBG("*PapiEventCode: %#x, attr_idx: %d, type: %d, name: %s, description: %s\n", *PapiEventCode, attr_idx, ainfo.type, ainfo.name, ainfo.desc);

		if (strlen(event_string) + strlen(ainfo.name) + 35 > sizeof(event_string)) {
			SUBDBG("EXIT: Event name and mask will not fit into buffer\n");
			return PAPI_EBUF;
		}

		strcat (event_string, ":");
		strcat (event_string, ainfo.name);
		switch (ainfo.type) {
			case PFM_ATTR_UMASK:
				break;
			case PFM_ATTR_MOD_BOOL:
			case PFM_ATTR_MOD_INTEGER:
				// a few attributes require a non-zero value to encode correctly (most would accept zero here)
				strcat(event_string,"=0");
				break;
			default:
				SUBDBG("EXIT: Unsupported attribute type: %d", ainfo.type);
				return PAPI_EATTR;
		}

		// go allocate this event, need to create tables used by the get event info call that will follow
		if ((our_event = allocate_native_event(event_string, *PapiEventCode, event_table)) == NULL) {
			// allocate may have created the event table but returned NULL to tell the caller the event string was invalid.
			// if the caller wants to use this event to count something, it must report the error
			// but if the caller is just interested in listing the event (like this code), then find the table that was created and return its libpfm4 index
			int evt_idx;
			if ((evt_idx = find_existing_event(event_string, event_table)) < 0) {
				SUBDBG("EXIT: Allocating event: '%s' failed\n", event_string);
				return PAPI_ENOEVNT;
			}
			// bump so next time we will use next attribute
			attr_idx++;
			// give back the new event code
			*PapiEventCode = event_table->native_events[evt_idx].libpfm4_idx;
			SUBDBG("EXIT: event code: %#x\n", *PapiEventCode);
			return PAPI_OK;
		}

		// the event table may have been created as part of processing the papi preset events
		// if that happened, it may not have a valid attribute index yet, so put correct index into event table
		our_event->attr_idx = attr_idx;
//      SUBDBG("our_event: %p, papi_event_code: %#x, libpfm4_idx: %#x, attr_idx: %d\n", our_event, our_event->papi_event_code, our_event->libpfm4_idx, our_event->attr_idx);

		// bump so next time we will use next attribute
		attr_idx++;

		// give back the new event code
		*PapiEventCode = our_event->libpfm4_idx;

		SUBDBG("EXIT: event code: %#x\n", *PapiEventCode);
		return PAPI_OK;
	}

	/* Enumerate PAPI_NTV_ENUM_GROUPS (groups on an event) */
	if ( modifier == PAPI_NTV_ENUM_GROUPS ) {
		SUBDBG("EXIT: do not support enumerating groups in this component\n");
		return PAPI_ENOIMPL;
	}

	/* An unknown enumeration method was indicated */

	SUBDBG("EXIT: Invalid modifier argument provided\n");
	return PAPI_ENOIMPL;
}


/** @class  _pe_libpfm4_shutdown
 *  @brief  Shutdown any initialization done by the libpfm4 code
 *
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       We always return PAPI_OK
 *
 */

int 
_pe_libpfm4_shutdown(struct native_event_table_t *event_table) {
  SUBDBG("ENTER: event_table: %p\n", event_table);

  int i;

  /* clean out and free the native events structure */
  _papi_hwi_lock( NAMELIB_LOCK );

     /* free memory allocated with strdup or malloc */
     for( i=0; i<event_table->num_native_events; i++) {
        free(event_table->native_events[i].base_name);
        free(event_table->native_events[i].pmu_plus_name);
        free(event_table->native_events[i].pmu);
        free(event_table->native_events[i].allocated_name);
        free(event_table->native_events[i].mask_string);
     }

     memset(event_table->native_events,0,
	 sizeof(struct native_event_t)*event_table->allocated_native_events);
     event_table->num_native_events=0;
     event_table->allocated_native_events=0;
     free(event_table->native_events);

  _papi_hwi_unlock( NAMELIB_LOCK );

  return PAPI_OK;
}


/** @class  _pe_libpfm4_init
 *  @brief  Initialize the libpfm4 code
 *
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       We initialized correctly
 *  @retval PAPI_ECMP     There was an error initializing the component
 *
 */

int
_pe_libpfm4_init(papi_vector_t *my_vector, int cidx,
		   struct native_event_table_t *event_table,
		   int pmu_type) {

   int detected_pmus=0, found_default=0;
   int i;
   pfm_err_t retval = PFM_SUCCESS;
   unsigned int ncnt;
   pfm_pmu_info_t pinfo;

   /* allocate the native event structure */

   event_table->num_native_events=0;
   event_table->pmu_type=pmu_type;

   event_table->native_events=calloc(NATIVE_EVENT_CHUNK,
					   sizeof(struct native_event_t));
   if (event_table->native_events==NULL) {
      return PAPI_ENOMEM;
   }
   event_table->allocated_native_events=NATIVE_EVENT_CHUNK;

   /* Count number of present PMUs */
   detected_pmus=0;
   ncnt=0;

   /* need to init pinfo or pfmlib might complain */
   memset(&(event_table->default_pmu), 0, sizeof(pfm_pmu_info_t));
   /* init default pmu */
   retval=pfm_get_pmu_info(0, &(event_table->default_pmu));
   
   SUBDBG("Detected pmus:\n");
   for(i=0;i<PFM_PMU_MAX;i++) {
      memset(&pinfo,0,sizeof(pfm_pmu_info_t));
      retval=pfm_get_pmu_info(i, &pinfo);
      if (retval!=PFM_SUCCESS) {
	 continue;
      }

      if (pmu_is_present_and_right_type(&pinfo,pmu_type)) {
	 SUBDBG("\t%d %s %s %d\n",i,pinfo.name,pinfo.desc,pinfo.type);

         detected_pmus++;
	 ncnt+=pinfo.nevents;
       
         if (pmu_type&PMU_TYPE_CORE) {

	    /* Hack to have "default" PMU */
	    if ( (pinfo.type==PFM_PMU_TYPE_CORE) &&
                  strcmp(pinfo.name,"ix86arch")) {

	       SUBDBG("\t  %s is default\n",pinfo.name);
	       memcpy(&(event_table->default_pmu),
		      &pinfo,sizeof(pfm_pmu_info_t));
	       found_default++;
	    }
	 }
         if (pmu_type==PMU_TYPE_UNCORE) {
	     /* To avoid confusion, no "default" CPU for uncore */
	       found_default=1;
	 }
      }
   }
   SUBDBG("%d native events detected on %d pmus\n",ncnt,detected_pmus);

   if (!found_default) {
      SUBDBG("Could not find default PMU\n");
      return PAPI_ECMP;
   }

   if (found_default>1) {
     PAPIERROR("Found too many default PMUs!\n");
     return PAPI_ECMP;
   }

   my_vector->cmp_info.num_native_events = ncnt;

   my_vector->cmp_info.num_cntrs = event_table->default_pmu.num_cntrs+
                                   event_table->default_pmu.num_fixed_cntrs;

   SUBDBG( "num_counters: %d\n", my_vector->cmp_info.num_cntrs );
   
   /* Setup presets, only if Component 0 */
   if (cidx==0) {
      retval = _papi_load_preset_table( (char *)event_table->default_pmu.name, 
				     event_table->default_pmu.pmu, cidx );
      if ( retval ) {
         return retval;
      }
   }

   return PAPI_OK;
}

