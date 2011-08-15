/*
* File:    papi_libpfm4_events.c
* Author:  Vince Weaver vweaver1 @ eecs.utk.edu
*          based heavily on existing papi_libpfm3_events.c
*/

#include <ctype.h>
#include <string.h>
#include <errno.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "papi_libpfm_events.h"

#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"

extern papi_vector_t MY_VECTOR;
volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];

/*******************************************************************
 *
 *
 *
 ******************************************************************/

/* FIXME -- make it handle arbitrary number */
#define MAX_NATIVE_EVENTS 1000

static struct native_event_t {
  int component;
  char *pmu;
  int papi_code;
  int perfmon_idx;
  char *allocated_name;
  char *base_name;
  //char *canonical_name;
  char *pmu_plus_name;
  int users;
} native_events[MAX_NATIVE_EVENTS];

static int num_native_events=0;


static struct native_event_t *find_existing_event(char *name) {

  int i;
  struct native_event_t *temp_event=NULL;

  SUBDBG("Looking for %s\n",name);

  _papi_hwi_lock( NAMELIB_LOCK );

  for(i=0;i<num_native_events;i++) {

    if (!strcmp(name,native_events[i].allocated_name)) {
      SUBDBG("Found %s (%x %x)\n",
	     native_events[i].allocated_name,
	     native_events[i].perfmon_idx,
	     native_events[i].papi_code);
       temp_event=&native_events[i];
       break;
    }
  }
  _papi_hwi_unlock( NAMELIB_LOCK );

  if (!temp_event) SUBDBG("%s not allocated yet\n",name);
  return temp_event;
}

static struct native_event_t *find_existing_event_by_number(int eventnum) {

  int i;
  struct native_event_t *temp_event=NULL;

  _papi_hwi_lock( NAMELIB_LOCK );

  for(i=0;i<num_native_events;i++) {
    if (eventnum==native_events[i].papi_code) {
       temp_event=&native_events[i];
       break;
    }
  }
  _papi_hwi_unlock( NAMELIB_LOCK );

  SUBDBG("Found %p for %x\n",temp_event,eventnum);

  return temp_event;
}


static int find_event_no_aliases(char *name) {

  int j,i, ret;
  pfm_pmu_info_t pinfo;
  pfm_event_info_t event_info;
  char blah[BUFSIZ];

  SUBDBG("Looking for %s\n",name);

  pfm_for_all_pmus(j) {

    memset(&pinfo,0,sizeof(pfm_pmu_info_t));
    pfm_get_pmu_info(j, &pinfo);
    if (!pinfo.is_present) {
       SUBDBG("PMU %d not present, skipping...\n",j);
       continue;
    }

    SUBDBG("Looking in pmu %d\n",j);   
    i = pinfo.first_event; 
    while(1) {
        memset(&event_info,0,sizeof(pfm_event_info_t));
        ret=pfm_get_event_info(i, PFM_OS_PERF_EVENT, &event_info);
	if (ret<0) break;
	
	sprintf(blah,"%s::%s",pinfo.name,event_info.name);
	//SUBDBG("Trying %x %s\n",i,blah);
	if (!strcmp(name,blah)) {
	  SUBDBG("FOUND %s %s %x\n",name,blah,i);
	  return i;
	}

	//SUBDBG("Trying %x %s\n",i,event_info.name);
	if (!strcmp(name,event_info.name)) {
	  SUBDBG("FOUND %s %s %x\n",name,event_info.name,i);
	  return i;
	}
	i++;
    }
  }
  return -1;

}


static int find_next_no_aliases(int code) {

  int current_pmu=0,current_event=0,ret;
  pfm_pmu_info_t pinfo;
  pfm_event_info_t event_info;

  memset(&pinfo,0,sizeof(pfm_pmu_info_t));
  memset(&event_info,0,sizeof(pfm_event_info_t));

  pfm_get_event_info(code, PFM_OS_PERF_EVENT, &event_info);
  current_pmu=event_info.pmu;
  current_event=code+1;

  SUBDBG("Current is %x guessing next is %x\n",code,current_event);

stupid_loop:

  ret=pfm_get_event_info(current_event, PFM_OS_PERF_EVENT, &event_info);
  if (ret>=0) {
    SUBDBG("Returning %x\n",current_event);
     return current_event;
  }

  /* need to increment pmu */
inc_pmu:
  current_pmu++;
  SUBDBG("Incrementing PMU: %x\n",current_pmu);
  if (current_pmu>PFM_PMU_MAX) return -1;

  memset(&pinfo,0,sizeof(pfm_pmu_info_t));
  pfm_get_pmu_info(current_pmu, &pinfo);
  if (!pinfo.is_present) goto inc_pmu;
 
  current_event=pinfo.first_event;

  goto stupid_loop;

}


static struct native_event_t *allocate_native_event(char *name, 
						    int event_idx) {

  int new_event;

  pfm_err_t ret;
  //int count=5;
  unsigned int i;
  //uint64_t *codes;
  //char *fstr=NULL;
  char *base_start;
  //int found_idx;
  pfm_event_info_t info;
  pfm_pmu_info_t pinfo;
  char base[BUFSIZ],pmuplusbase[BUFSIZ];

  /* allocate canonical string */

  //codes=calloc(count,sizeof(uint64_t));

  //  ret=pfm_get_event_encoding(name, 
  //			     PFM_PLM0|PFM_PLM3,
  //			     &fstr, 
  //			     &found_idx, 
  //			     &codes, 
  //			     &count);



  //if (codes) free(codes);

  //if (ret!=PFM_SUCCESS) {
  //   return NULL;
  //}

  /* get basename */	      
  memset(&info,0,sizeof(pfm_event_info_t));
  memset(&pinfo,0,sizeof(pfm_pmu_info_t));
  ret = pfm_get_event_info(event_idx, PFM_OS_PERF_EVENT, &info);
  if (ret!=PFM_SUCCESS) {
     return NULL;
  }

  memset(&pinfo,0,sizeof(pfm_pmu_info_t));
  pfm_get_pmu_info(info.pmu, &pinfo);

  strncpy(base,name,BUFSIZ);
  i=0;
  base_start=base;
  while(i<strlen(base)) {
    if (base[i]==':') {
      if (base[i+1]==':') {
          i++;
	  base_start=&base[i+1];
      }
      else {
	base[i]=0;
      }
    }
    i++;
  }

  _papi_hwi_lock( NAMELIB_LOCK );

  new_event=num_native_events;

  native_events[new_event].base_name=strdup(base_start);
  //if (fstr) {
    //     native_events[new_event].canonical_name=strdup(fstr);
    //free(fstr);
     //  }

  { char tmp[BUFSIZ];
    sprintf(tmp,"%s::%s",pinfo.name,info.name);
    native_events[new_event].pmu_plus_name=strdup(tmp);

    sprintf(pmuplusbase,"%s::%s",pinfo.name,base_start);
  }

  native_events[new_event].component=0;
  native_events[new_event].pmu=strdup(pinfo.name);
  native_events[new_event].papi_code=new_event | PAPI_NATIVE_MASK;
    
  native_events[new_event].perfmon_idx=find_event_no_aliases(pmuplusbase);
  SUBDBG("Using %x as index instead of %x for %s\n",
	 native_events[new_event].perfmon_idx,event_idx,pmuplusbase);

  native_events[new_event].allocated_name=strdup(name);

  native_events[new_event].users=0;

  SUBDBG("Creating event %s with papi %x perfidx %x\n",
	 name,
	 native_events[new_event].papi_code,
	 native_events[new_event].perfmon_idx);

  num_native_events++;

  _papi_hwi_unlock( NAMELIB_LOCK );


  /* FIXME -- simply allocate more */
  if (num_native_events >= MAX_NATIVE_EVENTS) {
     fprintf(stderr,"TOO MANY NATIVE EVENTS\n");
     exit(0);
  }

  return &native_events[new_event];

}




/* convert a collection of pfm mask bits into an array of pfm mask indices */
static inline int
prepare_umask( unsigned int foo, unsigned int *values )
{
	unsigned int tmp = foo, i;
	int j = 0;

  SUBDBG("ENTER\n");

	SUBDBG( "umask 0x%x\n", tmp );
	while ( ( i = ( unsigned int ) ffs( ( int ) tmp ) ) ) {
		tmp = tmp ^ ( 1 << ( i - 1 ) );
		values[j] = i - 1;
		SUBDBG( "umask %d is %d\n", j, values[j] );
		j++;
	}
	return ( j );
}

static int find_max_umask(struct native_event_t *current_event) {

  pfm_event_attr_info_t ainfo;
  char *b;
  int a, ret, max =0;
  pfm_event_info_t info;
  char event_string[BUFSIZ],*ptr;

  SUBDBG("Enter\n");

  SUBDBG("Trying to find max umask in %s\n",current_event->allocated_name);

  strcpy(event_string,current_event->allocated_name);

  if (strstr(event_string,"::")) {
    ptr=strstr(event_string,"::");
    ptr+=2;
    b=strtok(ptr,":");
  }
  else {
     b=strtok(event_string,":");
  }

  if (!b) {
     SUBDBG("No colon!\n");
     return -1;
  }

  memset(&info,0,sizeof(pfm_event_info_t));
  ret = pfm_get_event_info(current_event->perfmon_idx, PFM_OS_PERF_EVENT, &info);
  if (ret!=PFM_SUCCESS) {
     SUBDBG("get_event_info failed\n");
     return -1;
  }

  /* skip first */
  b=strtok(NULL,":");
  if (!b) {
     SUBDBG("Skipping first failed\n");
     return -1;
  }

  while(b) {
    a=0;
    while(1) {

      SUBDBG("get_event_attr %x %d %p\n",current_event->perfmon_idx,a,&ainfo);

      memset(&ainfo,0,sizeof(pfm_event_attr_info_t));

      ret = pfm_get_event_attr_info(current_event->perfmon_idx, a, 
				    PFM_OS_PERF_EVENT, &ainfo);

      if (ret != PFM_SUCCESS) {
	SUBDBG("get_event_attr failed %s\n",pfm_strerror(ret));
	return ret;
      }

      SUBDBG("Trying %s with %s\n",ainfo.name,b);

      if (!strcasecmp(ainfo.name, b)) {
	SUBDBG("Found %s %d\n",b,a);
	if (a>max) max=a;
	goto found_attr;
      }
      a++;
    }

    SUBDBG("attr=%s not found for event %s\n", b, info.name);

    return PAPI_ECNFLCT;

found_attr:

    b=strtok(NULL,":");
  }

  SUBDBG("Found max %d\n", max);

  return max;
}

static int
get_event_first_active(void)
{
  int pidx, pmu_idx, ret;

  pfm_pmu_info_t pinfo;

  memset(&pinfo,0,sizeof(pfm_pmu_info_t));

  pmu_idx=0;

  while(pmu_idx<PFM_PMU_MAX) {

    memset(&pinfo,0,sizeof(pfm_pmu_info_t));
    ret=pfm_get_pmu_info(pmu_idx, &pinfo);

    if ((ret==PFM_SUCCESS) && pinfo.is_present) {

      pidx=pinfo.first_event;

      return pidx;

    }
    pmu_idx++;

  }
  return PAPI_ENOEVNT;
  
}


/* first PMU, no leading PMU indicator */
/* subsequent, yes */

static int
convert_libpfm4_to_string( int code, char **event_name)
{

  int ret;
  pfm_event_info_t gete;//,first_info;
  pfm_pmu_info_t pinfo;
  char name[BUFSIZ];
  //int first;

  SUBDBG("ENTER %x\n",code);

  //first=get_event_first_active();

  memset( &gete, 0, sizeof ( pfm_event_info_t ) );
  //memset( &first_info, 0, sizeof ( pfm_event_info_t ) );
  memset(&pinfo,0,sizeof(pfm_pmu_info_t));

  ret=pfm_get_event_info(code, PFM_OS_PERF_EVENT, &gete);
  //  ret=pfm_get_event_info(first, PFM_OS_PERF_EVENT, &first_info);

  memset(&pinfo,0,sizeof(pfm_pmu_info_t));
  pfm_get_pmu_info(gete.pmu, &pinfo);
  /* VMW */
  /* FIXME, make a "is it the default" function */

  if ( (pinfo.type==PFM_PMU_TYPE_CORE) &&
       strcmp(pinfo.name,"ix86arch")) {
    //  if (gete.pmu==first_info.pmu) {
     *event_name=strdup(gete.name);
  }
  else {
     sprintf(name,"%s::%s",pinfo.name,gete.name);
     *event_name=strdup(name);
  }

  SUBDBG("Found name: %s\n",*event_name);

  return ret;

}

static int convert_pfmidx_to_native(int code, unsigned int *PapiEventCode) {

  int ret;
  char *name=NULL;

  ret=convert_libpfm4_to_string( code, &name);
  SUBDBG("Converted %x to %s\n",code,name);
  if (ret==PFM_SUCCESS) {
     ret=_papi_libpfm_ntv_name_to_code(name,PapiEventCode);
     SUBDBG("RETURNING FIRST: %x %s\n",*PapiEventCode,name);
  }

  if (name) free(name);
  return ret;

}




static int find_next_umask(struct native_event_t *current_event,
                           int current,char *umask_name) {

  char temp_string[BUFSIZ];
  pfm_event_info_t event_info;
  pfm_event_attr_info_t *ainfo=NULL;
  int num_masks=0;
  pfm_err_t ret;
  int i;
  //  int actual_val=0;

  /* get number of attributes */

  memset(&event_info, 0, sizeof(event_info));
  ret=pfm_get_event_info(current_event->perfmon_idx, PFM_OS_PERF_EVENT, &event_info);
	
  SUBDBG("%d possible attributes for event %s\n",
	 event_info.nattrs,
	 event_info.name);

  ainfo = malloc(event_info.nattrs * sizeof(*ainfo));
  if (!ainfo) {
     return PAPI_ENOMEM;
  }

  pfm_for_each_event_attr(i, &event_info) {
     ainfo[i].size = sizeof(*ainfo);

     ret = pfm_get_event_attr_info(event_info.idx, i, PFM_OS_PERF_EVENT, 
				   &ainfo[i]);
     if (ret != PFM_SUCCESS) {
        SUBDBG("Not found\n");
        if (ainfo) free(ainfo);
	return PAPI_ENOEVNT;
     }

     if (ainfo[i].type == PFM_ATTR_UMASK) {
	SUBDBG("nm %d looking for %d\n",num_masks,current);
	if (num_masks==current+1) {	  
	   SUBDBG("Found attribute %d: %s type: %d\n",i,ainfo[i].name,ainfo[i].type);
	
           sprintf(temp_string,"%s",ainfo[i].name);
           strncpy(umask_name,temp_string,BUFSIZ);

	   if (ainfo) free(ainfo);
	   return current+1;
	}
	num_masks++;
     }
  }

  if (ainfo) free(ainfo);
  return -1;

}


/***********************************************************/
/* Exported functions                                      */
/***********************************************************/


/** @class  _papi_libpfm_ntv_name_to_code
 *  @brief  Take an event name and convert it to an event code.
 *
 *  @param[in] *name
 *        -- name of event to convert
 *  @param[out] *event_code
 *        -- pointer to an integer to hold the event code
 *
 *  @retval PAPI_OK event was found and an event assigned
 *  @retval PAPI_ENOEVENT event was not found
 */

int
_papi_libpfm_ntv_name_to_code( char *name, unsigned int *event_code )
{

  int actual_idx;
  struct native_event_t *our_event;

  SUBDBG( "Converting %s\n", name);

  our_event=find_existing_event(name);

  if (our_event==NULL) {

     /* event currently doesn't exist, so try to find it */
     /* using libpfm4                                    */

     SUBDBG("Using pfm to look up event %s\n",name);
     actual_idx=pfm_find_event(name);

     /* FIXME!  Map the libpfm4 error to a PAPI error */
     if (actual_idx<0) {
        return PAPI_ENOEVNT;
     }

     SUBDBG("Using %x as the index\n",actual_idx);

     /* We were found in libpfm4, so allocate our copy of the event */

     our_event=allocate_native_event(name,actual_idx);
  }

  if (our_event!=NULL) {      
     *event_code=our_event->papi_code;
     SUBDBG("Found code: %x\n",*event_code);
     return PAPI_OK;
  }

  /* Failure here means allocate_native_event failed */
  /* Can we give a better error?                     */

  SUBDBG("Event %s not found\n",name);

  return PAPI_ENOEVNT;   

}

int
_papi_libpfm_ntv_code_to_name( unsigned int EventCode, char *ntv_name, int len )
{

        struct native_event_t *our_event;

        SUBDBG("ENTER %x\n",EventCode);

        our_event=find_existing_event_by_number(EventCode);

	if (our_event==NULL) return PAPI_ENOEVNT;

	/* use actual rather than canonical to not break enum */
	strncpy(ntv_name,our_event->allocated_name,len);

	return PAPI_OK;
}

/* attributes concactinated onto end of descr separated by ", masks" */
/* then comma separated */

int
_papi_libpfm_ntv_code_to_descr( unsigned int EventCode, char *ntv_descr, int len )
{
  int ret,a,first_mask=1;
  char *eventd, *tmp=NULL;
	//int i, first_desc=1;
  pfm_event_info_t gete;
	//     	size_t total_len = 0;


  pfm_event_attr_info_t ainfo;
  char *b;
  //  pfm_event_info_t info;
  char event_string[BUFSIZ],*ptr;

	struct native_event_t *our_event;

        SUBDBG("ENTER %x\n",EventCode);

	our_event=find_existing_event_by_number(EventCode);

	memset( &gete, 0, sizeof ( gete ) );

	SUBDBG("Getting info on %x\n",our_event->perfmon_idx);
	ret=pfm_get_event_info(our_event->perfmon_idx, PFM_OS_PERF_EVENT, &gete);
	SUBDBG("Return=%d\n",ret);

	/* error check?*/

	eventd=strdup(gete.desc);

	tmp = ( char * ) malloc( strlen( eventd ) + 1 );
	if ( tmp == NULL ) {
	   free( eventd );
	   return PAPI_ENOMEM;
	}
	tmp[0] = '\0';
	strcat( tmp, eventd );
	free( eventd );
	
	/* Handle Umasks */
  strcpy(event_string,our_event->allocated_name);

  if (strstr(event_string,"::")) {
    ptr=strstr(event_string,"::");
    ptr+=2;
    b=strtok(ptr,":");
  }
  else {
    b=strtok(event_string,":");
  }
  
  if (!b) {
     SUBDBG("No colon!\n"); /* no umask */
     goto descr_in_tmp;
  }

  /* skip first */
  b=strtok(NULL,":");
  if (!b) {
     SUBDBG("Skipping first failed\n");
     goto descr_in_tmp;
  }

  while(b) {
    a=0;
    while(1) {

      SUBDBG("get_event_attr %x %p\n",our_event->perfmon_idx,&ainfo);

      memset(&ainfo,0,sizeof(pfm_event_attr_info_t));

      ret = pfm_get_event_attr_info(our_event->perfmon_idx, a, 
				    PFM_OS_PERF_EVENT, &ainfo);

      if (ret != PFM_SUCCESS) {
	SUBDBG("get_event_attr failed %s\n",pfm_strerror(ret));
	return ret;
      }

      SUBDBG("Trying %s with %s\n",ainfo.name,b);

      if (!strcasecmp(ainfo.name, b)) {
	int new_length;

	 SUBDBG("Found %s\n",b);
	 new_length=strlen(ainfo.desc);

	 if (first_mask) {
	    tmp=realloc(tmp,strlen(tmp)+new_length+1+strlen(", masks:"));
	    strcat(tmp,", masks:");
	 }
	 else {
	    tmp=realloc(tmp,strlen(tmp)+new_length+1+strlen(","));
	    strcat(tmp,",");
	 }
	 strcat(tmp,ainfo.desc);

	 goto found_attr;
      }
      a++;
    }

    SUBDBG("attr=%s not found for event %s\n", b, ainfo.name);

    return PAPI_ECNFLCT;

found_attr:

    b=strtok(NULL,":");
  }

descr_in_tmp:
	strncpy( ntv_descr, tmp, ( size_t ) len );
	if ( ( int ) strlen( tmp ) > len - 1 )
		ret = PAPI_EBUF;
	else
		ret = PAPI_OK;
	free( tmp );

	SUBDBG("PFM4 Code: %x %s\n",EventCode,ntv_descr);

	return ret;

}

int
_papi_libpfm_ntv_enum_events( unsigned int *PapiEventCode, int modifier )
{
	int code,ret;
	struct native_event_t *current_event;

        SUBDBG("ENTER\n");

	/* return first event if so specified */
	if ( modifier == PAPI_ENUM_FIRST ) {
	   unsigned int blah=0;
           SUBDBG("ENUM_FIRST\n");

	   code=get_event_first_active();
	   ret=convert_pfmidx_to_native(code, &blah);
	   *PapiEventCode=(unsigned int)blah;
           SUBDBG("FOUND %x (from %x) ret=%d\n",*PapiEventCode,code,ret);

	   return ret;
	}

	current_event=find_existing_event_by_number(*PapiEventCode);
	if (current_event==NULL) {
           SUBDBG("EVENTS %x not found\n",*PapiEventCode);
	   return PAPI_ENOEVNT;
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
	   SUBDBG("ENUM_EVENTS %x\n",*PapiEventCode);
	   unsigned int blah=0;

	   code=current_event->perfmon_idx;

	   ret=find_next_no_aliases(code);

	   SUBDBG("find_next_no_aliases() Returned %x\n",ret);
	   if (ret<0) {
	      SUBDBG("<0 so returning\n");
	      return ret;
	   }

	   SUBDBG("VMW BLAH1\n");

	   ret=convert_pfmidx_to_native(ret, &blah);

	   SUBDBG("VMW BLAH2\n");

	     if (ret<0) {
	       SUBDBG("Couldn't convert to native %d %s\n",
		      ret,PAPI_strerror(ret));
	     }
	     *PapiEventCode=(unsigned int)blah;

	     if ((ret!=PAPI_OK) && (blah!=0)) {
	        SUBDBG("Faking PAPI_OK because blah!=0\n");
	        return PAPI_OK;
	     }

             SUBDBG("Returning PAPI_OK\n");
	     return ret;

	}

	if ( modifier == PAPI_NTV_ENUM_UMASK_COMBOS ) {
		return PAPI_ENOEVNT;
	} 

	if ( modifier == PAPI_NTV_ENUM_UMASKS ) {

	   int max_umask,next_umask;
	   char umask_string[BUFSIZ],new_name[BUFSIZ];

	   SUBDBG("Finding maximum mask in event %s\n",
		  		  current_event->allocated_name);

	   max_umask=find_max_umask(current_event);
	   SUBDBG("Found max %d\n",max_umask);
	   next_umask=find_next_umask(current_event,max_umask,
				      umask_string);
	   SUBDBG("Found next %d\n",next_umask);
	   if (next_umask>=0) {
	     unsigned int blah;
	      sprintf(new_name,"%s:%s",current_event->base_name,
		     umask_string);
     
              ret=_papi_libpfm_ntv_name_to_code(new_name,&blah);
	      if (ret!=PAPI_OK) return PAPI_ENOEVNT;

	      *PapiEventCode=(unsigned int)blah;
	      SUBDBG("found code %x\n",*PapiEventCode);
	      return PAPI_OK;
	   }

	   SUBDBG("couldn't find umask\n");

	   return PAPI_ENOEVNT;

	} else {
		return PAPI_EINVAL;
	}
}


int
_papi_libpfm_ntv_code_to_bits( unsigned int EventCode, hwd_register_t *bits )
{

  *(int *)bits=EventCode;

  return PAPI_OK;
}


/* This function would return info on which counters an event could be in */
/* libpfm4 currently does not support this */

int
_papi_libpfm_ntv_bits_to_info( hwd_register_t * bits, char *names,
			    unsigned int *values, int name_len, int count )
{

  (void)bits;
  (void)names;
  (void)values;
  (void)name_len;
  (void)count;

  return PAPI_OK;

}

int 
_papi_libpfm_shutdown(void) {

  SUBDBG("shutdown\n");

  _papi_hwi_lock( NAMELIB_LOCK );
  memset(&native_events,0,sizeof(struct native_event_t)*MAX_NATIVE_EVENTS);
  num_native_events=0;
  _papi_hwi_unlock( NAMELIB_LOCK );

  return PAPI_OK;
}

int
_papi_libpfm_init(void) {

  int detected_pmus=0, found_default=0;
   pfm_pmu_info_t default_pmu;
 
   int i, version;
   pfm_err_t retval;
   unsigned int ncnt;
   pfm_pmu_info_t pinfo;

   /* The following checks the version of the PFM library
      against the version PAPI linked to... */
   if ( ( retval = pfm_initialize(  ) ) != PFM_SUCCESS ) {
      PAPIERROR( "pfm_initialize(): %s", pfm_strerror( retval ) );
      return PAPI_ESBSTR;
   }

   /* get the libpfm4 version */
   SUBDBG( "pfm_get_version()\n");
   if ( (version=pfm_get_version( )) < 0 ) {
      PAPIERROR( "pfm_get_version(): %s", pfm_strerror( retval ) );
      return PAPI_ESBSTR;
   }

   /* Set the version */
   sprintf( MY_VECTOR.cmp_info.support_version, "%d.%d",
	    PFM_MAJ_VERSION( version ), PFM_MIN_VERSION( version ) );

   /* Complain if the compiled-against version doesn't match current version */
   if ( PFM_MAJ_VERSION( version ) != PFM_MAJ_VERSION( LIBPFM_VERSION ) ) {
      PAPIERROR( "Version mismatch of libpfm: compiled %x vs. installed %x\n",
				   PFM_MAJ_VERSION( LIBPFM_VERSION ),
				   PFM_MAJ_VERSION( version ) );
      return PAPI_ESBSTR;
   }

   /* Count number of present PMUs */
   detected_pmus=0;
   ncnt=0;
   /* need to init pinfo or pfmlib might complain */
   memset(&default_pmu, 0, sizeof(pfm_pmu_info_t));
   /* init default pmu */
   retval=pfm_get_pmu_info(0, &default_pmu);
   
   SUBDBG("Detected pmus:\n");
   for(i=0;i<PFM_PMU_MAX;i++) {
      memset(&pinfo,0,sizeof(pfm_pmu_info_t));
      retval=pfm_get_pmu_info(i, &pinfo);
      if (retval!=PFM_SUCCESS) continue;
      if (pinfo.is_present) {
	SUBDBG("\t%d %s %s %d\n",i,pinfo.name,pinfo.desc,pinfo.type);

         detected_pmus++;
	 ncnt+=pinfo.nevents;
	 if ( (pinfo.type==PFM_PMU_TYPE_CORE) &&
              strcmp(pinfo.name,"ix86arch")) {

	    SUBDBG("\t  %s is default\n",pinfo.name);
	    memcpy(&default_pmu,&pinfo,sizeof(pfm_pmu_info_t));
	    found_default++;
	 }
      }
   }
   SUBDBG("%d native events detected on %d pmus\n",ncnt,detected_pmus);

   if (!found_default) {
      PAPIERROR("Could not find default PMU\n");
      return PAPI_ESBSTR;
   }

   if (found_default>1) {
     PAPIERROR("Found too many default PMUs!\n");
     return PAPI_ESBSTR;
   }

   MY_VECTOR.cmp_info.num_native_events = ncnt;

   MY_VECTOR.cmp_info.num_cntrs = default_pmu.num_cntrs+
                                  default_pmu.num_fixed_cntrs;
   SUBDBG( "num_counters: %d\n", MY_VECTOR.cmp_info.num_cntrs );

   MY_VECTOR.cmp_info.num_mpx_cntrs = MAX_MPX_EVENTS;
   
   /* Setup presets */
   retval = _papi_libpfm_setup_presets( (char *)default_pmu.name, 
				     default_pmu.pmu );
   if ( retval )
      return retval;
	
   return PAPI_OK;
}

int
_papi_libpfm_setup_counters( struct perf_event_attr *attr,
			   hwd_register_t *ni_bits ) {

  int ret;
  int our_idx;
  char our_name[BUFSIZ];
   
  pfm_perf_encode_arg_t perf_arg;

  memset(&perf_arg,0,sizeof(pfm_perf_encode_arg_t));
  perf_arg.attr=attr;
   
  our_idx=*(int *)(ni_bits);

  _papi_libpfm_ntv_code_to_name( our_idx,our_name,BUFSIZ);

  SUBDBG("trying \"%s\" %x\n",our_name,our_idx);

  ret = pfm_get_os_event_encoding(our_name, 
				  PFM_PLM0 | PFM_PLM3, 
                                  PFM_OS_PERF_EVENT_EXT, 
				  &perf_arg);
  if (ret!=PFM_SUCCESS) {
     return PAPI_ENOEVNT;
  }
  
  SUBDBG( "pe_event: config 0x%"PRIx64" config1 0x%"PRIx64" type 0x%"PRIx32"\n", 
          perf_arg.attr->config, 
	  perf_arg.attr->config1,
	  perf_arg.attr->type);
	  

  return PAPI_OK;
}





