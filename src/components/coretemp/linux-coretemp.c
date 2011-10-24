#include <string.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

#include "linux-coretemp.h"

/* this is what I found on my core2 machine 
 * but I have not explored this widely yet*/
#define REFRESH_LAT 4000

#define INVALID_RESULT -1000000L

/* temporary event */
struct temp_event {
  char name[PAPI_MAX_STR_LEN];
  char path[PATH_MAX];
  int stone;
  long count;
  struct temp_event *next;
};

static struct temp_event* root = NULL;
static CORETEMP_native_enent_entry_t * _coretemp_native_events;
static int NUM_EVENTS		= 0;
static int is_initialized	= 0;

/*******************************************************************************
 ********  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *********
 ******************************************************************************/

/*
 * find all coretemp information reported by the kernel
 */
int generateEventList(char *base_dir)
{
  char path[PATH_MAX];
  DIR *dir,*d;
  int count = 0;
  struct dirent *hwmonx,*events;
  struct temp_event *temp;
  struct temp_event *last = NULL;

  dir = opendir(base_dir);
  if ( dir == NULL ) {
	PAPIERROR("Oops: I can't find %s, are you sure the coretemp module is loaded?\n", base_dir);
	return( PAPI_ESYS );
  }
  while( (hwmonx = readdir(dir) ) ) {
	if ( !strncmp("hwmon", hwmonx->d_name, 5) ) {
	  snprintf(path, PATH_MAX, "%s/%s/device", base_dir, hwmonx->d_name);

	  SUBDBG("Trying to open %s\n",path);
	  d = opendir(path);
	  if (d==NULL) {
	     continue;
	  }

	  while( (events = readdir(d)) ) {
		if ( events->d_name[0] == '.' )
		  continue;

		if ( !strncmp("temp", events->d_name, 4) ||
			 !strncmp("fan", events->d_name, 3) ) {
		  /* new_event   path, events->d_name */
		  temp = (struct temp_event *)papi_malloc(sizeof(struct temp_event));
		  if (!temp) {
			PAPIERROR("out of memory!");
			/* We should also free any previously allocated data */
			return PAPI_ENOMEM;
		  }

		  temp->next = NULL;

		  if (root == NULL) {
		     root = temp;
		  }
		  else if (last) {
		     last->next = temp;
		  }
		  else {
		    /* Because this is a function, it is possible */
		    /* we are called with root!=NULL but no last  */
		    /* so add this to keep coverity happy         */
		    free(temp);
		    PAPIERROR("This shouldn't be possible\n");

		    return PAPI_ESBSTR;
		  }

		  last = temp;

		  snprintf(temp->name, PAPI_MAX_STR_LEN, "%s.%s", hwmonx->d_name, events->d_name);
		  snprintf(temp->path, PATH_MAX, "%s/%s", path, events->d_name);

		  /* don't optimize this yet....
		  s = strchr(events->d_name, (int)'_');
		  s++;
		  if ( !strcmp("min", s) ||
			   !strcmp("max", s) ) {
			temp->stone = 1;
			fp = fopen(temp->path,"r");
			fgets(s,fp);
		  }
		  */
		  count++;
		}
	  }
	  closedir(d);
	}
  }

  closedir(dir);
  return (count);
}

/*
 * This is called whenever a thread is initialized
 */
int coretemp_init( hwd_context_t *ctx )
{
  ( void ) ctx;
  return (PAPI_OK );
}

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int coretemp_init_substrate( )
{
  int i		= 0;
  struct temp_event *t,*last;

  if ( is_initialized )
	return (PAPI_OK );

  is_initialized = 1;
  /* This is the prefered method, all coretemp sensors are symlinked here
   * see $(kernel_src)/Documentation/hwmon/sysfs-interface */
  NUM_EVENTS = generateEventList("/sys/class/hwmon");

  if ( NUM_EVENTS < 0 ) 
	return ( NUM_EVENTS );

  if ( NUM_EVENTS == 0 ) 
	return ( PAPI_OK );

  t = root;
  _coretemp_native_events = (CORETEMP_native_enent_entry_t*)
	papi_malloc(sizeof(CORETEMP_native_enent_entry_t) * NUM_EVENTS);

  do {
	strncpy(_coretemp_native_events[i].name,t->name,PAPI_MAX_STR_LEN);
	strncpy(_coretemp_native_events[i].path,t->path,PATH_MAX);
	_coretemp_native_events[i].stone = 0;
	_coretemp_native_events[i].resources.selector = i + 1;
	last	= t;
	t		= t->next;
	papi_free(last);
	i++;
  } while (t != NULL);
  root = NULL;
  return (PAPI_OK);
}


long getEventValue( int index ) 
{
  char buf[PAPI_MAX_STR_LEN];
  FILE* fp;
  long result;

  if (_coretemp_native_events[index].stone) {
	return _coretemp_native_events[index].value;
  }

  fp = fopen(_coretemp_native_events[index].path, "r");
  if (fp==NULL) {
     return INVALID_RESULT;
  }

  if (fgets(buf, PAPI_MAX_STR_LEN, fp)==NULL) {
     result=INVALID_RESULT;
  }
  else {
     result=strtol(buf, NULL, 10);
  }
  fclose(fp);
  return result;
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int coretemp_init_control_state( hwd_control_state_t * ctl)
{
  int i;

  for ( i=0; i < NUM_EVENTS; i++ )
	( ( CORETEMP_control_state_t *) ctl )->counts[i] = getEventValue(i);
  
  ( ( CORETEMP_control_state_t *) ctl)->lastupdate = PAPI_get_real_usec();

  return (PAPI_OK);
}

int coretemp_start( hwd_context_t *ctx, hwd_control_state_t * ctl)
{
  ( void ) ctx;
  ( void ) ctl;

  return ( PAPI_OK );
}

int coretemp_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
	long long ** events, int flags)
{
  (void) flags;
  (void) ctx;

  CORETEMP_control_state_t* control = (CORETEMP_control_state_t*) ctl;
  long long now = PAPI_get_real_usec();
  int i;

  if ( now - control->lastupdate > REFRESH_LAT ) {
	for ( i = 0; i < NUM_EVENTS; i++ ) {
	  control->counts[i] = getEventValue( i );
	}
	control->lastupdate = now;
  }
  *events = control->counts;

  return ( PAPI_OK );
}

int coretemp_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
  (void) ctx;
  /* read values */
  CORETEMP_control_state_t* control = (CORETEMP_control_state_t*) ctl;
  int i;

  for ( i = 0; i < NUM_EVENTS; i++ ) {
	control->counts[i] = getEventValue( i );
  }

  return ( PAPI_OK );
}


/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/






/*
 *
 */
int
coretemp_shutdown( hwd_context_t * ctx )
{
    ( void ) ctx;

	papi_free(_coretemp_native_events);

	return ( PAPI_OK );
}



/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
coretemp_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
    ( void ) ctx;
	( void ) code;
	( void ) option;
	return ( PAPI_OK );
}



int
coretemp_update_control_state(	hwd_control_state_t * ptr,
								NativeInfo_t * native, int count,
								hwd_context_t * ctx )
{
	int i, index;
    ( void ) ctx;
	( void ) ptr;

	for ( i = 0; i < count; i++ ) {
		index =
			native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position = _coretemp_native_events[index].resources.selector - 1;
	}
	return ( PAPI_OK );
}


/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
int
coretemp_set_domain( hwd_control_state_t * cntrl, int domain )
{
	int found = 0;
    ( void ) cntrl;
	
	if ( PAPI_DOM_USER & domain )
		found = 1;

	if ( PAPI_DOM_KERNEL & domain )
		found = 1;

	if ( PAPI_DOM_OTHER & domain )
		found = 1;

	if ( !found )
		return ( PAPI_EINVAL );

	return ( PAPI_OK );
}


int
coretemp_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
    ( void ) ctx;
	( void ) ctrl;
	return ( PAPI_OK );
}


/*
 * Native Event functions
 */
int
coretemp_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );

	switch ( modifier ) {
	case PAPI_ENUM_FIRST:
		*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );

		return ( PAPI_OK );
		break;

	case PAPI_ENUM_EVENTS:
	{
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if ( index < NUM_EVENTS - 1 ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );

		break;
	}
	default:
		return ( PAPI_EINVAL );
	}
	return ( PAPI_EINVAL );
}

/*
 *
 */
int
coretemp_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	if ( index >= 0 && index < NUM_EVENTS ) {
	  strncpy( name, _coretemp_native_events[index].name, len );
	  return ( PAPI_OK );
	}
	return ( PAPI_ENOEVNT );
}

/*
 *
 */
int
coretemp_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	if ( index >= 0 && index < NUM_EVENTS ) {
	
	  strncpy( name, _coretemp_native_events[index].description, len );
	}
	return ( PAPI_ENOEVNT );
}

/*
 *
 */
int
coretemp_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	if ( 0 > index || NUM_EVENTS <= index )
	  return ( PAPI_ENOEVNT );
	memcpy( ( CORETEMP_register_t * ) bits,
			&( _coretemp_native_events[index].resources ),
			sizeof ( CORETEMP_register_t ) );
	return ( PAPI_OK );
}



/*
 *
 */
papi_vector_t _coretemp_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name =
				 "$Id$",
				 .version = "$Revision$",
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .num_cntrs = CORETEMP_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_USER,
				 //.available_domains = PAPI_DOM_USER,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 0,
				 .fast_virtual_timer = 0,
				 .attach = 0,
				 .attach_must_ptrace = 0,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( CORETEMP_context_t ),
			 .control_state = sizeof ( CORETEMP_control_state_t ),
			 .reg_value = sizeof ( CORETEMP_register_t ),
			 .reg_alloc = sizeof ( CORETEMP_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init = coretemp_init,
	.init_substrate = coretemp_init_substrate,
	.init_control_state = coretemp_init_control_state,
	.start = coretemp_start,
	.stop = coretemp_stop,
	.read = coretemp_read,
	.shutdown = coretemp_shutdown,
	.ctl = coretemp_ctl,

	.update_control_state = coretemp_update_control_state,
	.set_domain = coretemp_set_domain,
	.reset = coretemp_reset,

	.ntv_enum_events = coretemp_ntv_enum_events,
	.ntv_code_to_name = coretemp_ntv_code_to_name,
	.ntv_code_to_bits = coretemp_ntv_code_to_bits,
	.ntv_bits_to_info = NULL,
};
