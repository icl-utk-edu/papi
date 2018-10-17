/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
* @file     papi_hl.c
* @author   Frank Winkler
*           frank.winkler@icl.utk.edu
* @author   Philip Mucci
*           mucci@cs.utk.edu
* @brief This file contains the 'high level' interface to PAPI.
*  BASIC is a high level language. ;-) */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <search.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <error.h>
#include <errno.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include "papi.h"
#include "papi_internal.h"


#define verbose_fprintf \
   if (verbosity == 1) fprintf

/* defaults for number of components and events */
#define PAPIHL_NUM_OF_COMPONENTS 10
#define PAPIHL_NUM_OF_EVENTS_PER_COMPONENT 10


/* global components data begin *****************************************/
typedef struct components
{
   int component_id;
   int num_of_events;
   int max_num_of_events;
   char **event_names;
   int *event_codes;
   short *event_types;
   int EventSet; //only for testing at initialization phase
} components_t;

components_t *components = NULL;
int num_of_components = 0;
int max_num_of_components = PAPIHL_NUM_OF_COMPONENTS;
int total_num_events = 0;

/* global components data end *******************************************/


/* thread local components data begin ***********************************/
typedef struct local_components
{
   int EventSet;
   /** Return values for the eventsets */
   long_long *values;
} local_components_t;

__thread local_components_t *_local_components = NULL;
__thread long_long _local_cycles;

/* thread local components data end *************************************/


/* global event storage data begin **************************************/
typedef struct reads
{
   struct reads *next;
   struct reads *prev;
   long_long value;        /**< Event value */
} reads_t;

typedef struct
{
   long_long offset;       /**< Event value for region_begin */
   long_long total;        /**< Event value for region_end - region_begin + previous value */
   reads_t *read_values;   /**< List of read event values inside a region */
} value_t;

typedef struct regions
{
   char *region;           /**< Region name */
   struct regions *next;
   struct regions *prev;
   value_t values[];       /**< Array of event values based on current eventset */
} regions_t;

typedef struct
{
   unsigned long key;      /**< Thread ID */
   regions_t *value;       /**< List of regions */
} threads_t;

int compar(const void *l, const void *r)
{
   const threads_t *lm = l;
   const threads_t *lr = r;
   return lm->key - lr->key;
}

typedef struct
{
   void *root;             /**< Root of binary tree */
   threads_t *find_p;      /**< Pointer that is used for finding a thread node */ 
} binary_tree_t;

/**< Global binary tree that stores events from all threads */
binary_tree_t* binary_tree = NULL;

/* global event storage data end ****************************************/


/* global auxiliary variables begin *************************************/
enum region_type { REGION_BEGIN, REGION_READ, REGION_END };

char **requested_event_names = NULL; /**< Events from user or default */
int num_of_requested_events = 0;

bool hl_initiated = false;       /**< Check PAPI-HL has been initiated */
bool hl_finalized = false;       /**< Check PAPI-HL has been fininalized */
bool events_determined = false;  /**< Check if events are determined */
bool output_generated = false;   /**< Check if output has been already generated */
static char *absolute_output_file_path = NULL;
int output_counter = 0;
short verbosity = 0;

/* global auxiliary variables end ***************************************/


static int _internal_onetime_library_init(void);

/* functions for creating eventsets for different components */
static int _internal_checkCounter ( char* counter );
static int _internal_determine_rank();
static char *_internal_remove_spaces( char *str );
static int _internal_hl_determine_default_events();
static int _internal_hl_read_user_events();
static int _internal_hl_new_component(int component_id, components_t *component);
static int _internal_hl_add_event_to_component(char *event_name, int event,
                                        short event_type, components_t *component);
static int _internal_hl_create_components();
static int _internal_hl_read_events(const char* events);
static int _internal_hl_create_event_sets();

/* functions for storing events */
static inline reads_t* _internal_hl_insert_read_node( reads_t** head_node );
static inline int _internal_hl_add_values_to_region( regions_t *node, enum region_type reg_typ );
static inline regions_t* _internal_hl_insert_region_node( regions_t** head_node, const char *region );
static inline regions_t* _internal_hl_find_region_node( regions_t* head_node, const char *region );
static inline threads_t* _internal_hl_insert_thread_node( unsigned long tid );
static inline threads_t* _internal_hl_find_thread_node( unsigned long tid );
static int _internal_hl_store_counters( unsigned long tid, const char *region,
                                        enum region_type reg_typ );
static int _internal_hl_read_counters();
static int _internal_hl_create_global_binary_tree();

/* functions for output generation */
static int _internal_mkdir(const char *dir);
static int _internal_hl_determine_output_path();
static void _internal_hl_write_output();

/* functions for cleaning up heap memory */
static int _internal_clean_up_local_data();
static void _internal_clean_up_global_data();


/* For dynamic linking to libpapi */
/* Weak symbol for pthread_mutex_trylock to avoid additional linking
   against libpthread when not used. */
#pragma weak pthread_mutex_trylock

static int _internal_onetime_library_init(void)
{
   static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
   static int done = 0;
   int retval;

   HLDBG("Initialize!\n");
   /*  failure means we've already initialized or attempted! */
   if (pthread_mutex_trylock(&mutex) == 0) {
      if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) 
         error_at_line(0, retval, __FILE__ ,__LINE__, "PAPI_library_init");
      done = 1;
   reg:
      if ((retval = PAPI_thread_init(&pthread_self)) != PAPI_OK)
         error_at_line(0, retval, __FILE__ ,__LINE__, "PAPI_thread_init");
      HLDBG("Done!\n");
      return ( PAPI_OK );
   } 

   while (!done) {
      HLDBG("Initialization conflict, waiting...\n");
      usleep(10);
   }
   goto reg;
}

static int
_internal_checkCounter ( char* counter )
{
   int EventSet = PAPI_NULL;
   int eventcode;
   int retval;

   if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
      return ( retval );

   if ( ( retval = PAPI_event_name_to_code( counter, &eventcode ) ) != PAPI_OK )
      return ( retval );

   if ( ( retval = PAPI_add_event (EventSet, eventcode) ) != PAPI_OK )
      return ( retval );

   if ( ( retval = PAPI_cleanup_eventset (EventSet) ) != PAPI_OK )
      return ( retval );

   if ( ( retval = PAPI_destroy_eventset (&EventSet) ) != PAPI_OK )
      return ( retval );

   return ( PAPI_OK );
}

static int _internal_determine_rank()
{
   int rank = -1;
   /* check environment variables for rank identification */

   if ( getenv("OMPI_COMM_WORLD_RANK") != NULL )
      rank = atoi(getenv("OMPI_COMM_WORLD_RANK"));
   else if ( getenv("ALPS_APP_PE") != NULL )
      rank = atoi(getenv("ALPS_APP_PE"));
   else if ( getenv("PMI_RANK") != NULL )
      rank = atoi(getenv("PMI_RANK"));
   else if ( getenv("SLURM_PROCID") != NULL )
      rank = atoi(getenv("SLURM_PROCID"));

   return rank;
}

static char *_internal_remove_spaces( char *str )
{
   char *out = str, *put = str;
   for(; *str != '\0'; ++str) {
      if(*str != ' ')
         *put++ = *str;
   }
   *put = '\0';
   return out;
}

static int _internal_hl_determine_default_events()
{
   int i;
   int num_of_defaults = 5;
   char *default_events[num_of_defaults];

   default_events[0] = "perf::TASK-CLOCK";
   default_events[1] = "PAPI_TOT_INS";
   default_events[2] = "PAPI_TOT_CYC";
   default_events[3] = "PAPI_FP_INS";
   default_events[4] = "PAPI_FP_OPS";

   /* allocate memory for requested events */
   requested_event_names = (char**)malloc(num_of_defaults * sizeof(char*));
   if ( requested_event_names == NULL )
      return ( PAPI_ENOMEM );

   /* check if default events are available on the current machine */
   for ( i = 0; i < num_of_defaults; i++ ) {
      if ( _internal_checkCounter( default_events[i] ) == PAPI_OK ) {
         requested_event_names[num_of_requested_events++] = strdup(default_events[i]);
      }
   }

   return ( PAPI_OK );
}

static int _internal_hl_read_user_events(const char *user_events)
{
   char* user_events_copy;
   const char *separator; //separator for events
   int num_of_req_events = 1; //number of events in string
   int req_event_index = 0; //index of event
   const char *position = NULL; //current position in processed string
   char *token;
   
   user_events_copy = strdup(user_events);
   if ( user_events_copy == NULL )
      return ( PAPI_ENOMEM );

   /* check if string is not empty */
   if ( strlen( user_events_copy ) > 0 )
   {
      /* count number of separator characters */
      position = user_events_copy;
      separator=",";
      while ( *position ) {
         if ( strchr( separator, *position ) ) {
            num_of_req_events++;
         }
            position++;
      }

      /* allocate memory for requested events */
      requested_event_names = (char**)malloc(num_of_req_events * sizeof(char*));
      if ( requested_event_names == NULL )
         return ( PAPI_ENOMEM );

      /* parse list of event names */
      token = strtok( user_events_copy, separator );
      while ( token ) {
         if ( req_event_index >= num_of_req_events ){
            /* more entries as in the first run */
            return PAPI_EINVAL;
         }
         requested_event_names[req_event_index] = strdup(_internal_remove_spaces(token));
         if ( requested_event_names[req_event_index] == NULL )
            return ( PAPI_ENOMEM );
         token = strtok( NULL, separator );
         req_event_index++;
      }
   }

   num_of_requested_events = num_of_req_events;
   free(user_events_copy);
   if ( num_of_requested_events == 0 )
      return PAPI_EINVAL;

   return ( PAPI_OK );
}

static int _internal_hl_new_component(int component_id, components_t *component)
{
   int retval;

   /* create new EventSet */
   component->EventSet = PAPI_NULL;
   if ( ( retval = PAPI_create_eventset( &component->EventSet ) ) != PAPI_OK ) {
      verbose_fprintf(stdout, "\nPAPI-HL Error: Cannot create EventSet for component %d.\n", component_id);
      return ( retval );
   }

   component->component_id = component_id;
   component->num_of_events = 0;
   component->max_num_of_events = PAPIHL_NUM_OF_EVENTS_PER_COMPONENT;

   component->event_names = NULL;
   component->event_names = (char**)malloc(component->max_num_of_events * sizeof(char*));
   if ( component->event_names == NULL )
      return ( PAPI_ENOMEM );

   component->event_codes = NULL;
   component->event_codes = (int*)malloc(component->max_num_of_events * sizeof(int));
   if ( component->event_codes == NULL )
      return ( PAPI_ENOMEM );

   component->event_types = NULL;
   component->event_types = (short*)malloc(component->max_num_of_events * sizeof(short));
   if ( component->event_types == NULL )
      return ( PAPI_ENOMEM );

   num_of_components += 1;
   return ( PAPI_OK );
}

static int _internal_hl_add_event_to_component(char *event_name, int event,
                                        short event_type, components_t *component)
{
   int i, retval;

   /* check if we need to reallocate memory for event_names, event_codes and event_types */
   if ( component->num_of_events == component->max_num_of_events ) {
      component->max_num_of_events *= 2;

      component->event_names = (char**)realloc(component->event_names, component->max_num_of_events * sizeof(char*));
      if ( component->event_names == NULL )
         return ( PAPI_ENOMEM );

      component->event_codes = (int*)realloc(component->event_codes, component->max_num_of_events * sizeof(int));
      if ( component->event_codes == NULL )
         return ( PAPI_ENOMEM );

      component->event_types = (short*)realloc(component->event_types, component->max_num_of_events * sizeof(short));
      if ( component->event_types == NULL )
         return ( PAPI_ENOMEM );
   }

   retval = PAPI_add_event( component->EventSet, event );
   if ( retval != PAPI_OK ) {
      const PAPI_component_info_t* cmpinfo;
      cmpinfo = PAPI_get_component_info( component->component_id );
      verbose_fprintf(stdout, "\nPAPI-HL Warning: Cannot add %s to component %s.\n", event_name, cmpinfo->name);
      verbose_fprintf(stdout, "The following event combination is not supported:\n");
      for ( i = 0; i < component->num_of_events; i++ )
         verbose_fprintf(stdout, "  %s\n", component->event_names[i]);
      verbose_fprintf(stdout, "  %s\n", event_name);
      verbose_fprintf(stdout, "Advice: Use papi_event_chooser to obtain an appropriate event set for this component.\n\n");
      return PAPI_EINVAL;
   }

   component->event_names[component->num_of_events] = event_name;
   component->event_codes[component->num_of_events] = event;
   component->event_types[component->num_of_events] = event_type;
   component->num_of_events += 1;

   total_num_events += 1;

   return PAPI_OK;
}

static int _internal_hl_create_components()
{
   int i, j, retval, event;
   int component_id = -1;
   int comp_index = 0;
   bool component_exists = false;
   short event_type = 0;

   components = (components_t*)malloc(max_num_of_components * sizeof(components_t));
   if ( components == NULL )
      return ( PAPI_ENOMEM );

   for ( i = 0; i < num_of_requested_events; i++ ) {
      /* check if requested event contains event type (instant or delta) */
      const char sep = '=';
      char *ret;
      int index;
      /* search for '=' in event name */
      ret = strchr(requested_event_names[i], sep);
      if (ret) {
         if ( strcmp(ret, "=instant") == 0 )
            event_type = 1;
         else
            event_type = 0;

         /* get index of '=' in event name */
         index = (int)(ret - requested_event_names[i]);
         /* remove event type from string */
         requested_event_names[i][index] = '\0';
      }

      /* determine event code and corresponding component id */
      retval = PAPI_event_name_to_code( requested_event_names[i], &event );
      if ( retval != PAPI_OK ) {
         verbose_fprintf(stdout, "\nPAPI-HL Warning: \"%s\" does not exists or is not supported on this machine.\n\n", requested_event_names[i]);
      } else {
         component_id = PAPI_COMPONENT_INDEX( event );

         /* check if component_id already exists in global components structure */
         for ( j = 0; j < num_of_components; j++ ) {
            if ( components[j].component_id == component_id ) {
               component_exists = true;
               comp_index = j;
               break;
            }
            else {
               component_exists = false;
            }
         }

         /* create new component */
         if ( false == component_exists ) {
            /* check if we need to reallocate memory for components */
            if ( num_of_components == max_num_of_components ) {
               max_num_of_components *= 2;
               components = (components_t*)realloc(components, max_num_of_components * sizeof(components_t));
               if ( components == NULL )
                  return ( PAPI_ENOMEM );
            }
            comp_index = num_of_components;
            retval = _internal_hl_new_component(component_id, &components[comp_index]);
            if ( retval != PAPI_OK )
               return ( retval );
         }

         /* add event to current component */
         retval = _internal_hl_add_event_to_component(requested_event_names[i], event, event_type, &components[comp_index]);
         if ( retval != PAPI_OK )
            return ( retval );
      }
   }

   /* destroy all EventSets from global data */
   for ( i = 0; i < num_of_components; i++ ) {
      if ( ( retval = PAPI_cleanup_eventset (components[i].EventSet) ) != PAPI_OK )
         return ( retval );
      if ( ( retval = PAPI_destroy_eventset (&components[i].EventSet) ) != PAPI_OK )
         return ( retval );
      components[i].EventSet = PAPI_NULL;

      // printf("component_id = %d\n", components[i].component_id);
      // printf("num_of_events = %d\n", components[i].num_of_events);
      // for ( j = 0; j < components[i].num_of_events; j++ ) {
      //    printf("  %s type=%d\n", components[i].event_names[j], components[i].event_types[j]);
      // }
   }

   if ( num_of_components == 0 )
      return PAPI_EINVAL;

   return PAPI_OK;
}

static int _internal_hl_read_events(const char* events)
{
   int i, retval;
   if ( events != NULL ) {
      if ( _internal_hl_read_user_events(events) != PAPI_OK )
         if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK )
            return ( retval );

   /* check if user specified events via environment variable */
   } else if ( getenv("PAPI_EVENTS") != NULL ) {
      char *user_events_from_env = strdup( getenv("PAPI_EVENTS") );
      if ( user_events_from_env == NULL )
         return ( PAPI_ENOMEM );
      if ( _internal_hl_read_user_events(user_events_from_env) != PAPI_OK )
         if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK )
            return ( retval );
      free(user_events_from_env);
   } else {
      if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK )
            return ( retval );
   }

   /* create components based on requested events */
   if ( _internal_hl_create_components() != PAPI_OK )
   {
      /* requested events do not work at all, use default events */
      verbose_fprintf(stdout, "\nPAPI-HL Warning: All requested events do not work, using default.\n\n");

      for ( i = 0; i < num_of_requested_events; i++ )
         free(requested_event_names[i]);
      free(requested_event_names);
      num_of_requested_events = 0;
      if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK )
         return ( retval );
      if ( ( retval = _internal_hl_create_components() ) != PAPI_OK )
         return ( retval );
   }

   events_determined = true;
   return ( PAPI_OK );
}

static int _internal_hl_create_event_sets()
{
   int i, j, retval;
   long_long cycles;

   /* allocate memory for local components */
   _local_components = (local_components_t*)malloc(num_of_components * sizeof(local_components_t));
   if ( _local_components == NULL )
      return ( PAPI_ENOMEM );

   for ( i = 0; i < num_of_components; i++ ) {
      /* create EventSet */
      _local_components[i].EventSet = PAPI_NULL;
      if ( ( retval = PAPI_create_eventset( &_local_components[i].EventSet ) ) != PAPI_OK ) {
         return (retval );
      }
      /* add event to current EventSet */
      for ( j = 0; j < components[i].num_of_events; j++ ) {
         retval = PAPI_add_event( _local_components[i].EventSet, components[i].event_codes[j] );
         if ( retval != PAPI_OK ) {
            return (retval );
         }
      }
      /* allocate memory for return values */
      _local_components[i].values = (long_long*)malloc(components[i].num_of_events * sizeof(long_long));
      if ( _local_components[i].values == NULL )
         return ( PAPI_ENOMEM );
   }

   for ( i = 0; i < num_of_components; i++ ) {
      if ( ( retval = PAPI_start( _local_components[i].EventSet ) ) != PAPI_OK )
         return (retval );

      /* warm up PAPI code paths and data structures */
      if ( ( retval = PAPI_read_ts( _local_components[i].EventSet, _local_components[i].values, &cycles ) != PAPI_OK ) ) {
         return (retval );
      }
   }

   return PAPI_OK;
}

static inline reads_t* _internal_hl_insert_read_node(reads_t** head_node)
{
   reads_t *new_node;

   /* create new region node */
   if ( ( new_node = malloc(sizeof(reads_t)) ) == NULL )
      return ( NULL );
   new_node->next = NULL;
   new_node->prev = NULL;

   /* insert node in list */
   if ( *head_node == NULL ) {
      *head_node = new_node;
      return new_node;
   }
   (*head_node)->prev = new_node;
   new_node->next = *head_node;
   *head_node = new_node;

   return new_node;
}

static inline int _internal_hl_add_values_to_region( regions_t *node, enum region_type reg_typ )
{
   int i, j;
   int region_count = 1;
   int cmp_iter = 2;

   if ( reg_typ == REGION_BEGIN ) {
      /* set first fixed counters */
      node->values[0].offset = region_count;
      node->values[1].offset = _local_cycles;
      /* events from components */
      for ( i = 0; i < num_of_components; i++ )
         for ( j = 0; j < components[i].num_of_events; j++ )
            node->values[cmp_iter++].offset = _local_components[i].values[j];
   } else if ( reg_typ == REGION_READ ) {
      /* create a new read node and add values*/
      reads_t* read_node;
      if ( ( read_node = _internal_hl_insert_read_node(&node->values[1].read_values) ) == NULL )
         return ( PAPI_ENOMEM );
      read_node->value = _local_cycles - node->values[1].offset;
      for ( i = 0; i < num_of_components; i++ ) {
         for ( j = 0; j < components[i].num_of_events; j++ ) {
            reads_t* read_node;
            if ( ( read_node = _internal_hl_insert_read_node(&node->values[cmp_iter].read_values) ) == NULL )
               return ( PAPI_ENOMEM );
            if ( components[i].event_types[j] == 1 )
               read_node->value = _local_components[i].values[j];
            else
               read_node->value = _local_components[i].values[j] - node->values[cmp_iter].offset;
            cmp_iter++;
         }
      }
   } else if ( reg_typ == REGION_END ) {
      /* determine difference of current value and offset and add
         previous total value */
      node->values[0].total += node->values[0].offset;
      node->values[1].total += _local_cycles - node->values[1].offset;
      /* events from components */
      for ( i = 0; i < num_of_components; i++ )
         for ( j = 0; j < components[i].num_of_events; j++ ) {
            /* if event type is istant only save last value */
            if ( components[i].event_types[j] == 1 )
               node->values[cmp_iter].total += _local_components[i].values[j];
            else
               node->values[cmp_iter].total += _local_components[i].values[j] - node->values[cmp_iter].offset;
            cmp_iter++;
         }
   }
   return ( PAPI_OK );
}


static inline regions_t* _internal_hl_insert_region_node(regions_t** head_node, const char *region )
{
   regions_t *new_node;
   int i;
   int extended_total_num_events;

   /* number of all events including region count and CPU cycles */
   extended_total_num_events = total_num_events + 2;

   /* create new region node */
   new_node = malloc(sizeof(regions_t) + extended_total_num_events * sizeof(value_t));
   if ( new_node == NULL )
      return ( NULL );
   new_node->region = (char *)malloc((strlen(region) + 1) * sizeof(char));
   if ( new_node->region == NULL )
      return ( NULL );
   new_node->next = NULL;
   new_node->prev = NULL;
   strcpy(new_node->region, region);
   for ( i = 0; i < extended_total_num_events; i++ ) {
      new_node->values[i].total = 0;
      new_node->values[i].read_values = NULL;
   }

   /* insert node in list */
   if ( *head_node == NULL ) {
      *head_node = new_node;
      return new_node;
   }
   (*head_node)->prev = new_node;
   new_node->next = *head_node;
   *head_node = new_node;

   return new_node;
}


static inline regions_t* _internal_hl_find_region_node(regions_t* head_node, const char *region )
{
   regions_t* find_node = head_node;
   while ( find_node != NULL ) {
      if ( strcmp(find_node->region, region) == 0 ) {
         return find_node;
      }
      find_node = find_node->next;
   }
   find_node = NULL;
   return find_node;
}

static inline threads_t* _internal_hl_insert_thread_node(unsigned long tid)
{
   threads_t *new_node = (threads_t*)malloc(sizeof(threads_t));
   if ( new_node == NULL )
      return ( NULL );
   new_node->key = tid;
   new_node->value = NULL; /* head node of region list */
   tsearch(new_node, &binary_tree->root, compar);
   return new_node;
}

static inline threads_t* _internal_hl_find_thread_node(unsigned long tid)
{
   threads_t *find_node = binary_tree->find_p;
   find_node->key = tid;
   void *found = tfind(find_node, &binary_tree->root, compar);
   if ( found != NULL ) {
      find_node = (*(threads_t**)found);
      return find_node;
   }
   return NULL;
}


static int _internal_hl_store_counters( unsigned long tid, const char *region,
                                        enum region_type reg_typ )
{
   int retval;
   threads_t* current_thread_node;

   /* check if current thread is already stored in tree */
   current_thread_node = _internal_hl_find_thread_node(tid);
   if ( current_thread_node == NULL ) {

      _papi_hwi_lock( HIGHLEVEL_LOCK );
         if ( current_thread_node == NULL ) {
         /* insert new node for current thread in tree if type is REGION_BEGIN */
         if ( reg_typ == REGION_BEGIN ) {
            if ( ( current_thread_node = _internal_hl_insert_thread_node(tid) ) == NULL ) {
               _papi_hwi_unlock( HIGHLEVEL_LOCK );
               return ( PAPI_ENOMEM );
            }
         } else {
            _papi_hwi_unlock( HIGHLEVEL_LOCK );
            return ( PAPI_EINVAL );
         }
      }
      _papi_hwi_unlock( HIGHLEVEL_LOCK );
   }

   regions_t* current_region_node;
   /* check if node for current region already exists */
   current_region_node = _internal_hl_find_region_node(current_thread_node->value, region);

   if ( current_region_node == NULL ) {
      /* create new node for current region in list if type is REGION_BEGIN */
      if ( reg_typ == REGION_BEGIN ) {
         if ( ( current_region_node = _internal_hl_insert_region_node(&current_thread_node->value,region) ) == NULL )
            return ( PAPI_ENOMEM );
      } else
         return ( PAPI_EINVAL );
   }

   /* add recorded values to current region */
   if ( ( retval = _internal_hl_add_values_to_region( current_region_node, reg_typ ) ) != PAPI_OK )
      return ( retval );

   return ( PAPI_OK );
}


static int _internal_hl_read_counters()
{
   int i, retval;
   for ( i = 0; i < num_of_components; i++ ) {
      if ( i < ( num_of_components - 1 ) ) {
         retval = PAPI_read( _local_components[i].EventSet, _local_components[i].values);
      } else {
         /* get cycles for last component */
         retval = PAPI_read_ts( _local_components[i].EventSet, _local_components[i].values, &_local_cycles );
      }
      if ( retval != PAPI_OK )
         return ( retval );
   }
   return ( PAPI_OK );
}


static int _internal_hl_create_global_binary_tree()
{
   if ( ( binary_tree = (binary_tree_t*)malloc(sizeof(binary_tree_t)) ) == NULL )
      return ( PAPI_ENOMEM );
   binary_tree->root = NULL;
   if ( ( binary_tree->find_p = (threads_t*)malloc(sizeof(threads_t)) ) == NULL )
      return ( PAPI_ENOMEM );
   return ( PAPI_OK );
}


static int _internal_mkdir(const char *dir)
{
   int retval;
   int errno;
   char *tmp = NULL;
   char *p = NULL;
   size_t len;

   if ( ( tmp = strdup(dir) ) == NULL )
      return ( PAPI_ENOMEM );
   len = strlen(tmp);

   if(tmp[len - 1] == '/')
      tmp[len - 1] = 0;
   for(p = tmp + 1; *p; p++)
   {
      if(*p == '/')
      {
         *p = 0;
         errno = 0;
         retval = mkdir(tmp, S_IRWXU);
         if ( retval != 0 && errno != EEXIST )
            return ( PAPI_ESYS );
         *p = '/';
      }
   }
   retval = mkdir(tmp, S_IRWXU);
   if ( retval != 0 && errno != EEXIST )
      return ( PAPI_ESYS );
   free(tmp);

   return ( PAPI_OK );
}

static int _internal_hl_determine_output_path()
{
   /* check if PAPI_OUTPUT_DIRECTORY is set */
   char *output_prefix = NULL;
   if ( getenv("PAPI_OUTPUT_DIRECTORY") != NULL ) {
      if ( ( output_prefix = strdup( getenv("PAPI_OUTPUT_DIRECTORY") ) ) == NULL )
         return ( PAPI_ENOMEM );
   } else {
      if ( ( output_prefix = strdup( get_current_dir_name() ) ) == NULL )
         return ( PAPI_ENOMEM );
   }
   
   /* generate absolute path for measurement directory */
   if ( ( absolute_output_file_path = (char *)malloc((strlen(output_prefix) + 64) * sizeof(char)) ) == NULL )
      return ( PAPI_ENOMEM );
   if ( output_counter > 0 )
      sprintf(absolute_output_file_path, "%s/papi_%d", output_prefix, output_counter);
   else
      sprintf(absolute_output_file_path, "%s/papi", output_prefix);

   /* check if directory already exists */
   struct stat buf;
   if ( stat(absolute_output_file_path, &buf) == 0 && S_ISDIR(buf.st_mode) ) {

      /* rename old directory by adding a timestamp */
      char *new_absolute_output_file_path = NULL;
      if ( ( new_absolute_output_file_path = (char *)malloc((strlen(absolute_output_file_path) + 64) * sizeof(char)) ) == NULL )
         return ( PAPI_ENOMEM );

      /* create timestamp */
      time_t t = time(NULL);
      struct tm tm = *localtime(&t);
      char m_time[16];
      sprintf(m_time, "%d%d%dT%d%d%d", tm.tm_year+1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
      /* add timestamp to existing folder string */
      sprintf(new_absolute_output_file_path, "%s-%s", absolute_output_file_path, m_time);

      uintmax_t current_unix_time = (uintmax_t)t;
      uintmax_t unix_time_from_old_directory = buf.st_mtime;

      /* This is a workaround for MPI applications!!!
       * Only rename existing measurement directory when it is older than
       * current timestamp. If it's not, we assume that another MPI process already
       * created a new measurement directory. */
      if ( unix_time_from_old_directory < current_unix_time ) {

         if ( rename(absolute_output_file_path, new_absolute_output_file_path) != 0 ) {
            verbose_fprintf(stdout, "\nPAPI-HL Warning: Cannot rename old measurement directory.\n");
            verbose_fprintf(stdout, "If you use MPI, another process may have already renamed the directory.\n\n");
         }
      }

      free(new_absolute_output_file_path);
   }
   free(output_prefix);
   output_counter++;

   return ( PAPI_OK );
}

static void _internal_hl_write_output()
{
   if ( output_generated == false )
   {
      _papi_hwi_lock( HIGHLEVEL_LOCK );
      if ( output_generated == false ) {
         char **all_event_names = NULL;
         int extended_total_num_events;
         unsigned long *tids = NULL;
         int i, j, cmp_iter, number_of_threads;
         FILE *output_file;
         /* current CPU frequency in MHz */
         int cpu_freq;

         /* create new measurement directory */
         if ( ( _internal_mkdir(absolute_output_file_path) ) != PAPI_OK ) {
            verbose_fprintf(stdout, "\nPAPI-HL Error: Cannot create measurement directory %s.\n", absolute_output_file_path);
            return;
         }

         /* determine rank for output file */
         int rank = _internal_determine_rank();

         if ( rank < 0 )
         {
            /* generate unique rank number */
            sprintf(absolute_output_file_path, "%s/rank_XXXXXX", absolute_output_file_path);
            int fd;
            fd = mkstemp(absolute_output_file_path);
            close(fd);
         }
         else
         {
            sprintf(absolute_output_file_path, "%s/rank_%d", absolute_output_file_path, rank);
         }

         /* determine current cpu frequency */
         cpu_freq = PAPI_get_opt( PAPI_CLOCKRATE, NULL );

         output_file = fopen(absolute_output_file_path, "w");

         if ( output_file == NULL )
         {
            verbose_fprintf(stdout, "PAPI-HL Error: Cannot create output file %s!\n", absolute_output_file_path);
            return;
         }
         else
         {
            /* generate array of all events including region count and CPU cycles for output */
            extended_total_num_events = total_num_events + 2;
            all_event_names = (char**)malloc(extended_total_num_events * sizeof(char*));
            all_event_names[0] = "REGION_COUNT";
            all_event_names[1] = "CYCLES";
            cmp_iter = 2;
            for ( i = 0; i < num_of_components; i++ ) {
               for ( j = 0; j < components[i].num_of_events; j++ ) {
                  all_event_names[cmp_iter++] = components[i].event_names[j];
               }
            }

            /* list all threads */
            if ( PAPI_list_threads( tids, &number_of_threads ) != PAPI_OK ) {
               verbose_fprintf(stdout, "PAPI-HL Error: PAPI_list_threads call failed!\n");
               return;
            }
            if ( ( tids = malloc( number_of_threads * sizeof(unsigned long) ) ) == NULL ) {
               verbose_fprintf(stdout, "PAPI-HL Error: OOM!\n");
               return;
            }
            if ( PAPI_list_threads( tids, &number_of_threads ) != PAPI_OK ) {
               verbose_fprintf(stdout, "PAPI-HL Error: PAPI_list_threads call failed!\n");
               return;
            }

            /* example output
            * CPU in MHz:1995
            * Thread,list<Region:list<Event:Value>>
            * 1,<"calc_1":<"PAPI_TOT_INS":57258,"PAPI_TOT_CYC":39439>,"calc_2":<"PAPI_TOT_INS":57258,"    
               PAPI_TOT_CYC":39439>>
            */

            /* write current CPU frequency in output file */
            fprintf(output_file, "CPU in MHz:%d\n", cpu_freq);

            /* write all regions with events per thread */
            fprintf(output_file, "Thread,JSON{Region:{Event:Value,...},...}");
            for ( i = 0; i < number_of_threads; i++ )
            {
               HLDBG("Thread %lu\n", tids[i]);
               /* find values of current thread in global binary tree */
               threads_t* thread_node = _internal_hl_find_thread_node(tids[i]);
               if ( thread_node != NULL ) {
                  /* do we really need the exact thread id? */
                  fprintf(output_file, "\n%lu,{", thread_node->key);

                  /* in case we only store iterator id as thread id */
                  //fprintf(output_file, "\n%d,{", i);

                  /* iterate over regions list */
                  regions_t *regions = thread_node->value;

                  /* going to last node */
                  while ( regions->next != NULL ) {
                     regions = regions->next;
                  }

                  /* read regions in reverse order */
                  while (regions != NULL) {
                     fprintf(output_file, "\"%s\":{", regions->region);

                     for ( j = 0; j < extended_total_num_events; j++ ) {

                        /* print read values if available */
                        if ( regions->values[j].read_values != NULL) {
                           reads_t* read_node = regions->values[j].read_values;
                           /* going to last node */
                           while ( read_node->next != NULL ) {
                              read_node = read_node->next;
                           }
                           /* read values in reverse order */
                           int read_cnt = 1;
                           fprintf(output_file, "\"%s\":{\"Total\":\"%lld\"", all_event_names[j],regions->values[j].total);

                           while ( read_node != NULL ) {
                              fprintf(output_file, ",\"Read_%d\":\"%lld\"", read_cnt,read_node->value);

                              read_node = read_node->prev;

                              if ( read_node == NULL )
                                 fprintf(output_file, "}");
                              if ( read_node == NULL && j < ( extended_total_num_events - 1 ) )
                                 fprintf(output_file, ",");
                              if ( read_node == NULL && j == ( extended_total_num_events - 1 ) )
                                 fprintf(output_file, "}");
                              read_cnt++;
                           }
                        } else {
                           if ( j == ( extended_total_num_events - 1 ) )
                              fprintf(output_file, "\"%s\":\"%lld\"}", all_event_names[j], regions->values[j].total);
                           else
                              fprintf(output_file, "\"%s\":\"%lld\",", all_event_names[j], regions->values[j].total);
                        }
                     }

                     regions = regions->prev;
                     if (regions == NULL )
                        fprintf(output_file, "}");
                     else
                        fprintf(output_file, ",");
                  }
               }
            }
            fprintf(output_file, "\n");
            free(all_event_names);
            fclose(output_file);
            free(tids);
         }

         output_generated = true;
      }
      _papi_hwi_unlock( HIGHLEVEL_LOCK );
   }
}

static int _internal_clean_up_local_data()
{
   int i, retval;
   /* destroy all EventSets from local data */
   for ( i = 0; i < num_of_components; i++ ) {
      if ( ( retval = PAPI_stop( _local_components[i].EventSet, _local_components[i].values ) ) != PAPI_OK )
         return ( retval );
      if ( ( retval = PAPI_cleanup_eventset (_local_components[i].EventSet) ) != PAPI_OK )
         return ( retval );
      if ( ( retval = PAPI_destroy_eventset (&_local_components[i].EventSet) ) != PAPI_OK )
         return ( retval );
      free(_local_components[i].values);
   }
   free(_local_components);
   _local_components = NULL;

   return ( PAPI_OK );
}

static void _internal_clean_up_global_data()
{
   int i;
   int extended_total_num_events;

   /* clean up binary tree of recorded events */
   threads_t *thread_node;
   while ( binary_tree->root != NULL ) {
      thread_node = *(threads_t **)binary_tree->root;

      /* clean up double linked list of region data */
      regions_t *region = thread_node->value;
      regions_t *tmp;
      while ( region != NULL ) {

         /* clean up read node list */
         extended_total_num_events = total_num_events + 2;
         for ( i = 0; i < extended_total_num_events; i++ ) {
            reads_t *read_node = region->values[i].read_values;
            reads_t *read_node_tmp;
            while ( read_node != NULL ) {
               read_node_tmp = read_node;
               read_node = read_node->next;
               free(read_node_tmp);
            }
         }

         tmp = region;
         region = region->next;

         free(tmp->region);
         free(tmp);
      }
      free(region);

      tdelete(thread_node, &binary_tree->root, compar);
      free(thread_node);
    }


   /* clean up global component data */
   for ( i = 0; i < num_of_components; i++ ) {
      free(components[i].event_names);
      free(components[i].event_codes);
      free(components[i].event_types);
   }
   free(components);
   components = NULL;
   num_of_components = 0;
   max_num_of_components = PAPIHL_NUM_OF_COMPONENTS;
   total_num_events = 0;

   /* clean up requested event names */
   for ( i = 0; i < num_of_requested_events; i++ )
      free(requested_event_names[i]);
   free(requested_event_names);
   requested_event_names = NULL;

   num_of_requested_events = 0;
   events_determined = false;
   output_generated = false;
   free(absolute_output_file_path);
   absolute_output_file_path = NULL;
}

/** @class PAPI_hl_init
 * @brief Initializes the high-level PAPI library.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_init();
 *
 * @retval PAPI_OK 
 * @retval PAPI_HIGH_LEVEL_INITED 
 * -- Initialization was already called.
 *
 * PAPI_hl_init initializes the PAPI library and some high-level specific features.
 * If your application is making use of threads you do not need to call any other low level
 * initialization functions as PAPI_hl_init includes thread support.
 * Note that the first call of PAPI_hl_region_begin will automatically call PAPI_hl_init
 * if not already called.
 *
 * @par Example:
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_init();
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @see PAPI_hl_finalize
 * @see PAPI_hl_set_events
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_read
 * @see PAPI_hl_region_end
 * @see PAPI_hl_print_output
 */
int
PAPI_hl_init()
{
   int retval;
   if ( hl_initiated == false && hl_finalized == false )
   {
      if ( ( retval = _internal_onetime_library_init() ) != PAPI_OK )
         return ( retval );
      _papi_hwi_lock( HIGHLEVEL_LOCK );
      if ( hl_initiated == false && hl_finalized == false )
      {
         /* check VERBOSE level */
         if ( getenv("PAPI_VERBOSE") != NULL ) {
            if ( strcmp("1", getenv("PAPI_VERBOSE")) == 0 )
               verbosity = 1;
         }

         /* determine output directory and output file */
         if ( ( retval = _internal_hl_determine_output_path() ) != PAPI_OK ) {
            _papi_hwi_unlock( HIGHLEVEL_LOCK );
            return ( retval );
         }

         /* register the termination function for output */
         atexit(PAPI_hl_print_output);
         hl_initiated = true;
      }
      _papi_hwi_unlock( HIGHLEVEL_LOCK );
      return ( PAPI_OK );
   }
   return ( PAPI_HIGH_LEVEL_INITED );
}

/** @class PAPI_hl_finalize
 * @brief Finalizes the high-level PAPI library.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_finalize( );
 *
 * @retval PAPI_OK
 * @retval PAPI_EINVAL
 * -- Attempting to destroy a non-empty event set or passing in a null pointer to be destroyed.
 * @retval PAPI_ENOEVST
 * -- The EventSet specified does not exist.
 * @retval PAPI_EISRUN
 * -- The EventSet is currently counting events.
 * @retval PAPI_EBUG
 * -- Internal error, send mail to ptools-perfapi@icl.utk.edu and complain.
 *
 * PAPI_hl_finalize finalizes the high-level library by destroying all counting event sets
 * and internal data structures.
 *
 * @par Example:
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_finalize();
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @see PAPI_hl_init
 * @see PAPI_hl_set_events
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_read
 * @see PAPI_hl_region_end
 * @see PAPI_hl_print_output
 */
int PAPI_hl_finalize()
{
   int retval;
   if ( hl_initiated == true ) {
      if ( ( retval = _internal_clean_up_local_data() ) != PAPI_OK )
         return ( retval );
      _papi_hwi_lock( HIGHLEVEL_LOCK );
      if ( hl_initiated == true ) {
         /* clean up data */
         _internal_clean_up_global_data();
         hl_initiated = false;
         hl_finalized = true;
      }
      _papi_hwi_unlock( HIGHLEVEL_LOCK );
   }
   return ( PAPI_OK );
}

/** @class PAPI_hl_set_events
 * @brief Generates event sets based on a list of hardware events.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_set_events( const char* events );
 *
 * @param events
 * -- list of hardware events separated by commas
 *
 * @retval PAPI_OK 
 *
 * PAPI_hl_set_events offers the user the possibility to determine hardware events in
 * the source code as an alternative to the environment variable PAPI_EVENTS.
 * Note that the content of PAPI_EVENTS is ignored if PAPI_hl_set_events was successfully  executed.
 * If the events argument cannot be interpreted, default hardware events are
 * taken for the measurement.
 *
 * @par Example:
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_set_events("PAPI_TOT_INS,PAPI_TOT_CYC");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @see PAPI_hl_init
 * @see PAPI_hl_finalize
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_read
 * @see PAPI_hl_region_end
 * @see PAPI_hl_print_output
 */
int
PAPI_hl_set_events(const char* events)
{
   int retval;
   if ( hl_initiated == true ) {
      if ( events_determined == false )
      {
         _papi_hwi_lock( HIGHLEVEL_LOCK );
         if ( events_determined == false )
         {
            if ( ( retval = _internal_hl_read_events(events) ) != PAPI_OK ) {
               _papi_hwi_unlock( HIGHLEVEL_LOCK );
               return ( retval );
            }
            if ( ( retval = _internal_hl_create_global_binary_tree() ) != PAPI_OK ) {
               _papi_hwi_unlock( HIGHLEVEL_LOCK );
               return ( retval );
            }
         }
         _papi_hwi_unlock( HIGHLEVEL_LOCK );
      }
   }
   return ( PAPI_OK );
}

/** @class PAPI_hl_print_output
 * @brief Prints values of hardware events.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * void PAPI_hl_print_output( );
 *
 * PAPI_hl_print_output prints the measured values of hardware events in one file for serial
 * or thread parallel applications.
 * Multi-processing applications, such as MPI, will have one output file per process.
 * Each output file contains measured values of all threads.
 * The entire measurement can be converted in a better readable output via python.
 * For more information, see <a href="https://bitbucket.org/icl/papi/wiki/papi-hl.md">High Level API</a>.
 * Note that if PAPI_hl_print_output is not called explicitly PAPI will try to generate output
 * at the end of the application. However, for some reason, this feature sometimes does not  work.
 * It is therefore recommended to call PAPI_hl_print_output for larger applications.
 *
 * @par Example:
 *
 * @code
 *
 * PAPI_hl_print_output();
 *
 * @endcode
 *
 * @see PAPI_hl_init
 * @see PAPI_hl_finalize
 * @see PAPI_hl_set_events
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_read
 * @see PAPI_hl_region_end 
 */
void
PAPI_hl_print_output()
{
   if ( hl_initiated == true ) {
      if ( output_generated == false )
         _internal_hl_write_output();
   }
}

/** @class PAPI_hl_region_begin
 * @brief Reads and stores hardware events at the beginning of an instrumented code region.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_region_begin( const char* region );
 *
 * @param region
 * -- a unique region name
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 *
 * PAPI_hl_region_begin reads hardware events and stores them internally at the beginning
 * of an instrumented code region.
 * If not specified via environment variable PAPI_EVENTS, default events are used.
 * The first call sets all counters implicitly to zero and starts counting.
 * Note that if PAPI_EVENTS is not set or cannot be interpreted, default hardware events are
 * recorded.
 *
 * @par Example:
 *
 * @code
 * export PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC"
 * @endcode
 *
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_region_begin("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 *  //Do some computation here
 *
 * retval = PAPI_hl_region_end("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @see PAPI_hl_read
 * @see PAPI_hl_region_end
 */
int
PAPI_hl_region_begin( const char* region )
{
   int retval;

   if ( hl_finalized == true )
      return ( PAPI_ENOTRUN );

   if ( hl_initiated == false ) {
      if ( ( retval = PAPI_hl_init() ) != PAPI_OK )
         return ( retval );
   }

   if ( events_determined == false ) {
      if ( ( retval = PAPI_hl_set_events(NULL) ) != PAPI_OK )
         return ( retval );
   }

   if ( _local_components == NULL ) {
      if ( ( retval = _internal_hl_create_event_sets() ) != PAPI_OK ) {
         _local_components = NULL;
         return ( retval );
      }
   }

   /* read all events */
   if ( ( retval = _internal_hl_read_counters() ) != PAPI_OK )
      return ( retval );

   /* store all events */
   if ( ( retval = _internal_hl_store_counters( PAPI_thread_id(), region, REGION_BEGIN) ) != PAPI_OK )
      return ( retval );

   return ( PAPI_OK );
}

/** @class PAPI_hl_read
 * @brief Reads and stores hardware events inside of an instrumented code region.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_read( const char* region );
 *
 * @param region
 * -- a unique region name corresponding to PAPI_hl_region_begin
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 *
 * PAPI_hl_read reads hardware events and stores them internally inside
 * of an instrumented code region.
 * Assumes that PAPI_hl_region_begin was called before.
 *
 * @par Example:
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_region_begin("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 *  //Do some computation here
 *
 * retval = PAPI_hl_read("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 *  //Do some computation here
 *
 * retval = PAPI_hl_region_end("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_region_end
 */
int
PAPI_hl_read(const char* region)
{
   int retval;

   if ( _local_components == NULL )
      return ( PAPI_ENOTRUN );

   /* read all events */
   if ( ( retval = _internal_hl_read_counters() ) != PAPI_OK )
      return ( retval );

   /* store all events */
   if ( ( retval = _internal_hl_store_counters( PAPI_thread_id(), region, REGION_READ) ) != PAPI_OK )
      return ( retval );

   return ( PAPI_OK );
}

/** @class PAPI_hl_region_end
 * @brief Reads and stores hardware events at the end of an instrumented code region.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_region_end( const char* region );
 *
 * @param region
 * -- a unique region name corresponding to PAPI_hl_region_begin
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 *
 * PAPI_hl_region_end reads hardware events and stores the difference to the values from
 * PAPI_hl_region_begin at the end of an instrumented code region.
 * Assumes that PAPI_hl_region_begin was called before.
 * Note that an output is automatically generated when your application terminates.
 * If the automatic output does not work for any reason, PAPI_hl_print_output must be called.
 * 
 *
 * @par Example:
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_region_begin("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 *  //Do some computation here
 *
 * retval = PAPI_hl_region_end("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_read
 */
int
PAPI_hl_region_end( const char* region )
{
   int retval;

   if ( _local_components == NULL )
      return ( PAPI_ENOTRUN );

   /* read all events */
   if ( ( retval = _internal_hl_read_counters() ) != PAPI_OK )
      return ( retval );

   /* store all events */
   if ( ( retval = _internal_hl_store_counters( PAPI_thread_id(), region, REGION_END) ) != PAPI_OK )
      return ( retval );

   return ( PAPI_OK );
}

