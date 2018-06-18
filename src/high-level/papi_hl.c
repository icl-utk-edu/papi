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
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include "papi.h"
#include "papi_hl_priv.h"

/* For dynamic linking to libpapi */
/* Weak symbol for pthread_mutex_trylock to avoid additional linking
 * against libpthread when not used.*/
#pragma weak pthread_mutex_trylock
int pthread_mutex_trylock(pthread_mutex_t *mutex); __attribute__((weak));

void _internal_onetime_library_init(void)
{
   static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
   static int done = 0;
   int retval;

   APIDBG("Initialize!\n");
   /*  failure means already we've already initialized or attempted! */
   if (pthread_mutex_trylock(&mutex) == 0) {
      if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) 
      error_at_line(0, retval, __FILE__ ,__LINE__, "PAPI_library_init"); 
      if ((retval = PAPI_thread_init(&pthread_self)) != PAPI_OK)
      error_at_line(0, retval, __FILE__ ,__LINE__, "PAPI_thread_init"); 
      if ((retval = PAPI_multiplex_init()) != PAPI_OK)
      error_at_line(0, retval, __FILE__ ,__LINE__, "PAPI_multiplex_init"); 
      if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
      error_at_line(0, retval, __FILE__ ,__LINE__, "PAPI_set_debug");
      done = 1;
   reg:
      if ((retval = PAPI_register_thread()) != PAPI_OK)
      error_at_line(0, retval, __FILE__ ,__LINE__, "PAPI_register_thread");
      APIDBG("Done!\n");
      return;
   } 

   while (!done) {
      APIDBG("Initialization conflict, waiting...\n");
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

int _internal_determine_rank()
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

char *_internal_remove_spaces( char *str )
{
   char *out = str, *put = str;
   for(; *str != '\0'; ++str) {
      if(*str != ' ')
         *put++ = *str;
   }
   *put = '\0';
   return out;
}

int _internal_hl_determine_default_events()
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

   /* check if default events are available on the current machine */
   for ( i = 0; i < num_of_defaults; i++ ) {
      if ( _internal_checkCounter( default_events[i] ) == PAPI_OK ) {
         requested_event_names[num_of_requested_events++] = strdup(default_events[i]);
      }
   }

   return ( PAPI_OK );
}

int _internal_hl_read_user_events(const char *user_events)
{
   char* user_events_copy;
   const char *separator; //separator for events
   int num_of_req_events = 1; //number of events in string
   int req_event_index = 0; //index of event
   const char *position = NULL; //current position in processed string
   char *token;
   
   user_events_copy = strdup(user_events);

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

      /* parse list of event names */
      token = strtok( user_events_copy, separator );
      while ( token ) {
         if ( req_event_index >= num_of_req_events ){
            /* more entries as in the first run */
            return PAPI_EINVAL;
         }
         requested_event_names[req_event_index] = strdup(_internal_remove_spaces(token));
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

void _internal_hl_new_component(int component_id, components_t *component)
{
   int retval;

   /* create new EventSet */
   component->EventSet = PAPI_NULL;
   if ( ( retval = PAPI_create_eventset( &component->EventSet ) ) != PAPI_OK ) {
      verbose_fprintf(stdout, "\nPAPI-HL Error: Cannot create EventSet for component %d.\n", component_id);
      exit(EXIT_FAILURE);
   }

   component->component_id = component_id;
   component->num_of_events = 0;
   component->max_num_of_events = PAPIHL_NUM_OF_EVENTS_PER_COMPONENT;
   component->event_names = NULL;
   component->event_names = (char**)malloc(component->max_num_of_events * sizeof(char*));
   component->event_types = NULL;
   component->event_types = (short*)malloc(component->max_num_of_events * sizeof(short));

   num_of_components += 1;
}

int _internal_hl_add_event_to_component(char *event_name, int event,
                                        short event_type, components_t *component)
{
   int i, retval;

   /* check if we need to reallocate memory for event_names and event_types */
   if ( component->num_of_events == component->max_num_of_events ) {
      component->max_num_of_events *= 2;
      component->event_names = (char**)realloc(component->event_names, component->max_num_of_events * sizeof(char*));
      component->event_types = (short*)realloc(component->event_types, component->max_num_of_events * sizeof(short));
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
   component->event_types[component->num_of_events] = event_type;
   component->num_of_events += 1;

   total_num_events += 1;

   return PAPI_OK;
}

int _internal_hl_create_components()
{
   int i, j, retval, event;
   int component_id = -1;
   int comp_index = 0;
   bool component_exists = false;
   short event_type = 0;

   components = (components_t*)malloc(max_num_of_components * sizeof(components_t));

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
            }
            comp_index = num_of_components;
            _internal_hl_new_component(component_id, &components[comp_index]);
         }

         /* add event to current component */
         _internal_hl_add_event_to_component(requested_event_names[i], event, event_type, &components[comp_index]);
      }
   }

   /* destroy all EventSets from global data */
   for ( i = 0; i < num_of_components; i++ ) {
      if ( ( retval = PAPI_cleanup_eventset (components[i].EventSet) ) != PAPI_OK )
         exit(EXIT_FAILURE);
      if ( ( retval = PAPI_destroy_eventset (&components[i].EventSet) ) != PAPI_OK )
         exit(EXIT_FAILURE);
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

int _internal_hl_read_events(const char* events)
{
   int i;
   if ( events != NULL ) {
      if ( _internal_hl_read_user_events(events) != PAPI_OK )
         _internal_hl_determine_default_events();

   /* check if user specified events via environment variable */
   } else if ( getenv("PAPI_EVENTS") != NULL ) {
      char *user_events_from_env = strdup( getenv("PAPI_EVENTS") );
      if ( _internal_hl_read_user_events(user_events_from_env) != PAPI_OK )
         _internal_hl_determine_default_events();
      free(user_events_from_env);
   } else {
      _internal_hl_determine_default_events();
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
      _internal_hl_determine_default_events();
      _internal_hl_create_components();
   }

   events_determined = true;
   return ( PAPI_OK );
}

int _internal_hl_create_event_sets()
{
   int i, j, event, retval;
   long_long cycles;

   /* allocate memory for local components */
   _local_components = (local_components_t*)malloc(num_of_components * sizeof(local_components_t));
   for ( i = 0; i < num_of_components; i++ ) {
      /* create EventSet */
      _local_components[i].EventSet = PAPI_NULL;
      if ( ( retval = PAPI_create_eventset( &_local_components[i].EventSet ) ) != PAPI_OK ) {
         exit(EXIT_FAILURE);
      }
      /* add event to current EventSet */
      for ( j = 0; j < components[i].num_of_events; j++ ) {
         retval = PAPI_event_name_to_code( components[i].event_names[j], &event );
         if ( retval != PAPI_OK ) {
            exit(EXIT_FAILURE);
         }
         retval = PAPI_add_event( _local_components[i].EventSet, event );
         if ( retval != PAPI_OK ) {
            exit(EXIT_FAILURE);
         }
      }
      /* allocate memory for return values */
      _local_components[i].values = (long_long*)malloc(components[i].num_of_events * sizeof(long_long));
   }

   for ( i = 0; i < num_of_components; i++ ) {
      if ( PAPI_start( _local_components[i].EventSet ) != PAPI_OK )
         exit(EXIT_FAILURE);

      /* warm up PAPI code paths and data structures */
      if ( PAPI_read_ts( _local_components[i].EventSet, _local_components[i].values, &cycles ) != PAPI_OK ) {
         exit(EXIT_FAILURE);
      }
   }

   return PAPI_OK;
}

reads_t* _internal_hl_insert_read_node(reads_t** head_node)
{
   reads_t *new_node;

   /* create new region node */
   new_node = malloc(sizeof(reads_t));
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

void _internal_hl_add_values_to_region( regions_t *node, long_long cycles,
                                        enum region_type reg_typ )
{
   int i, j;
   int region_count = 1;
   int cmp_iter = 2;

   if ( reg_typ == REGION_BEGIN ) {
      /* set first fixed counters */
      node->values[0].offset = region_count;
      node->values[1].offset = cycles;
      /* events from components */
      for ( i = 0; i < num_of_components; i++ )
         for ( j = 0; j < components[i].num_of_events; j++ )
            node->values[cmp_iter++].offset = _local_components[i].values[j];
   } else if ( reg_typ == REGION_READ ) {
      /* create a new read node and add values*/
      reads_t* read_node = _internal_hl_insert_read_node(&node->values[1].read_values);
      read_node->value = cycles - node->values[1].offset;
      for ( i = 0; i < num_of_components; i++ ) {
         for ( j = 0; j < components[i].num_of_events; j++ ) {
            reads_t* read_node = _internal_hl_insert_read_node(&node->values[cmp_iter].read_values);
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
      node->values[1].total += cycles - node->values[1].offset;
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
}


regions_t* _internal_hl_insert_region_node(regions_t** head_node, const char *region )
{
   regions_t *new_node;
   int i;
   int extended_total_num_events;

   /* number of all events including region count and CPU cycles */
   extended_total_num_events = total_num_events + 2;

   /* create new region node */
   new_node = malloc(sizeof(regions_t) + extended_total_num_events * sizeof(value_t));
   new_node->region = (char *)malloc((strlen(region) + 1) * sizeof(char));
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


regions_t* _internal_hl_find_region_node(regions_t* head_node, const char *region )
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

threads_t* _internal_hl_insert_thread_node(unsigned long tid)
{
   threads_t *new_node = (threads_t*)malloc(sizeof(threads_t));
   new_node->key = tid;
   new_node->value = NULL; /* head node of region list */
   tsearch(new_node, &binary_tree->root, compar);
   return new_node;
}

threads_t* _internal_hl_find_thread_node(unsigned long tid)
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


int _internal_hl_store_values( unsigned long tid, const char *region,
                               long_long cycles, enum region_type reg_typ )
{
   threads_t* current_thread_node;

   /* check if current thread is already stored in tree */
   current_thread_node = _internal_hl_find_thread_node(tid);
   if ( current_thread_node == NULL ) {
      /* insert new node for current thread in tree if type is REGION_BEGIN */
      if ( reg_typ == REGION_BEGIN ) {
         current_thread_node = _internal_hl_insert_thread_node(tid);
      } else
         return ( PAPI_EINVAL );
   }

   regions_t* current_region_node;
   /* check if node for current region already exists */
   current_region_node = _internal_hl_find_region_node(current_thread_node->value, region);

   if ( current_region_node == NULL ) {
      /* create new node for current region in list if type is REGION_BEGIN */
      if ( reg_typ == REGION_BEGIN ) {
         current_region_node = _internal_hl_insert_region_node(&current_thread_node->value,region);
      } else
         return ( PAPI_EINVAL );
   }

   /* add recorded values to current region */
   _internal_hl_add_values_to_region( current_region_node, cycles, reg_typ );

   return ( PAPI_OK );
}


void _internal_hl_create_global_binary_tree()
{
   binary_tree = (binary_tree_t*)malloc(sizeof(binary_tree_t));
   binary_tree->root = NULL;
   binary_tree->find_p = (threads_t*)malloc(sizeof(threads_t));
}


static void _internal_mkdir(const char *dir)
{
   char *tmp = NULL;
   char *p = NULL;
   size_t len;

   tmp = strdup(dir);
   len = strlen(tmp);

   if(tmp[len - 1] == '/')
      tmp[len - 1] = 0;
   for(p = tmp + 1; *p; p++)
   {
      if(*p == '/')
      {
         *p = 0;
         mkdir(tmp, S_IRWXU);
         *p = '/';
      }
   }
   mkdir(tmp, S_IRWXU);
   free(tmp);
}

void _internal_hl_determine_output_path()
{
   /* check if PAPI_OUTPUT_DIRECTORY is set */
   char *output_prefix = NULL;
   if ( getenv("PAPI_OUTPUT_DIRECTORY") != NULL )
      output_prefix = strdup( getenv("PAPI_OUTPUT_DIRECTORY") );
   else
      output_prefix = strdup( get_current_dir_name() );
   
   /* generate absolute path for measurement directory */
   absolute_output_file_path = (char *)malloc((strlen(output_prefix) + 64) * sizeof(char));
   if ( output_counter > 0 )
      sprintf(absolute_output_file_path, "%s/papi_%d", output_prefix, output_counter);
   else
      sprintf(absolute_output_file_path, "%s/papi", output_prefix);

   /* check if directory already exists */
   struct stat buf;
   if ( stat(absolute_output_file_path, &buf) == 0 && S_ISDIR(buf.st_mode) ) {

      /* rename old directory by adding a timestamp */
      char *new_absolute_output_file_path = NULL;
      new_absolute_output_file_path = (char *)malloc((strlen(absolute_output_file_path) + 64) * sizeof(char));

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
}

void _internal_hl_write_output()
{
   if ( output_generated == false )
   {
      PAPI_lock(PAPIHL_LOCK);
      if ( output_generated == false ) {
         char **all_event_names = NULL;
         int extended_total_num_events;
         unsigned long *tids = NULL;
         int i, j, cmp_iter, number_of_threads;
         FILE *output_file;
         /* current CPU frequency in MHz */
         int cpu_freq;

         /* create new measurement directory */
         _internal_mkdir(absolute_output_file_path);

         /* determine rank for output file */
         int rank = _internal_determine_rank();

         if ( rank < 0 )
         {
            /* generate unique rank number */
            sprintf(absolute_output_file_path, "%s/rank_XXXXXX", absolute_output_file_path);
            mkstemp(absolute_output_file_path);
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
            PAPI_list_threads( tids, &number_of_threads );
            tids = malloc( number_of_threads * sizeof(unsigned long) );
            PAPI_list_threads( tids, &number_of_threads );

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
               APIDBG("Thread %lu\n", tids[i]);
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
      PAPI_unlock( PAPIHL_LOCK );
   }
}

void _internal_clean_up_local_data()
{
   int i;
   /* destroy all EventSets from local data */
   for ( i = 0; i < num_of_components; i++ ) {
      if ( PAPI_stop( _local_components[i].EventSet, _local_components[i].values ) != PAPI_OK )
         exit(EXIT_FAILURE);
      if ( PAPI_cleanup_eventset (_local_components[i].EventSet) != PAPI_OK )
         exit(EXIT_FAILURE);
      if ( PAPI_destroy_eventset (&_local_components[i].EventSet) != PAPI_OK )
         exit(EXIT_FAILURE);
      free(_local_components[i].values);
   }
   free(_local_components);
   _local_components = NULL;
}

void _internal_clean_up_global_data()
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

int
PAPI_hl_init()
{
   if ( hl_initiated == false && hl_finalized == false )
   {
      _internal_onetime_library_init();
      PAPI_lock( PAPIHL_LOCK );
      if ( hl_initiated == false && hl_finalized == false )
      {
         /* check VERBOSE level */
         if ( getenv("PAPI_VERBOSE") != NULL ) {
            if ( strcmp("1", getenv("PAPI_VERBOSE")) == 0 )
               verbosity = 1;
         }

         /* determine output directory and output file */
         _internal_hl_determine_output_path();

         /* register the termination function for output */
         atexit(PAPI_hl_print_output);
         hl_initiated = true;
      }
      PAPI_unlock( PAPIHL_LOCK );
   }

   return ( PAPI_OK );
}

int PAPI_hl_finalize()
{
   if ( hl_initiated == true ) {
      _internal_clean_up_local_data();
      PAPI_lock( PAPIHL_LOCK );
      if ( hl_initiated == true ) {
         /* clean up data */
         _internal_clean_up_global_data();
         hl_initiated = false;
         hl_finalized = true;
      }
      PAPI_unlock( PAPIHL_LOCK );
   }
   return ( PAPI_OK );
}

int
PAPI_hl_set_events(const char* events)
{
   if ( hl_initiated == true ) {
      if ( events_determined == false )
      {
         PAPI_lock( PAPIHL_LOCK );
         if ( events_determined == false )
         {
            _internal_hl_read_events(events);
            _internal_hl_create_global_binary_tree();
         }
         PAPI_unlock( PAPIHL_LOCK );
      }
   }
   return ( PAPI_OK );
}

void
PAPI_hl_print_output()
{
   if ( hl_initiated == true ) {
      if ( output_generated == false )
         _internal_hl_write_output();
   }

   /* test */
   if ( hl_initiated == true ) {
      _internal_clean_up_local_data();
      PAPI_lock( PAPIHL_LOCK );
      if ( hl_initiated == true ) {
         /* clean up data */
         _internal_clean_up_global_data();
         _internal_hl_determine_output_path();
      }
      PAPI_unlock( PAPIHL_LOCK );
   }
}

int
PAPI_hl_region_begin( const char* region )
{
   int i;
   long_long cycles;

   if ( hl_finalized == true )
      return ( PAPI_EINVAL );

   if ( hl_initiated == false )
      PAPI_hl_init();

   if ( events_determined == false )
      PAPI_hl_set_events(NULL);

   if ( _local_components == NULL )
      _internal_hl_create_event_sets();

   for ( i = 0; i < num_of_components; i++ ) {
      if ( PAPI_read_ts( _local_components[i].EventSet, _local_components[i].values, &cycles ) != PAPI_OK )
         exit(EXIT_FAILURE);
   }

   /* store all offset values (cycles has the value of the last event set) */
   PAPI_lock( PAPIHL_LOCK );
   _internal_hl_store_values( PAPI_thread_id(), region, cycles, REGION_BEGIN);
   PAPI_unlock( PAPIHL_LOCK );

   return ( PAPI_OK );
}

int
PAPI_hl_read(const char* region)
{
   int i;
   long_long cycles;

   if ( _local_components == NULL )
      return ( PAPI_EINVAL );

   for ( i = 0; i < num_of_components; i++ ) {
      if ( PAPI_read_ts( _local_components[i].EventSet, _local_components[i].values, &cycles ) != PAPI_OK )
         exit(EXIT_FAILURE);
   }

   /* store all offset values (cycles has the value of the last event set) */
   PAPI_lock( PAPIHL_LOCK );
   _internal_hl_store_values( PAPI_thread_id(), region, cycles, REGION_READ);
   PAPI_unlock( PAPIHL_LOCK );

   return ( PAPI_OK );
}

int
PAPI_hl_region_end( const char* region )
{
   int i;
   long_long cycles;

   if ( _local_components == NULL )
      return ( PAPI_EINVAL );

   for ( i = 0; i < num_of_components; i++ ) {
      if ( PAPI_read_ts( _local_components[i].EventSet, _local_components[i].values, &cycles ) != PAPI_OK )
         exit(EXIT_FAILURE);
   }

   /* store all values (cycles has the value of the last event set) */
   PAPI_lock( PAPIHL_LOCK );
   _internal_hl_store_values( PAPI_thread_id(), region, cycles, REGION_END);
   PAPI_unlock( PAPIHL_LOCK );

   return ( PAPI_OK );
}

