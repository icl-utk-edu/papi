/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
* @file     papi_hl_priv.h
* @author   Frank Winkler
*           frank.winkler@icl.utk.edu
* @author   Philip Mucci
*           mucci@icl.utk.edu
* @brief    This file contains private, library internal definitons for the high level interface
*           to PAPI. 
*/

#ifndef PAPI_HL_PRIV_H
#define PAPI_HL_PRIV_H

#ifdef CONFIG_PAPIHLLIB_DEBUG
#define APIDBG(format, args...) fprintf(stderr, format, ## args)
#else
#define APIDBG(format, args...) { ; }
#endif

#define verbose_fprintf \
   if (verbosity == 1) fprintf

#define PAPIHL_LOCK PAPI_LOCK_USR2

/* these should be exported */
#define PAPIHL_NUM_OF_COMPONENTS 10
#define PAPIHL_NUM_OF_EVENTS_PER_COMPONENT 10


/* global components data begin *****************************************/
typedef struct components
{
   int component_id;
   int num_of_events;
   int max_num_of_events;
   char **event_names;
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


void _internal_onetime_library_init(void);

/* functions for creating eventsets for different components */
static int _internal_checkCounter ( char* counter );
int _internal_determine_rank();
char *_internal_remove_spaces( char *str );
int _internal_hl_determine_default_events();
int _internal_hl_read_user_events();
void _internal_hl_new_component(int component_id, components_t *component);
int _internal_hl_add_event_to_component(char *event_name, int event,
                                        short event_type, components_t *component);
int _internal_hl_create_components();
int _internal_hl_read_events(const char* events);
int _internal_hl_create_event_sets();

/* functions for storing events */
reads_t* _internal_hl_insert_read_node( reads_t** head_node );
void _internal_hl_add_values_to_region( regions_t *node, long_long cycles,
                                        enum region_type reg_typ );
regions_t* _internal_hl_insert_region_node( regions_t** head_node, const char *region );
regions_t* _internal_hl_find_region_node( regions_t* head_node, const char *region );
threads_t* _internal_hl_insert_thread_node( unsigned long tid );
threads_t* _internal_hl_find_thread_node( unsigned long tid );
int _internal_hl_store_values( unsigned long tid, const char *region,
                               long_long cycles, enum region_type reg_typ );
void _internal_hl_create_global_binary_tree();

/* functions for output generation */
static void _internal_mkdir(const char *dir);
void _internal_hl_determine_output_path();
void _internal_hl_write_output();

/* functions for cleaning up heap memory */
void _internal_clean_up_local_data();
void _internal_clean_up_global_data();

#endif
