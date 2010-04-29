#ifndef _HAVE_HOST_COUNTER_H_
#define _HAVE_HOST_COUNTER_H_

#include <stdint.h>
//#include <dataheap_misc.h>
#include <time.h>

#define COUNTER_TYPE_ACCUMULATING   1
#define COUNTER_TYPE_ABSOLUTE       2

#define MAGIC1 0x0123456789ACDDEFULL
#define MAGIC2 0xDEADBEEFDEADBEEFULL

#define MAX_SUBSCRIBED_COUNTER 100

/**
 * describes a single counter with its properties as
 * it is being presented to VT
 */
typedef struct counter_info_struct
{
    int idx;
    char*    name;
    char*    description;
    char*    unit;
    uint64_t value;
    struct counter_info_struct* next;
} counter_info;

typedef struct {
    uint64_t magic1;
    uint64_t magic2;
    uint64_t sizeof_new_counter_log;
    uint64_t sizeof_value_event;
} trace_header;

typedef enum {
    LOG_TYPE_NEW_COUNTER,
    LOG_TYPE_VALUE,
} logtype;

typedef struct {
    char log_type;
    char counter_type;
    int counter_id;
    char name[128];
    char description[64];
    char unit[64];
} new_counter_log;

typedef struct {
    char log_type;
    int counter_id;
    unsigned long long value;
    struct timespec timestamp;
} value_event;

typedef struct  {
  int count;
  char **data;
} string_list;

/**
 * lists all counters known
 * @return a string list
 */
string_list *host_listCounter();

/** 
 * deletes a string list
 */
void host_deleteStringList( string_list* to_delete );

/**
 * subscribes to a one counter to get
 * updates for counter values
 * @param name the name of the counter
 * @return the counter id, 0 on failure
 */
uint64_t host_subscribe( const char *name );

/**
 * gets a human readable description for a counter
 * @param name the name of the counter
 * @return a description
 */
const char *host_description( const char *name );

/**
 * gets a measument unit for a counter
 * @param name the name of the counter
 * @return a unit
 */
const char *host_unit( const char *name );

/**
 * initializes the connection to the server and announces
 * the object type on the server side. once this function
 * returned the other functions can be called
 */
void host_initialize();

/**
 * reads the current values and stores them in the provided
 * memory location
 */
void host_read_values( long long *data);

/**
 * finalizes the interface, returns when the started thread has finished
 */
void host_finalize();

#endif /* _HAVE_HOST_COUNTER_H_ */
