/* file: papi.h */

/*
Return Codes

All of the functions contained in the PerfAPI return standardized error codes.
Values greater than or equal to zero indicate success, less than zero indicates
failure. 
*/

#define PAPI_OK_MPX    1  /*No error, multiplexing has been enabled 
                            and is now active*/
#define PAPI_OK        0  /*No error*/
#define PAPI_EINVAL   -1  /*Invalid argument*/
#define PAPI_ENOMEM   -2  /*Insufficient memory*/
#define PAPI_ESYS     -3  /*A System C library call failed, please check errno*/
#define PAPI_ESBSTR   -4  /*Substrate returned an error, 
			    usually the result of an unimplemented feature*/
#define PAPI_ECLOST   -5  /*Access to the counters was lost or interrupted*/
#define PAPI_EBUG     -6  /*Internal error, please send mail to the developers*/
#define PAPI_ENOEVNT  -7  /*Hardware Event does not exist*/
#define PAPI_ECNFLCT  -8  /*Hardware Event exists, but cannot be counted 
                            due to counter resource limitations*/ 
#define PAPI_ENOTRUN  -9  /*No Events or EventSets are currently counting*/
#define PAPI_EMISC   -10 /* No clue as to what this error code means */

/*
Constants

All of the functions in the PerfAPI should use the following set of constants.
*/

#define PAPI_NULL       -1    /*A nonexistent hardware event used as a placeholder*/ 
#define PAPI_USER        0    /*Counts are accumulated for events occuring in the 
				the user context*/
#define PAPI_KERNEL	 1    /*Counts are accumulated for events occurring in the
				the kernel context*/
#define PAPI_SYSTEM	 2    /*Counts are accumulated for events occuring in
			 	either the user context or the kernel context*/

#define PAPI_PER_THR     0    /*Counts are accumulated on a per kernel thread basis*/ 	
#define PAPI_PER_PROC    1    /*Counts are accumulated on a per process basis*/
#define PAPI_PER_CPU     2    /*Counts are accumulated on a per cpu basis*/
#define PAPI_PER_NODE    3    /*Counts are accumulated on a per node or 
				processor basis*/ 

#define PAPI_RUNNING     1    /*EventSet is running*/
#define PAPI_STOPPED     2    /*EventSet is stopped*/ 
#define PAPI_PAUSED      3    /*EventSet is temporarily disabled by the library*/
#define PAPI_NOT_INIT    4    /*EventSet defined, but has not yet been initiated*/
			      /* :::::PAPI_NOT_INIT NOT IN STANDARD:::::   	
				The constant PAPI_NOT_INIT is not in the standard, 
				but this definition is reserved for future use.*/  

#define PAPI_NUM_ERRORS  11   /* Number of error messages spec'd */
#define PAPI_QUIET       0    /*Option to not do any automatic error reporting 
				to stderr*/
#define PAPI_VERB_ECONT  1    /*Option to automatically report any return codes <0 
				to stderr [error-continue]*/ 
#define PAPI_VERB_ESTOP  2    /*Option to automatically report any error codes < 0 
				to stderr and call exit(PAPI_ERROR) [error-stop]*/

#define PAPI_SET_MPXRES  1    /*Option to enable and set the resolution of the 
				multiplexing hardware*/
#define PAPI_GET_MPXRES  2    /*Option to query the status of the 
                                multiplexing software*/
#define PAPI_DEF_MPXRES  1000 /*Default resolution in microseconds of the 
				multiplexing software*/

#define PAPI_DEBUG	 3    /*Option to turn on debugging features of 
				the PerfAPI library*/
#define PAPI_SET_OVRFLO  4    /*Option to turn on the overflow reporting software*/
#define PAPI_GET_OVRFLO  5    /*Option to query the status of the overflow
				reporting software*/

#define PAPI_ONESHOT	 1    /*Option to have the overflow handler called once*/
#define PAPI_RANDOMIZE	 2    /*Option to have the threshold of the overflow
				handler randomized*/

#define PAPI_MAX_EVNTS   16   /*The maximum number of spontaneous events 
				countable by the platform specific hardware 
				without multiplexing*/
#define PAPI_MAX_PRESET_EVENTS 64 /*The maximum number of preset events
				defined*/

#define PAPI_INIT_SLOTS  64     /*Number of initialized slots in
                                DynamicArray of EventSets */

#define PAPI_ERROR	 123  /*Exit code for PerfAPI executables that have 
				PAPI_VERB_ESTOP option set*/


#define PAPI_MAX_ERRMS   16   /*Number of internal error messages.*/

#define PAPI_MAX_ERROR   10   /*Highest number of defined error messages.*/ 	

/*
The Low Level API

The following functions represent the low level portion of the PerfAPI. These
functions provide greatly increased efficiency and functionality over the high
level API presented in the next section. All of the following functions are callable
from both C and Fortran except where noted. As mentioned in the introduction,
the low level API is only as powerful as the substrate upon which it is built. Thus
some features may not be available on every platform. The converse may also be
true, that more advanced features may be available and defined in the header file.
The user is encouraged to read the documentation carefully. 
*/


int PAPI_set_granularity(int granularity);
int PAPI_set_context(int context);
int PAPI_perror(int code, char *destination, int length);

int PAPI_add_event(int *EventSet, int Event);
int PAPI_add_events(int *EventSet, int *Events, int number);
int PAPI_add_pevent(int *EventSet, int code, void *inout);
int PAPI_rem_event(int EventSet, int Event); 
int PAPI_rem_events(int EventSet, int *Events, int number); 
int PAPI_list_events(int EventSet, int *Events, int *number);

int PAPI_start(int EventSet);
int PAPI_stop(int EventSet, long long *values);
int PAPI_read(int EventSet, long long *values);
int PAPI_accum(int EventSet, long long *values);
int PAPI_write(int EventSet, long long *values);

int PAPI_reset(int EventSet);
int PAPI_cleanup(int *EventSet);
void PAPI_shutdown(void);

int PAPI_state(int EventSet, int *status);
int PAPI_set_opt(int option, int value, PAPI_option_t *ptr); 

void (*handler)(int signal, void *siginfo, void *ucontext, 
              int EventSet, int Event, int count); 
int PAPI_get_opt(int option, int *value, PAPI_option_t *ptr);

int PAPI_num_events(void);

/*
The High Level API

The simple interface implemented by the following four routines
allows the user to access and count specific hardware events from
both C and Fortran. It should be noted that this API can be used in
conjunction with the low level API. If counter multiplexing is
enabled by the user, the high level API is only able to access those
events countable simultaneously by the underlying hardware. 

*/

int PAPI_start_counters(int *events, int array_len);
int PAPI_read_counters(long long *values, int array_len);
int PAPI_stop_counters(long long *values, int array_len);
