/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    vmware.h
 * @author  Matt Johnson
 *          mrj@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * VMware component
 *
 * @brief
 *	This is the VMware component for PAPI-V. It will allow the user access to
 *	the underlying hardware information available from a VMware virtual machine.
 */

#ifndef _PAPI_VMWARE_H
#define _PAPI_VMWARE_H

#define VMWARE_MAX_COUNTERS 256

#define VMWARE_CPU_LIMIT_MHZ            0
#define VMWARE_CPU_RESERVATION_MHZ      1
#define VMWARE_CPU_SHARES               2
#define VMWARE_CPU_STOLEN_MS            3
#define VMWARE_CPU_USED_MS              4
#define VMWARE_ELAPSED_MS               5

#define VMWARE_MEM_ACTIVE_MB            6
#define VMWARE_MEM_BALLOONED_MB         7
#define VMWARE_MEM_LIMIT_MB             8
#define VMWARE_MEM_MAPPED_MB            9
#define VMWARE_MEM_OVERHEAD_MB          10
#define VMWARE_MEM_RESERVATION_MB       11
#define VMWARE_MEM_SHARED_MB            12
#define VMWARE_MEM_SHARES               13
#define VMWARE_MEM_SWAPPED_MB           14
#define VMWARE_MEM_TARGET_SIZE_MB       15
#define VMWARE_MEM_USED_MB              16

#define VMWARE_HOST_CPU_MHZ             17

#define VMWARE_HOST_TSC					18	// These first 3 can be used for timing
#define VMWARE_ELAPSED_TIME             19  // --       "   "       --
#define VMWARE_ELAPSED_APPARENT         20  // --       "   "       --

/* typedef's */

/** Structure that stores private information for each event */
typedef struct VMWARE_register {
	unsigned int selector;
    /**< Signifies which counter slot is being used */
    /**< Indexed from 1 as 0 has a special meaning  */
} VMWARE_register_t;

/** This structure is used to build the table of events */
typedef struct VMWARE_native_event_entry {
	VMWARE_register_t resources;        /**< Per counter resources       */
	char name[PAPI_MAX_STR_LEN];        /**< Name of the counter         */
	char description[PAPI_HUGE_STR_LEN]; /**< Description of the counter  */
	int writable;                       /**< Whether counter is writable */
    /* any other counter parameters go here */
} VMWARE_native_event_entry_t;

/** This structure is used when doing register allocation it possibly is not necessary when there are no register constraints */
typedef struct VMWARE_reg_alloc {
	VMWARE_register_t ra_bits;
} VMWARE_reg_alloc_t;

/** Holds control flags, usually out-of band configuration of the hardware */
typedef struct VMWARE_control_state {
	long_long counter[VMWARE_MAX_COUNTERS];    /**< Copy of counts, used for caching */
	long_long lastupdate;                       /**< Last update time, used for caching */
} VMWARE_control_state_t;

/** Holds per-thread information */
typedef struct VMWARE_context {
	VMWARE_control_state_t state;
} VMWARE_context_t;

#ifdef __cplusplus
extern "C" {
#endif
    
    /* function prototypes */
    	/* There aren't any */
    
#ifdef __cplusplus
}  /* extern C */
#endif


#endif
