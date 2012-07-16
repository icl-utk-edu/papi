#ifndef _PAPI_FreeBSD_CONTEXT_H
#define _PAPI_FreeBSD_CONTEXT_H

typedef struct hwd_ucontext {
	int placeholder; 
} hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx) (0x80000000)

#endif /* _PAPI_FreeBSD_CONTEXT_H */
