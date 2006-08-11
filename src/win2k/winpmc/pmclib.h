/* $Id$
 * Performance-Monitoring Counters driver
 *
 */

#ifndef _PMCLIB_H
#define _PMCLIB_H 

#include "WinPMC.h"

/****WIN32: Supporting var args in Windows will take some research
    For now we'll just suppress the printing of debug statements in the 
    kernel driver. Also, this may need to be folded into papi_internal.
*/
//#ifdef DEBUG
//  #define PMCDBG(a) { extern int _papi_hwi_debug; if (papi_hwi_debug) { fprintf(stderr,"DEBUG:%s:%d: ",__FILE__,__LINE__); fprintf a; } }
//#else
  #define PMCDBG(a)

/* user-visible state of an mmap:ed virtual perfctr */
struct pmc_state {
	int status;	/* DEAD, STOPPED, or # of active counters */
	unsigned int control_id;	/* identify inheritance trees */
	struct pmc_control control;
	struct pmc_sum_ctrs sum;
	struct pmc_low_ctrs start;
	struct pmc_sum_ctrs children;
};
#define PMC_STATUS_DEAD	-1
#define PMC_STATUS_STOPPED	0


/*
 * Raw device interface.
 */

int pmc_read_state(int nrctrs, struct pmc_state *);
int pmc_init(HANDLE kd);
int pmc_set_control(HANDLE kd, struct pmc_control *);
HANDLE pmc_open(void);
void pmc_close(HANDLE kd);
char *pmc_revision(void);
char *pmc_kernel_version(HANDLE kd);

#endif // _PMCLIB_H 
