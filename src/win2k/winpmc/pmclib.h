/* $Id$
 * Performance-Monitoring Counters driver
 *
 */

#include "WinPMC.h"

#ifdef DEBUG
  #define DBG(a) { extern int papi_debug; if (papi_debug) { fprintf(stderr,"DEBUG:%s:%d: ",__FILE__,__LINE__); fprintf a; } }
#else
  #define DBG(a)
#endif


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
int pmc_control(HANDLE kd, struct pmc_control *);
HANDLE pmc_open(void);
void pmc_close(HANDLE kd);
