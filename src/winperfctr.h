/* $Id$
 * Performance-Monitoring Counters driver
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include "win_extras.h"

#ifdef DEBUG
  #define DBG(a) { extern int papi_debug; if (papi_debug) { fprintf(stderr,"DEBUG:%s:%d: ",__FILE__,__LINE__); fprintf a; } }
#else
  #define DBG(a)
#endif

struct perfctr_sum_ctrs {
	u_long_long ctr[5];	/* tsc, pmc0, ..., pmc3 */
};

struct perfctr_low_ctrs {
	unsigned int ctr[5];		/* tsc, pmc0, ..., pmc3 */
};

struct perfctr_control {
	unsigned int evntsel[4];
};

struct vperfctr;	/* opaque */	// lifted from <perfctr.h>

/* returned by the PERFCTR_INFO ioctl */
struct perfctr_info {
	unsigned char version_major;
	unsigned char version_minor;
	unsigned char version_micro;
	unsigned char nrcpus;
	unsigned char cpu_type;
	unsigned char cpu_features;
	unsigned long cpu_khz;
};

#define PERFCTR_X86_GENERIC	0	/* any x86 with rdtsc */
#define PERFCTR_X86_INTEL_P5	1	/* no rdpmc */
#define PERFCTR_X86_INTEL_P5MMX	2
#define PERFCTR_X86_INTEL_P6	3
#define PERFCTR_X86_INTEL_PII	4
#define PERFCTR_X86_INTEL_PIII	5
#define PERFCTR_X86_CYRIX_MII	6
#define PERFCTR_X86_WINCHIP_C6	7	/* no rdtsc */
#define PERFCTR_X86_WINCHIP_2	8	/* no rdtsc */
#define PERFCTR_X86_AMD_K7	9

#define PERFCTR_FEATURE_RDPMC	0x01
#define PERFCTR_FEATURE_RDTSC	0x02

#define _PERFCTR_IOCTL	0xD0	/* 'P'+128, currently unassigned */

#define PERFCTR_INFO		(_PERFCTR_IOCTL + 0)
#define VPERFCTR_SAMPLE		(_PERFCTR_IOCTL + 2)
#define VPERFCTR_UNLINK		(_PERFCTR_IOCTL + 3)
#define VPERFCTR_CONTROL	(_PERFCTR_IOCTL + 4)
#define VPERFCTR_STOP		(_PERFCTR_IOCTL + 5)

/* user-visible state of an mmap:ed virtual perfctr */
struct vperfctr_state {
	int status;	/* DEAD, STOPPED, or # of active counters */
	unsigned int control_id;	/* identify inheritance trees */
	struct perfctr_control control;
	struct perfctr_sum_ctrs sum;
	struct perfctr_low_ctrs start;
	struct perfctr_sum_ctrs children;
};
#define VPERFCTR_STATUS_DEAD	-1
#define VPERFCTR_STATUS_STOPPED	0

/* parameter in GPERFCTR_CONTROL ioctl */
struct gperfctr_control {
	unsigned long interval_usec;
	unsigned int nrcpus;
	struct perfctr_control cpu_control[1];	/* actually 'nrcpus' */
};

/* returned by GPERFCTR_READ ioctl */
struct gperfctr_cpu_state {
	int nrctrs;
	struct perfctr_control control;
	struct perfctr_sum_ctrs sum;
};
struct gperfctr_state {
	unsigned nrcpus;
	struct gperfctr_cpu_state cpu_state[1];	/* actually 'nrcpus' */
};


/*
 * Raw device interface.
 */

struct perfctr_dev;	/* opaque */
struct perfctr_dev *perfctr_dev_open(void);
void perfctr_dev_close(struct perfctr_dev*);
int perfctr_syscall(const struct perfctr_dev *dev, unsigned cmd, long arg);

/* Create an access point to your own vperfctr. 'dev' can be closed
   afterwards: the vperfctr remains valid until closed. */
struct vperfctr *vperfctr_attach(const struct perfctr_dev *dev);

u_long_long vperfctr_read_one(const struct vperfctr*, int);
int vperfctr_read_state(const struct vperfctr*, struct vperfctr_state*);
int vperfctr_control(const struct vperfctr*, struct perfctr_control*);
int vperfctr_stop(const struct vperfctr*);
int vperfctr_unlink(const struct vperfctr*);
void vperfctr_close(struct vperfctr*);

/* Experimental /proc/self/perfctr interface. */
struct vperfctr *vperfctr_open(void);
int vperfctr_info(const struct vperfctr*, struct perfctr_info*);

/*
 * Miscellaneous operations.
 */

int perfctr_info(const struct perfctr_dev*, struct perfctr_info*);
unsigned perfctr_cpu_nrctrs(const struct perfctr_info*);
const char *perfctr_cpu_name(const struct perfctr_info*);
unsigned perfctr_evntsel_num_insns(const struct perfctr_info*);
