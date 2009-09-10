/* $Id$
 * x86 Performance-Monitoring Counters driver
 *
 */
#ifndef _PMC_KERNEL_H
#define _PMC_KERNEL_H

#ifndef PENTIUM4
#define nPMC 4
#else
#define nPMC 18
#endif

struct pmc_sum_ctrs {
	ULONGLONG tsc;			/* tsc */
	ULONGLONG pmc[nPMC];		/* pmc[0], ..., pmc[n] */
};

struct pmc_large_ctrs {
	ULARGE_INTEGER tsc;		/* tsc */
	ULARGE_INTEGER pmc[nPMC];	/* pmc[0], ..., pmc[n] */
};

struct pmc_low_ctrs {
	unsigned int tsc;		/* tsc */
	unsigned int pmc[nPMC];		/* pmc[0], ..., pmc[n] */
};

struct pmc_cpu_control {
	unsigned int tsc_on;
	unsigned int nractrs;		/* # of a-mode counters */
	unsigned int nrictrs;		/* # of i-mode counters */
	unsigned int pmc_map[nPMC];
	unsigned int evntsel[nPMC];	/* one per counter, even on P5 */
	struct {
		unsigned int pebs_enable;	/* for replay tagging */
		unsigned int pebs_matrix_vert;	/* for replay tagging */
		unsigned int escr[nPMC];	/* P4 ESCR contents */
	} p4;
	int ireset[nPMC];			/* <= 0, for i-mode counters */
};

struct vpmc_control {
	int si_signo;
	struct pmc_cpu_control cpu_control;
};

struct pmc_control {
	unsigned int evntsel[nPMC];
};

struct pmc_info {
	unsigned int family;
	unsigned int features;
	unsigned int stepping;
	unsigned int model;
	char vendor[12];
};

/* External entry points */
extern int kern_pmc_init(void);
extern int kern_pmc_info(struct pmc_info *info);
extern int kern_pmc_control(struct pmc_control *control);
extern void kern_pmc_exit(void);


// define a whole passel of status codes to aid in debugging
// see <ntstatus.h> for details
//  Values are 32 bit values layed out as follows:
//
//   3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
//   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
//  +---+-+-+-----------------------+-------------------------------+
//  |Sev|C|R|     Facility          |               Code            |
//  +---+-+-+-----------------------+-------------------------------+
//
//  where
//      Sev - is the severity code
//
//          00 - Success
//          01 - Informational
//          10 - Warning
//          11 - Error
//
//      C - is the Customer code flag
//      R - is a reserved bit
//      Facility - is the facility code
//      Code - is the facility's status code

#define STATUS_P5_RESERVED				((NTSTATUS)0xE0000001L)
#define STATUS_P6_RESERVED0				((NTSTATUS)0xE0000002L)
#define STATUS_P6_RESERVED1				((NTSTATUS)0xE0000003L)
#define STATUS_K7_RESERVED				((NTSTATUS)0xE0000004L)
#define STATUS_K7_DISABLED				((NTSTATUS)0xE000000EL)
#define STATUS_MII_RESERVED				((NTSTATUS)0xE0000005L)
#define STATUS_NO_INTEL_INIT			((NTSTATUS)0xE0000006L)
#define STATUS_NO_AMD_INIT				((NTSTATUS)0xE0000007L)
#define STATUS_NO_CYRIX_INIT			((NTSTATUS)0xE0000008L)
#define STATUS_UNKNOWN_CPU_INIT			((NTSTATUS)0xE0000009L)
#define STATUS_NO_MSR					((NTSTATUS)0xE000000AL)
#define STATUS_NO_TSC					((NTSTATUS)0xE000000BL)
#define STATUS_NO_MMX					((NTSTATUS)0xE000000CL)
#define STATUS_NO_CPUID					((NTSTATUS)0xE000000DL)

#define STATUS_DEBUG_ERROR_0			((NTSTATUS)0xE0000010L)
#define STATUS_DEBUG_ERROR_1			((NTSTATUS)0xE0000011L)
#define STATUS_DEBUG_ERROR_2			((NTSTATUS)0xE0000012L)
#define STATUS_DEBUG_ERROR_3			((NTSTATUS)0xE0000013L)

#define STATUS_DEBUG_WARNING_0			((NTSTATUS)0xA0000010L)
#define STATUS_DEBUG_WARNING_1			((NTSTATUS)0xA0000011L)
#define STATUS_DEBUG_WARNING_2			((NTSTATUS)0xA0000012L)
#define STATUS_DEBUG_WARNING_3			((NTSTATUS)0xA0000013L)

#define STATUS_DEBUG_INFO_0				((NTSTATUS)0x60000010L)
#define STATUS_DEBUG_INFO_1				((NTSTATUS)0x60000011L)
#define STATUS_DEBUG_INFO_2				((NTSTATUS)0x60000012L)
#define STATUS_DEBUG_INFO_3				((NTSTATUS)0x60000013L)


#endif // _PMC_KERNEL_H

