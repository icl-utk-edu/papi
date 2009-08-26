/* $Id$
 * x86 Performance-Monitoring Counters driver
 *
 */
#ifndef _PMC_KERNEL_H
#define _PMC_KERNEL_H

#if !defined(__i386__)
#define __i386__
#endif

#include <perfmon/perfmon.h>

typedef unsigned uint;

uint (*translate_pmc)(uint reg);
uint (*translate_pmd)(uint reg);

void write_control(pfarg_pmc_t *r, uint count);
void read_write_data(pfarg_pmd_t *r, uint count, uint write);

struct CPUInfo {
	uint family;
	uint features;
	uint stepping;
	uint model;
	char vendor[12];
};

/* External entry points */
int kern_pmc_init();
void kern_pmc_exit();


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

