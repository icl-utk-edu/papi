/* PMCLib: Library interface to WIndows Performance-Monitoring Counter Driver.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <wtypes.h>
#include <winioctl.h>
#include "pmclib.h"


#define PMC_DEVICE "\\\\.\\WinPMC"


// NOTE: All functions in this file return positive for success and negative or zero for failure.

HANDLE pmc_open(void)
{
	HANDLE	process, kd;
	DWORD processAffinityMask, systemAffinityMask;
	DWORD lastError;

	/* force this process to run only on the lowest available processor */
	process = GetCurrentProcess();
	if (GetProcessAffinityMask(process, &processAffinityMask, &systemAffinityMask)) {
		/* set the mask to the lowest possible processor
			(I think this one should ALWAYS be here...) */
		processAffinityMask = 0x00000001;
		/* scan for the lowest bit in the system mask */
		while (!processAffinityMask & systemAffinityMask)
			processAffinityMask <<= 1;
		/* set affinity to lowest processor only */
		if (!SetProcessAffinityMask(process, processAffinityMask)) {
			lastError = GetLastError();
			PMCDBG((stderr, "Error setting affinity: %d \n", lastError));
			return NULL;
		}
	}
	else {
		lastError = GetLastError();
		PMCDBG((stderr, "Error getting affinity: %d \n", lastError));
		return NULL;
    }

	kd = CreateFile(
		PMC_DEVICE,							// pointer to name of the file 
        GENERIC_READ | GENERIC_WRITE, 
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        0,									// Default security
        OPEN_EXISTING,
        0,									// Don't Perform asynchronous I/O
        0);									// No template

    if(kd == INVALID_HANDLE_VALUE) {
		lastError = GetLastError();
		PMCDBG((stderr, "Error opening pmc: %d \n", lastError));
		return NULL;
    }
    return kd;
}


//static void pmc_read_now(int nrctrs, struct pmc_low_ctrs *current)
//{
//	struct pmc_low_ctrs now;

static void pmc_read_now(int nrctrs, struct pmc_sum_ctrs *current)
{
	struct pmc_large_ctrs now;
// This produces the tightest assembly for the cases of less than 5
// performance counters, which is currently universal.
// nrctrs MAY be the number of counters ENABLED, not the number that exist.
// If an x86 processor appears with more than 4 counters (like Pentium4!!!),
// this code will need to be modified...

	// Read the time stamp counter into tsc
	// Best I can tell, this is *always* a full 64-bit value
	__asm rdtsc
	__asm mov now.tsc.LowPart, eax
	__asm mov now.tsc.HighPart, edx

	// the counters are 40-bit values for anything less than Pentium 4
	// we mask off the low 8 bits in the high word to exclude possible stray bits
	switch (nrctrs) {
		case 5:							// move 4 counters to now.cnt[]
			__asm mov ecx, 3
			__asm rdpmc
			__asm mov now.pmc.LowPart[3 * TYPE ULARGE_INTEGER], eax
			__asm and edx, 0x000000FF
			__asm mov now.pmc.HighPart[3 * TYPE ULARGE_INTEGER], edx
		case 4:
			__asm mov ecx, 2
			__asm rdpmc
			__asm mov now.pmc.LowPart[2 * TYPE ULARGE_INTEGER], eax
			__asm and edx, 0x000000FF
			__asm mov now.pmc.HighPart[2 * TYPE ULARGE_INTEGER], edx
		case 3:							// move 2 counters to now.cnt[]
			__asm mov ecx, 1
			__asm rdpmc
			__asm mov now.pmc.LowPart[1 * TYPE ULARGE_INTEGER], eax
			__asm and edx, 0x000000FF
			__asm mov now.pmc.HighPart[1 * TYPE ULARGE_INTEGER], edx
		case 2:
			__asm mov ecx, 0
			__asm rdpmc
			__asm mov now.pmc.LowPart[0 * TYPE ULARGE_INTEGER], eax
			__asm and edx, 0x000000FF
			__asm mov now.pmc.HighPart[0 * TYPE ULARGE_INTEGER], edx
			break;
	}
	// Copy the counter values to the current structure
	*(struct pmc_large_ctrs *)current = now;
}

static	struct pmc_sum_ctrs start;

int pmc_read_state(int nrctrs, struct pmc_state *state)
{
    /*	This is drastically simplified from the PAPI 2 version
	in recognition that until context switch saves are supported
	we can eliminate the extra overhead of summing the counts
	on a read.
    */
    pmc_read_now(nrctrs+1, &(state->sum));
    return TRUE;
}

/*
int pmc_init(HANDLE kd)
{
	struct pmc_info info;
	DWORD dwBytesReturned;
	int retval;

	if (!cpu_id(info.vendor, &info.family, &info.features))
		return -1;

	retval = DeviceIoControl(
		kd,							// handle to device of interest
		IOCTL_PMC_INIT,				// control code of operation to perform
		(LPVOID)(&info),				// pointer to buffer to supply input data
		sizeof(struct pmc_info),		// size of input buffer
		(LPVOID)(&info),				// pointer to buffer to receive output data
		sizeof(struct pmc_info),		// size of output buffer
		&dwBytesReturned,			// pointer to variable to receive output byte count
		NULL); 	// pointer to overlapped structure for asynchronous operation
	if (!retval) {
		retval = GetLastError();
		if (retval > 0) retval = -retval;
	}
	return(retval);
}
*/

int pmc_set_control(HANDLE kd, struct pmc_control *control)
{
	DWORD dwBytesReturned;
	int retval;
	int nrctrs;


	retval = DeviceIoControl(
		kd,							// handle to device of interest
		IOCTL_PMC_CONTROL,			// control code of operation to perform
		(LPVOID)(control),			// pointer to buffer to supply input data
		sizeof(struct pmc_control),	// size of input buffer
		(LPVOID)(&nrctrs),			// pointer to buffer to receive output data
		sizeof(nrctrs),				// size of output buffer
		&dwBytesReturned,			// pointer to variable to receive output byte count
		NULL); 	// pointer to overlapped structure for asynchronous operation
	if (!retval) {
		retval = GetLastError();
		if (retval > 0) retval = -retval;
	} // else pmc_read_now(nrctrs, &start);
	return(retval);
}

void pmc_close(HANDLE kd)
{
	HANDLE	process;
	DWORD processAffinityMask, systemAffinityMask;
	DWORD lastError;

    if( kd > 0 ) CloseHandle(kd);

	/* restore this process to run on all available processors */
	process = GetCurrentProcess();
	if (GetProcessAffinityMask(process, &processAffinityMask, &systemAffinityMask)) {
		/* set affinity to all processors */
		if (!SetProcessAffinityMask(process, systemAffinityMask)) {
			lastError = GetLastError();
			PMCDBG((stderr, "Error restoring affinity: %d \n", lastError));
			return;
		}
	}
	else {
		lastError = GetLastError();
		PMCDBG((stderr, "Error getting affinity: %d \n", lastError));
		return;
    }
}

char *pmc_revision(void)
{
    return("$Revision$"); // assumes this file is under cvs control
}

char *pmc_kernel_version(HANDLE kd)
{
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;
    static char version[256]; 

    // Dispatch the PMC_VERSION_STRING IOCTL to our driver.
    bReturnCode = DeviceIoControl(kd, IOCTL_PMC_VERSION_STRING,
	NULL, 0, (int *)version, 255, &dwBytesReturned, NULL);
    version[dwBytesReturned-2] = 0; // make sure it's terminated
    return(version);
}


