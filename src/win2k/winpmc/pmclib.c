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
	HANDLE	kd;
	DWORD lastError;

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
		DBG((stderr, "/proc/self/perfctr\n"));	/* XXX: temporary */
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
// If an x86 processor appears with more than 4 counters,
// this code will need to be modified...

	// Read the time stamp counter into ctr[0]
	// Best I can tell, this is *always* a full 64-bit value
	__asm rdtsc
	__asm mov now.ctr.LowPart[0 * TYPE ULARGE_INTEGER], eax	// TSC -> ctr[0]
	__asm mov now.ctr.HighPart[0 * TYPE ULARGE_INTEGER], edx	// TSC -> ctr[0]

	// the counters are 40-bit values for anything less than Pentium IV
	// we mask of the low 8 bits in the high word to exclude possible stray bits
	switch (nrctrs) {
		case 5:							// move 4 counters to now.cnt[]
			__asm mov ecx, 3
			__asm rdpmc
			__asm mov now.ctr.LowPart[4 * TYPE ULARGE_INTEGER], eax
			__asm and edx, 0x000000FF
			__asm mov now.ctr.HighPart[4 * TYPE ULARGE_INTEGER], edx
		case 4:
			__asm mov ecx, 2
			__asm rdpmc
			__asm mov now.ctr.LowPart[3 * TYPE ULARGE_INTEGER], eax
			__asm and edx, 0x000000FF
			__asm mov now.ctr.HighPart[3 * TYPE ULARGE_INTEGER], edx
		case 3:							// move 2 counters to now.cnt[]
			__asm mov ecx, 1
			__asm rdpmc
			__asm mov now.ctr.LowPart[2 * TYPE ULARGE_INTEGER], eax
			__asm and edx, 0x000000FF
			__asm mov now.ctr.HighPart[2 * TYPE ULARGE_INTEGER], edx
		case 2:
			__asm mov ecx, 0
			__asm rdpmc
			__asm mov now.ctr.LowPart[1 * TYPE ULARGE_INTEGER], eax
			__asm and edx, 0x000000FF
			__asm mov now.ctr.HighPart[1 * TYPE ULARGE_INTEGER], edx
			break;
	}
	// Copy the counter values to the current structure
	*(struct pmc_large_ctrs *)current = now;
}

//static	struct pmc_low_ctrs start;
static	struct pmc_sum_ctrs start;

int pmc_read_state(int nrctrs, struct pmc_state *state)
{
//	struct pmc_low_ctrs now;
	struct pmc_sum_ctrs now;
	int i;

	pmc_read_now(nrctrs, &now);

    /* update the sums */
	for(i = 0; i < nrctrs; ++i)
//		state->sum.ctr[i] += now.ctr[i] - state->start.ctr[i];
//		state->sum.ctr[i] += now.ctr[i] - start.ctr[i];
//		state->sum.ctr[i] = now.ctr[i] - start.ctr[i];
		state->sum.ctr[i] += now.ctr[i];
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

int pmc_control(HANDLE kd, struct pmc_control *control)
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
    if( kd > 0 ) CloseHandle(kd);
}

