/* $Id$
 * Library interface to Linux x86 Performance-Monitoring Counters.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
//#include <unistd.h>
//#include <sys/mman.h>
//#include <sys/ioctl.h>
//#include <fcntl.h>
#include "winperfctr.h"

//#define PAGE_SIZE	4096

#define NO_DRIVER

#define WINPERFCTR_DEVICE "\\\\.\\WINPERFCTR"

struct vperfctr {
    int fd;
    volatile const struct vperfctr_state *kstate;
    unsigned char have_rdpmc;
    unsigned char have_rdtsc;
};


/*
 * Operations on the process' own virtual-mode perfctrs.
 */

struct vperfctr *vperfctr_open(void)
{
    struct vperfctr *perfctr;
    struct perfctr_info info;

    perfctr = malloc(sizeof(*perfctr));
    if( !perfctr )
	return NULL;
//    perfctr->fd = open("/proc/self/perfctr", O_RDONLY);
#ifdef NO_DRIVER
	perfctr->fd = 1;
#else
	perfctr->fd = (int)CreateFile(
		WINPERFCTR_DEVICE,					// pointer to name of the file 
		GENERIC_READ + GENERIC_WRITE,		// access (read-write) mode 
		FILE_SHARE_READ	+ FILE_SHARE_WRITE,	// share mode 
		NULL,								// pointer to security attributes 
		OPEN_EXISTING,						// how to create 
		0,									// file attributes 
		NULL );								// handle to file with attributes to copy  
#endif
    if( perfctr->fd < 0 ) {
		DBG((stderr, "/proc/self/perfctr\n"));	/* XXX: temporary */
		goto out_perfctr;
    }
//    if( ioctl(perfctr->fd, PERFCTR_INFO, &info) != 0 ) {
    if(vperfctr_info(perfctr, &info) != 0) {
		DBG((stderr, "/proc/self/perfctr/info\n"));	/* XXX: temporary */
		goto out_fd;
    }
    perfctr->have_rdpmc = (info.cpu_features & PERFCTR_FEATURE_RDPMC) != 0;
    perfctr->have_rdtsc = (info.cpu_features & PERFCTR_FEATURE_RDTSC) != 0;
    perfctr->kstate = 0;
//    perfctr->kstate = mmap(NULL, PAGE_SIZE, PROT_READ,
//			   MAP_SHARED, perfctr->fd, 0);
//    if( perfctr->kstate != MAP_FAILED )
	return perfctr;
 out_fd:
    /* XXX: unlink if control_id == 0 ? */
//    close(perfctr->fd);
	CloseHandle((HANDLE)perfctr->fd);
 out_perfctr:
    free(perfctr);
    return NULL;
}

int vperfctr_info(const struct vperfctr *vperfctr, struct perfctr_info *info)
{
//    return ioctl(vperfctr->fd, PERFCTR_INFO, info);
	int	retval;

#ifdef NO_DRIVER
	#define VERSION_MAJOR	1
	#define VERSION_MINOR	6
	#define VERSION_MICRO	0

	// fake initialization of info structure
	info->version_major = VERSION_MAJOR;
	info->version_minor = VERSION_MINOR;
	info->version_micro = VERSION_MICRO;
	info->nrcpus = 1;
	info->cpu_type = PERFCTR_X86_INTEL_PII;
	info->cpu_features = PERFCTR_FEATURE_RDPMC + PERFCTR_FEATURE_RDTSC;
	info->cpu_khz = 450000;
	retval = 0;
#else

	int	BytesReturned;
	
	retval = !(int)DeviceIoControl(
		(HANDLE) vperfctr->fd,	// handle to device of interest
		PERFCTR_INFO,			// control code of operation to perform
		NULL,					// pointer to buffer to supply input data
		0,						// size of input buffer
		(LPVOID) info,			// pointer to buffer to receive output data
		sizeof(struct perfctr_info),	// size of output buffer
		&BytesReturned,			// pointer to variable to receive output byte count
		NULL); 	// pointer to overlapped structure for asynchronous operation
#endif
	return (retval);
}


static __inline unsigned long get_rdtscl (void)
{
	volatile unsigned long val;

	__asm rdtsc				// Read Time Stamp Counter
	__asm mov val, eax		// move low part to val
    return (val);			// send 32 bits back
}

static __inline unsigned long get_rdpmcl (ctr)
{
	volatile unsigned long val;

	__asm mov ecx, ctr		// move high part to union
	__asm rdpmc				// Read Performance Monitor Counter
	__asm mov val, eax		// move low part to union
    return (val);			// send 64 bits back
}

int vperfctr_read_state(const struct vperfctr *self, struct vperfctr_state *state)
{
    unsigned int prev_tsc, next_tsc;
    struct perfctr_low_ctrs now;
    int i, nrctrs;

    /* If the counters are stopped, or if at least one PMC is enabled
     * but the CPU doesn't support RDPMC, then tell the kernel to sync
     * its state, and copy it from the mmap() buffer.
     */
    nrctrs = self->kstate->status;
    if( (nrctrs > 1 && !self->have_rdpmc) || nrctrs <= 0 ) {
	/* XXX: change SAMPLE to also update a given user-space buffer? */
//	ioctl(self->fd, VPERFCTR_SAMPLE, NULL);
	DeviceIoControl((HANDLE) self->fd, VPERFCTR_SAMPLE,
		NULL, 0, NULL, 0, NULL, NULL);
	do {
	    *state = *self->kstate;
	} while( state->start.ctr[0] != self->kstate->start.ctr[0] );
	return 0;
    }
    /* initial context-switch timestamp */
    next_tsc = self->kstate->start.ctr[0];
    do {
	prev_tsc = next_tsc;
	/* read register contents */
	if( self->have_rdtsc )	/* WinChip lossage */
		now.ctr[0] = get_rdtscl();
//	    rdtscl(now.ctr[0]);
	for(i = nrctrs - 1; --i >= 0;)
	    now.ctr[i+1] = get_rdpmcl(i); /* TSC is ctr[0], PMC i is ctr[i+1] */
//	    rdpmcl(i, now.ctr[i+1]); /* TSC is ctr[0], PMC i is ctr[i+1] */
	/* copy kernel state */
	*state = *self->kstate;
	/* next context-switch timestamp */
	next_tsc = self->kstate->start.ctr[0];
	/* loop until the copy is consistent (the timestamps match) */
    } while( prev_tsc != next_tsc );
    /* we have a good copy: update the sums */
    if( !self->have_rdtsc )	/* WinChip lossage */
	now.ctr[0] = 0;
    for(i = 0; i < nrctrs; ++i)
	state->sum.ctr[i] += now.ctr[i] - state->start.ctr[i];
    return 0;
}

int vperfctr_control(const struct vperfctr *perfctr,
		     struct perfctr_control *control)
{
//    return ioctl(perfctr->fd, VPERFCTR_CONTROL, control);
	int	retval;

#ifdef NO_DRIVER
	// fake initialization of info structure
	retval = 0;
#else
	retval = !(int)DeviceIoControl(
		(HANDLE) perfctr->fd,	// handle to device of interest
		VPERFCTR_CONTROL,		// control code of operation to perform
		(LPVOID) control,		// pointer to buffer to supply input data
		sizeof(struct perfctr_control),	// size of input buffer
		NULL,					// pointer to buffer to receive output data
		0,						// size of output buffer
		NULL,					// pointer to variable to receive output byte count
		NULL); 	// pointer to overlapped structure for asynchronous operation
#endif
	return (retval);
}

int vperfctr_stop(const struct vperfctr *perfctr)
{
//    return ioctl(perfctr->fd, VPERFCTR_STOP, NULL);
	int	retval;

#ifdef NO_DRIVER
	retval = 0;
#else
	retval = !(int)DeviceIoControl(
		(HANDLE) perfctr->fd,	// handle to device of interest
		VPERFCTR_STOP,			// control code of operation to perform
		NULL,					// pointer to buffer to supply input data
		0,						// size of input buffer
		NULL,					// pointer to buffer to receive output data
		0,						// size of output buffer
		NULL,					// pointer to variable to receive output byte count
		NULL); 	// pointer to overlapped structure for asynchronous operation
#endif
	return (retval);
}

int vperfctr_unlink(const struct vperfctr *perfctr)
{
//    return ioctl(perfctr->fd, VPERFCTR_UNLINK, NULL);
	int	retval;

#ifdef NO_DRIVER
	retval = 0;
#else
	retval = !(int)DeviceIoControl(
		(HANDLE) perfctr->fd,	// handle to device of interest
		VPERFCTR_UNLINK,		// control code of operation to perform
		NULL,					// pointer to buffer to supply input data
		0,						// size of input buffer
		NULL,					// pointer to buffer to receive output data
		0,						// size of output buffer
		NULL,					// pointer to variable to receive output byte count
		NULL); 	// pointer to overlapped structure for asynchronous operation
#endif
	return (retval);
}

void vperfctr_close(struct vperfctr *perfctr)
{
    if( perfctr->fd >= 0 ) {
		CloseHandle((HANDLE)perfctr->fd);
//	munmap((void*)perfctr->kstate, PAGE_SIZE);
//	close(perfctr->fd);
    }
    free(perfctr);
}

unsigned perfctr_cpu_nrctrs(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_X86_GENERIC:
	return 1;
      case PERFCTR_X86_AMD_K7:
	return 5;
      default:
	return 3;
    }
}

const char *perfctr_cpu_name(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_X86_GENERIC:
	return "Generic x86 with TSC";
      case PERFCTR_X86_INTEL_P5:
        return "Intel Pentium";
      case PERFCTR_X86_INTEL_P5MMX:
        return "Intel Pentium MMX";
      case PERFCTR_X86_INTEL_P6:
        return "Intel Pentium Pro";
      case PERFCTR_X86_INTEL_PII:
        return "Intel Pentium II";
      case PERFCTR_X86_INTEL_PIII:
        return "Intel Pentium III";
      case PERFCTR_X86_CYRIX_MII:
        return "Cyrix 6x86MX/MII/III";
      case PERFCTR_X86_WINCHIP_C6:
	return "WinChip C6";
      case PERFCTR_X86_WINCHIP_2:
	return "WinChip 2/3";
      case PERFCTR_X86_AMD_K7:
	return "AMD K7 Athlon";
      default:
        return "?";
    }
}
