/* $Id$
 * Library interface to Linux x86 Performance-Monitoring Counters.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include "libperfctr.h"

#define PAGE_SIZE	4096

struct perfctr_dev {
    int fd;
    struct perfctr_info info;
};

struct vperfctr {
    int fd;
    volatile const struct vperfctr_state *kstate;
    unsigned char have_rdpmc;
    unsigned char have_rdtsc;
};

/*
 * Raw device interface.
 */

struct perfctr_dev *perfctr_dev_open(void)
{
    struct perfctr_dev *dev;

    dev = malloc(sizeof(struct perfctr_dev));
    if( !dev )
	return NULL;
    dev->fd = open("/dev/perfctr", O_RDONLY);
    if( dev->fd >= 0 ) {
	if( ioctl(dev->fd, PERFCTR_INFO, &dev->info) == 0 )
	    return dev;
	close(dev->fd);
    }
    free(dev);
    return NULL;
}

void perfctr_dev_close(struct perfctr_dev *dev)
{
    close(dev->fd);
    free(dev);
}

int perfctr_syscall(const struct perfctr_dev *dev, unsigned cmd, long arg)
{
    return ioctl(dev->fd, cmd, arg);
}

/*
 * Operations on the process' own virtual-mode perfctrs.
 */

struct vperfctr *vperfctr_attach(const struct perfctr_dev *dev)
{
    struct vperfctr *perfctr;

    perfctr = malloc(sizeof(struct vperfctr));
    if( !perfctr )
	return NULL;
    perfctr->have_rdpmc = (dev->info.cpu_features & PERFCTR_FEATURE_RDPMC) != 0;
    perfctr->have_rdtsc = (dev->info.cpu_features & PERFCTR_FEATURE_RDTSC) != 0;
    perfctr->fd = ioctl(dev->fd, VPERFCTR_ATTACH, 0);
    if( perfctr->fd >= 0 ) {
	 perfctr->kstate = mmap(NULL, PAGE_SIZE, PROT_READ,
				MAP_SHARED, perfctr->fd, 0);
	 if( perfctr->kstate != MAP_FAILED )
	      return perfctr;
	 /* XXX: This leaves the process with a new but stopped vperfctr,
	    which will cause minor scheduling overhead. On the other hand,
	    we don't know if the attach above created the vperfctr or not,
	    so it wouldn't be correct to do an unlink here either. */
	 close(perfctr->fd);
    }
    free(perfctr);
    return NULL;
}

#define rdtscl(low)	\
	__asm__ __volatile__("rdtsc" : "=a"(low) : : "edx")
#define rdpmcl(ctr,low)	\
	__asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")

unsigned long long vperfctr_read_one(const struct vperfctr *self, int i)
{
    unsigned long long sum;
    unsigned int start, now;
    unsigned int tsc0, tsc1;

    if( i > 0 && !self->have_rdpmc ) {	/* Intel pre-MMX P5 lossage */
	ioctl(self->fd, VPERFCTR_SAMPLE, NULL);
	return self->kstate->sum.ctr[i];
    }
    tsc1 = self->kstate->start.ctr[0];
    if( i == 0 ) {	/* Retrieve the virtualised TSC. */
	do {
	    tsc0 = tsc1;
	    rdtscl(now);
	    sum = self->kstate->sum.ctr[0];
	    tsc1 = self->kstate->start.ctr[0];
	} while( tsc0 != tsc1 );
	start = tsc1;
    } else {		/* Retrieve the virtualised PMC #(i-1). */
	do {
	    tsc0 = tsc1;
	    rdpmcl(i-1, now);
	    sum = self->kstate->sum.ctr[i];
	    start = self->kstate->start.ctr[i];
	    tsc1 = self->kstate->start.ctr[0];
	} while( tsc0 != tsc1 );
    }
    return sum + (now - start);
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
	ioctl(self->fd, VPERFCTR_SAMPLE, NULL);
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
	    rdtscl(now.ctr[0]);
	for(i = nrctrs - 1; --i >= 0;)
	    rdpmcl(i, now.ctr[i+1]); /* TSC is ctr[0], PMC i is ctr[i+1] */
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
    return ioctl(perfctr->fd, VPERFCTR_CONTROL, control);
}

int vperfctr_stop(const struct vperfctr *perfctr)
{
    return ioctl(perfctr->fd, VPERFCTR_STOP, NULL);
}

int vperfctr_unlink(const struct vperfctr *perfctr)
{
    return ioctl(perfctr->fd, VPERFCTR_UNLINK, NULL);
}

void vperfctr_close(struct vperfctr *perfctr)
{
    if( perfctr->fd >= 0 ) {
	munmap((void*)perfctr->kstate, PAGE_SIZE);
	close(perfctr->fd);
    }
    free(perfctr);
}

/*
 * Operations on global-mode perfctrs.
 */

int perfctr_global_control(const struct perfctr_dev *dev, struct gperfctr_control *arg)
{
    return ioctl(dev->fd, GPERFCTR_CONTROL, (long)arg);
}

int perfctr_global_read(const struct perfctr_dev *dev, struct gperfctr_state *arg)
{
    return ioctl(dev->fd, GPERFCTR_READ, (long)arg);
}

int perfctr_global_stop(const struct perfctr_dev *dev)
{
    return ioctl(dev->fd, GPERFCTR_STOP, 0);
}

/*
 * Miscellaneous operations.
 */

int perfctr_info(const struct perfctr_dev *dev, struct perfctr_info *info)
{
    *info = dev->info;
    return 0;
}

unsigned perfctr_cpu_nrctrs(const struct perfctr_dev *dev)
{
    switch( dev->info.cpu_type ) {
      case PERFCTR_X86_GENERIC:
	return 1;
      case PERFCTR_X86_AMD_K7:
	return 5;
      default:
	return 3;
    }
}

const char *perfctr_cpu_name(const struct perfctr_dev *dev)
{
    switch( dev->info.cpu_type ) {
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
	return "WinChip 2";
      case PERFCTR_X86_AMD_K7:
	return "AMD K7 Athlon";
      default:
        return "?";
    }
}

unsigned perfctr_evntsel_num_insns(const struct perfctr_dev *dev)
{
    /* This is terribly naive. Assumes only one event will be
     * selected, at CPL > 0.
     */
    switch( dev->info.cpu_type ) {
      case PERFCTR_X86_INTEL_P5:
      case PERFCTR_X86_INTEL_P5MMX:
      case PERFCTR_X86_CYRIX_MII:
        /* event 0x16, count at CPL 3 */
        return 0x16 | (2 << 6);
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
      case PERFCTR_X86_AMD_K7:
        /* event 0xC0, count at CPL > 0, ENable */
        return 0xC0 | (1 << 16) | (1 << 22);
      case PERFCTR_X86_WINCHIP_C6:
        return 0x02;
      case PERFCTR_X86_WINCHIP_2:
        return 0x16;
      default:
	return 0;
    }
}
