/* $Id$
 * Library interface to virtual per-process performance counters.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include "libperfctr.h"
#include "marshal.h"
#include "arch.h"

#define ARRAY_SIZE(x)	(sizeof(x) / sizeof((x)[0]))

/*
 * Code to open (with or without creation) per-process perfctrs,
 * for both the current open("/proc/pid/perfctr", mode) and the
 * future ioctl(dev_perfctr_fd, VPERFCTR_{CREAT,OPEN}, pid) APIs.
 */

static int _vperfctr_open_pid_old(int pid, int try_creat, int try_rdonly, int *isnew)
{
    const char *filename;
    char namebuf[64];
    int fd;

    if( !pid )
	filename = "/proc/self/perfctr";
    else {
	snprintf(namebuf, sizeof namebuf, "/proc/%d/perfctr", pid);
	filename = namebuf;
    }
    *isnew = 1;
    fd = -1;
    if( try_creat )
	fd = open(filename, O_RDONLY|O_CREAT);
    if( fd < 0 && (try_creat ? errno == EEXIST : 1) && try_rdonly ) {
	*isnew = 0;
	fd = open(filename, O_RDONLY);
    }
    return fd;
}

static int _vperfctr_open_pid_new(int pid, int try_creat, int try_rdonly, int *isnew)
{
    int dev_perfctr_fd, fd;

    dev_perfctr_fd = open("/dev/perfctr", O_RDONLY);
    if( dev_perfctr_fd < 0 )
	return -1;
    *isnew = 1;
    fd = -1;
    if( try_creat )
	fd = ioctl(dev_perfctr_fd, VPERFCTR_CREAT, pid);
    if( fd < 0 && (try_creat ? errno == EEXIST : 1) && try_rdonly ) {
	*isnew = 0;
	fd = ioctl(dev_perfctr_fd, VPERFCTR_OPEN, pid);
    }
    close(dev_perfctr_fd);
    return fd;
}

/*
 * Operations using raw kernel handles, basically just open()/ioctl() wrappers.
 */

int _vperfctr_open(int creat)
{
    int dummy, fd;

    fd = _vperfctr_open_pid_new(0, creat, !creat, &dummy);
    if( fd < 0 )
	fd = _vperfctr_open_pid_old(0, creat, !creat, &dummy);
    return fd;
}

int _vperfctr_control(int fd, const struct vperfctr_control *control)
{
    return perfctr_ioctl_w(fd, VPERFCTR_CONTROL, control, &vperfctr_control_sdesc);
}

int _vperfctr_read_control(int fd, struct vperfctr_control *control)
{
    return perfctr_ioctl_r(fd, VPERFCTR_READ_CONTROL, control, &vperfctr_control_sdesc);
}

int _vperfctr_read_sum(int fd, struct perfctr_sum_ctrs *sum)
{
    return perfctr_ioctl_r(fd, VPERFCTR_READ_SUM, sum, &perfctr_sum_ctrs_sdesc);
}

/*
 * Operations using library objects.
 */

struct vperfctr {
    /* XXX: point to &vperfctr_state.cpu_state instead? */
    volatile const struct vperfctr_state *kstate;
    int fd;
    unsigned char have_rdpmc;
};

static int vperfctr_open_pid(int pid, struct vperfctr *perfctr)
{
    int fd, isnew;
    struct perfctr_info info;

    fd = _vperfctr_open_pid_new(pid, 1, 1, &isnew);
    if( fd < 0 ) {
	fd = _vperfctr_open_pid_old(pid, 1, 1, &isnew);
	if( fd < 0 )
	    goto out_perfctr;
    }
    perfctr->fd = fd;
    if( perfctr_abi_check_fd(perfctr->fd) < 0 )
	goto out_fd;
    if( perfctr_info(perfctr->fd, &info) < 0 )
	goto out_fd;
    perfctr->have_rdpmc = (info.cpu_features & PERFCTR_FEATURE_RDPMC) != 0;
    perfctr->kstate = mmap(NULL, PAGE_SIZE, PROT_READ,
			   MAP_SHARED, perfctr->fd, 0);
    if( perfctr->kstate != MAP_FAILED )
	return 0;
    munmap((void*)perfctr->kstate, PAGE_SIZE);
 out_fd:
    if( isnew )
	vperfctr_unlink(perfctr);
    close(perfctr->fd);
 out_perfctr:
    return -1;
}

struct vperfctr *vperfctr_open(void)
{
    struct vperfctr *perfctr;

    perfctr = malloc(sizeof(*perfctr));
    if( perfctr ) {
	if( vperfctr_open_pid(0, perfctr) == 0 )
	    return perfctr;
	free(perfctr);
    }
    return NULL;
}

int vperfctr_info(const struct vperfctr *vperfctr, struct perfctr_info *info)
{
    return perfctr_info(vperfctr->fd, info);
}

struct perfctr_cpus_info *vperfctr_cpus_info(const struct vperfctr *vperfctr)
{
    return perfctr_cpus_info(vperfctr->fd);
}

#if (__GNUC__ < 2) ||  (__GNUC__ == 2 && __GNUC_MINOR__ < 96)
#define __builtin_expect(x, expected_value) (x)
#endif

#define likely(x)	__builtin_expect((x),1)
#define unlikely(x)	__builtin_expect((x),0)

unsigned long long vperfctr_read_tsc(const struct vperfctr *self)
{
    unsigned long long sum;
    unsigned int tsc0, tsc1, now;
    volatile const struct vperfctr_state *kstate;

    kstate = self->kstate;
    if( likely(kstate->cpu_state.cstatus != 0) ) {
	tsc0 = kstate->cpu_state.tsc_start;
    retry:
	rdtscl(now);
	sum = kstate->cpu_state.tsc_sum;
	tsc1 = kstate->cpu_state.tsc_start;
	if( likely(tsc1 == tsc0) )
	    return sum += (now - tsc0);
	tsc0 = tsc1;
	goto retry; /* better gcc code than with a do{}while() loop */
    }
    return kstate->cpu_state.tsc_sum;
}

unsigned long long vperfctr_read_pmc(const struct vperfctr *self, unsigned i)
{
    unsigned long long sum;
    unsigned int start, now;
    unsigned int tsc0, tsc1;
    volatile const struct vperfctr_state *kstate;
    unsigned int cstatus;
    struct perfctr_sum_ctrs sum_ctrs;

    kstate = self->kstate;
    cstatus = kstate->cpu_state.cstatus;
    /* gcc 3.0 generates crap code for likely(E1 && E2) :-( */
    if( perfctr_cstatus_has_tsc(cstatus) && vperfctr_has_rdpmc(self) ) {
	 tsc0 = kstate->cpu_state.tsc_start;
    retry:
	 rdpmcl(kstate->cpu_state.pmc[i].map, now);
	 start = kstate->cpu_state.pmc[i].start;
	 sum = kstate->cpu_state.pmc[i].sum;
	 tsc1 = kstate->cpu_state.tsc_start;
	 if( likely(tsc1 == tsc0) ) {
	      return sum += (now - start);
	 }
	 tsc0 = tsc1;
	 goto retry;
    }
    if( _vperfctr_read_sum(self->fd, &sum_ctrs) < 0 )
	perror(__FUNCTION__);
    return sum_ctrs.pmc[i];
}

static int vperfctr_read_ctrs_slow(const struct vperfctr *vperfctr,
				   struct perfctr_sum_ctrs *sum)
{
    return _vperfctr_read_sum(vperfctr->fd, sum);
}

int vperfctr_read_ctrs(const struct vperfctr *self,
		       struct perfctr_sum_ctrs *sum)
{
    unsigned int tsc0, now;
    unsigned int cstatus, nrctrs;
    volatile const struct vperfctr_state *kstate;
    int i;

    /* Fast path is impossible if the TSC isn't being sampled (bad idea,
       but on WinChip you don't have a choice), or at least one PMC is
       enabled but the CPU doesn't have RDPMC. */
    kstate = self->kstate;
    cstatus = kstate->cpu_state.cstatus;
    nrctrs = perfctr_cstatus_nrctrs(cstatus);
    if( perfctr_cstatus_has_tsc(cstatus) && (!nrctrs || vperfctr_has_rdpmc(self)) ) {
    retry:
	tsc0 = kstate->cpu_state.tsc_start;
	rdtscl(now);
	sum->tsc = kstate->cpu_state.tsc_sum + (now - tsc0);
	for(i = nrctrs; --i >= 0;) {
	    rdpmcl(kstate->cpu_state.pmc[i].map, now);
	    sum->pmc[i] = kstate->cpu_state.pmc[i].sum + (now - kstate->cpu_state.pmc[i].start);
	}
	if( likely(tsc0 == kstate->cpu_state.tsc_start) )
	    return 0;
	goto retry;
    }
    return vperfctr_read_ctrs_slow(self, sum);
}

int vperfctr_read_state(const struct vperfctr *self, struct perfctr_sum_ctrs *sum,
			struct vperfctr_control *control)
{
    if( _vperfctr_read_sum(self->fd, sum) < 0 )
	return -1;
    /* For historical reasons, control may be NULL. */
    if( control && _vperfctr_read_control(self->fd, control) < 0 )
	return -1;
    return 0;
}

int vperfctr_control(const struct vperfctr *perfctr,
		     struct vperfctr_control *control)
{
    return _vperfctr_control(perfctr->fd, control);
}

int vperfctr_stop(const struct vperfctr *perfctr)
{
    struct vperfctr_control control;
    memset(&control, 0, sizeof control);
    return _vperfctr_control(perfctr->fd, &control);
}

int vperfctr_is_running(const struct vperfctr *perfctr)
{
    return perfctr->kstate->cpu_state.cstatus != 0;
}

int vperfctr_iresume(const struct vperfctr *perfctr)
{
    return ioctl(perfctr->fd, VPERFCTR_IRESUME, NULL);
}

int vperfctr_unlink(const struct vperfctr *perfctr)
{
    return ioctl(perfctr->fd, VPERFCTR_UNLINK, NULL);
}

void vperfctr_close(struct vperfctr *perfctr)
{
    munmap((void*)perfctr->kstate, PAGE_SIZE);
    close(perfctr->fd);
    free(perfctr);
}

/*
 * Operations on other processes' virtual-mode perfctrs.
 */

struct rvperfctr {
    struct vperfctr vperfctr;	/* must be first for the close() operation */
    int pid;
};

struct rvperfctr *rvperfctr_open(int pid)
{
    struct rvperfctr *rvperfctr;

    rvperfctr = malloc(sizeof(*rvperfctr));
    if( rvperfctr ) {
	if( vperfctr_open_pid(pid, &rvperfctr->vperfctr) == 0 ) {
	    rvperfctr->pid = pid;
	    return rvperfctr;
	}
	free(rvperfctr);
    }
    return NULL;
}

int rvperfctr_pid(const struct rvperfctr *rvperfctr)
{
    return rvperfctr->pid;
}

int rvperfctr_info(const struct rvperfctr *rvperfctr, struct perfctr_info *info)
{
    return vperfctr_info(&rvperfctr->vperfctr, info);
}

int rvperfctr_read_ctrs(const struct rvperfctr *rvperfctr,
			struct perfctr_sum_ctrs *sum)
{
    return vperfctr_read_ctrs_slow(&rvperfctr->vperfctr, sum);
}

int rvperfctr_read_state(const struct rvperfctr *rvperfctr,
			 struct perfctr_sum_ctrs *sum,
			 struct vperfctr_control *control)
{
    return vperfctr_read_state(&rvperfctr->vperfctr, sum, control);
}

int rvperfctr_control(const struct rvperfctr *rvperfctr,
		      struct vperfctr_control *control)
{
    return vperfctr_control(&rvperfctr->vperfctr, control);
}

int rvperfctr_stop(const struct rvperfctr *rvperfctr)
{
    return vperfctr_stop(&rvperfctr->vperfctr);
}

int rvperfctr_unlink(const struct rvperfctr *rvperfctr)
{
    return vperfctr_unlink(&rvperfctr->vperfctr);
}

void rvperfctr_close(struct rvperfctr *rvperfctr)
{
    /* this relies on offsetof(struct rvperfctr, vperfctr) == 0 */
    vperfctr_close(&rvperfctr->vperfctr);
}
