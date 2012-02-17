/* $Id: virtual.c,v 1.35 2005/06/06 21:07:58 mikpe Exp $
 * Library interface to virtual per-process performance counters.
 *
 * Copyright (C) 1999-2005  Mikael Pettersson
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "libperfctr.h"
#include "arch.h"

#define ARRAY_SIZE(x)	(sizeof(x) / sizeof((x)[0]))
#define STRUCT_ARRAY_SIZE(TYPE, MEMBER)	ARRAY_SIZE(((TYPE*)0)->MEMBER)

/*
 * Code to open (with or without creation) per-process perfctrs.
 */

static int _vperfctr_open_pid(int pid, int try_creat, int try_rdonly, int *isnew)
{
    int fd;

    *isnew = 1;
    fd = -1;
    if( try_creat )
	fd = _sys_vperfctr_open(-1, pid, 1);
    if( fd < 0 && (try_creat ? errno == EEXIST : 1) && try_rdonly ) {
	*isnew = 0;
	fd = _sys_vperfctr_open(-1, pid, 0);
    }
    return fd;
}

/*
 * Operations using raw kernel handles, basically just _sys_perfctr() wrappers.
 */

int _vperfctr_open(int creat)
{
    int dummy;

    return _vperfctr_open_pid(0, creat, !creat, &dummy);
}

int __vperfctr_control(int fd, unsigned int cpu_type, const struct vperfctr_control *control)
{
    return _sys_vperfctr_write_control(fd, cpu_type, control);
}

int _vperfctr_control(int fd, const struct vperfctr_control *control)
{
    struct perfctr_info info;
    memset(&info, 0, sizeof info);
    perfctr_info_cpu_init(&info);
    return __vperfctr_control(fd, info.cpu_type, control);
}

int __vperfctr_read_control(int fd, unsigned int cpu_type, struct vperfctr_control *control)
{
    return _sys_vperfctr_read_control(fd, cpu_type, control);
}

int _vperfctr_read_control(int fd, struct vperfctr_control *control)
{
    struct perfctr_info info;
    memset(&info, 0, sizeof info);
    perfctr_info_cpu_init(&info);
    return __vperfctr_read_control(fd, info.cpu_type, control);
}

int _vperfctr_read_sum(int fd, struct perfctr_sum_ctrs *sum)
{
    return _sys_vperfctr_read_sum(fd, sum);
}

int _vperfctr_read_children(int fd, struct perfctr_sum_ctrs *children)
{
    return _sys_vperfctr_read_children(fd, children);
}

/*
 * Operations using library objects.
 */

/* user's view of mmap:ed virtual perfctr */
struct vperfctr_state {
	struct perfctr_cpu_state_user cpu_state;
};

struct vperfctr {
    /* XXX: point to &vperfctr_state.cpu_state instead? */
    volatile const struct vperfctr_state *kstate;
    volatile const void *mapping;
    int mapping_size;
    int fd;
    unsigned int cpu_type;
    unsigned char have_rdpmc;
    /* Subset of the user's control data */
    unsigned int pmc_map[STRUCT_ARRAY_SIZE(struct perfctr_cpu_control, pmc_map)];
};

static int vperfctr_open_pid(int pid, struct vperfctr *perfctr)
{
    int fd, isnew;
    struct perfctr_info info;
    int offset;

    offset = _perfctr_get_state_user_offset();
    if (offset < 0)
	return -1;
    fd = _vperfctr_open_pid(pid, 1, 1, &isnew);
    if( fd < 0 ) {
	goto out_perfctr;
    }
    perfctr->fd = fd;
    if( perfctr_abi_check_fd(perfctr->fd) < 0 )
	goto out_fd;
    if( perfctr_info(perfctr->fd, &info) < 0 )
	goto out_fd;
    perfctr->cpu_type = info.cpu_type;
    perfctr->have_rdpmc = (info.cpu_features & PERFCTR_FEATURE_RDPMC) != 0;
    perfctr->mapping_size = getpagesize();
    perfctr->mapping = mmap(NULL, perfctr->mapping_size, PROT_READ,
			    MAP_SHARED, perfctr->fd, 0);
    if (perfctr->mapping != MAP_FAILED) {
	perfctr->kstate = (void*)((char*)perfctr->mapping + offset);
	return 0;
    }
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

#ifndef seq_read_barrier
/* mmap() based sampling is supported only for self-monitoring tasks,
 * so for most CPUs we should only need a compiler barrier.  This
 * ensures that the two reads of the sequence number will truly wrap
 * all the operations to make this sample. */
#define seq_read_barrier()	__asm__ __volatile__ ("" : : : "memory");
#endif
/* These are adaptations of read_seqcount_begin() and
 * read_seqcount_retry() from include/linux/seqlock.h.  They use an
 * explicit u32 instead of an opaque seqcount_t, since the type of
 * the lock is part of the kernel/user ABI. */
static inline __u32 read_perfseq_begin(const volatile __u32 *seq)
{
    __u32 ret = *seq;

    seq_read_barrier();
    return ret;
}

static inline int read_perfseq_retry(const volatile __u32 *seq, __u32 iv)
{
    seq_read_barrier();
    return (iv & 1) | ((*seq) ^ iv);
}

unsigned long long vperfctr_read_tsc(const struct vperfctr *self)
{
    unsigned long long sum;
    unsigned int start, now;
    volatile const struct vperfctr_state *kstate;
    __u32 seq;

    kstate = self->kstate;
    if (unlikely(kstate->cpu_state.cstatus == 0))
	return kstate->cpu_state.tsc_sum;

    do {
	seq = read_perfseq_begin(&kstate->cpu_state.sequence);
	rdtscl(now);
	sum = kstate->cpu_state.tsc_sum;
	start = kstate->cpu_state.tsc_start;
    } while (unlikely(read_perfseq_retry(&kstate->cpu_state.sequence, seq)));
    return sum + (now - start);
}

unsigned long long vperfctr_read_pmc(const struct vperfctr *self, unsigned i)
{
    unsigned long long sum;
    unsigned int start, now;
    volatile const struct vperfctr_state *kstate;
    unsigned int cstatus;
    __u32 seq;

    kstate = self->kstate;
    cstatus = kstate->cpu_state.cstatus;

    if (unlikely(!vperfctr_has_rdpmc(self))) {
	struct perfctr_sum_ctrs sum_ctrs;
	if (_vperfctr_read_sum(self->fd, &sum_ctrs) < 0)
	    perror(__FUNCTION__);
	return sum_ctrs.pmc[i];
    }
    do {
	seq = read_perfseq_begin(&kstate->cpu_state.sequence);
	rdpmcl(self->pmc_map[i], now);
	start = kstate->cpu_state.pmc[i].start;
	sum = kstate->cpu_state.pmc[i].sum;
    } while (unlikely(read_perfseq_retry(&kstate->cpu_state.sequence, seq)));
    return sum + (now - start);
}

static int vperfctr_read_ctrs_slow(const struct vperfctr *vperfctr,
				   struct perfctr_sum_ctrs *sum)
{
    return _vperfctr_read_sum(vperfctr->fd, sum);
}

int vperfctr_read_ctrs(const struct vperfctr *self,
		       struct perfctr_sum_ctrs *sum)
{
    unsigned int now;
    unsigned int cstatus, nrctrs;
    volatile const struct vperfctr_state *kstate;
    __u32 seq;
    int i;

    /* Fast path is impossible if at least east one PMC is
       enabled but the CPU doesn't have RDPMC. */
    kstate = self->kstate;
    cstatus = kstate->cpu_state.cstatus;
    nrctrs = perfctr_cstatus_nrctrs(cstatus);
    if (nrctrs && !vperfctr_has_rdpmc(self))
	return vperfctr_read_ctrs_slow(self, sum);

    do {
	seq = read_perfseq_begin(&kstate->cpu_state.sequence);
	for (i = nrctrs; --i >= 0;) {
	    rdpmcl(self->pmc_map[i], now);
	    sum->pmc[i] =
		kstate->cpu_state.pmc[i].sum +
		(now - (unsigned int)kstate->cpu_state.pmc[i].start);
	}
	rdtscl(now);
	sum->tsc =
	    kstate->cpu_state.tsc_sum +
	    (now - (unsigned int)kstate->cpu_state.tsc_start);
    } while (unlikely(read_perfseq_retry(&kstate->cpu_state.sequence, seq)));
    return 0;
}

int vperfctr_read_state(const struct vperfctr *self, struct perfctr_sum_ctrs *sum,
			struct vperfctr_control *control)
{
    if( _vperfctr_read_sum(self->fd, sum) < 0 )
	return -1;
    /* For historical reasons, control may be NULL. */
    if( control && __vperfctr_read_control(self->fd, self->cpu_type, control) < 0 )
	return -1;
    return 0;
}

int vperfctr_control(struct vperfctr *perfctr,
		     struct vperfctr_control *control)
{
    memcpy(perfctr->pmc_map, control->cpu_control.pmc_map, sizeof perfctr->pmc_map);
    return __vperfctr_control(perfctr->fd, perfctr->cpu_type, control);
}

int vperfctr_stop(struct vperfctr *perfctr)
{
    struct vperfctr_control control;
    memset(&control, 0, sizeof control);
    /* XXX: issue a SUSPEND command instead? */
    return vperfctr_control(perfctr, &control);
}

int vperfctr_is_running(const struct vperfctr *perfctr)
{
    return perfctr->kstate->cpu_state.cstatus != 0;
}

int vperfctr_iresume(const struct vperfctr *perfctr)
{
    return _sys_vperfctr_iresume(perfctr->fd);
}

int vperfctr_unlink(const struct vperfctr *perfctr)
{
    return _sys_vperfctr_unlink(perfctr->fd);
}

void vperfctr_close(struct vperfctr *perfctr)
{
    munmap((void*)perfctr->mapping, perfctr->mapping_size);
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

int rvperfctr_control(struct rvperfctr *rvperfctr,
		      struct vperfctr_control *control)
{
    return vperfctr_control(&rvperfctr->vperfctr, control);
}

int rvperfctr_stop(struct rvperfctr *rvperfctr)
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
