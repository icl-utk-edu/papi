/* $Id$
 * Library interface to Linux x86 Performance-Monitoring Counters.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include "libperfctr.h"

#define PAGE_SIZE	4096

static unsigned char cpu_type;
static int have_rdpmc = 0;	/* false on pre-MMX Intel P5 */
static int have_rdtsc = 0;	/* false on WinChip when PMCs are used */
static int dev_perfctr_fd = -1;
static unsigned long cpu_khz;

struct vperfctr {
    int fd;
    volatile const struct vperfctr_state *kstate;
};

static int check_init_unlocked(void)
{
    if( dev_perfctr_fd < 0 ) {
	struct perfctr_info info;

	if( (dev_perfctr_fd = open("/dev/perfctr", O_RDONLY)) < 0 ) {
	    perror("open /dev/perfctr");
	    return -1;
	}
	if( ioctl(dev_perfctr_fd, PERFCTR_INFO, &info) < 0 )
	    return -1;
	cpu_type = info.cpu_type;
	have_rdpmc = info.cpu_features & PERFCTR_FEATURE_RDPMC;
	have_rdtsc = info.cpu_features & PERFCTR_FEATURE_RDTSC;
	cpu_khz = info.cpu_khz;
    }
    return 1;
}

#include <asm/atomic.h>
static int check_init_locked(void)
{
  int result;
  volatile static atomic_t lock = ATOMIC_INIT(0);
  while (atomic_inc_and_test(&lock) != 0)
    ;
  result = check_init_unlocked();
  atomic_dec(&lock);
  return result;
}
#define check_init()	check_init_locked()

/*
 * Invokes the kernel driver directly.
 */

int perfctr_syscall(unsigned cmd, long arg)
{
    if( check_init() < 0 )
	return -1;
    return ioctl(dev_perfctr_fd, cmd, arg);
}

/*
 * Operations on explicitly attached perfctrs.
 */

static int
perfctr_attach_simple(int pid,
		      struct perfctr_control *control,
		      int writable,
		      struct vperfctr *perfctr)
{
    int oerrno;
    unsigned cmd;

    cmd = writable ? VPERFCTR_ATTACH_RDWR : VPERFCTR_ATTACH_RDONLY;
    perfctr->fd = perfctr_syscall(cmd, pid);
    if( perfctr->fd < 0 )
	return -1;
    if( !control || !ioctl(perfctr->fd, VPERFCTR_CONTROL, control) ) {
	perfctr->kstate = mmap(NULL, PAGE_SIZE, PROT_READ,
			       MAP_SHARED, perfctr->fd, 0);
	if( perfctr->kstate != MAP_FAILED )
	    return 0;
    }
    oerrno = errno;
    close(perfctr->fd);
    errno = oerrno;
    perfctr->fd = -1;
    perfctr->kstate = NULL;
    return -1;
}

static struct vperfctr *
perfctr_attach_hairy(int pid, struct perfctr_control *control, int writable)
{
    int status, oerrno;
    struct vperfctr *perfctr;

    perfctr = malloc(sizeof(struct vperfctr));
    if( !perfctr )
	return perfctr;
    status = -1;
    if( pid == 0 || pid == getpid() ) {
	status = perfctr_attach_simple(pid, control, writable, perfctr);
    } else if( ptrace(PTRACE_ATTACH, pid, 0, 0) == 0 ) {
	if( waitpid(pid, NULL, WUNTRACED) > 0 )
	    status = perfctr_attach_simple(pid, control, writable, perfctr);
	oerrno = errno;
	ptrace(PTRACE_DETACH, pid, 0, SIGCONT);
	errno = oerrno;
    }
    if( status < 0 ) {
	oerrno = errno;
	free(perfctr);
	errno = oerrno;
	perfctr = NULL;
    }
    return perfctr;
}

struct vperfctr *perfctr_attach_rdonly(int pid)
{
    return perfctr_attach_hairy(pid, NULL, 0);
}

struct vperfctr *perfctr_attach_rdwr(int pid, struct perfctr_control *control)
{
    return perfctr_attach_hairy(pid, control, 1);
}

int perfctr_read(const struct vperfctr *perfctr, struct vperfctr_state *state)
{
    do {
	*state = *perfctr->kstate;
    } while( state->start.ctr[0] != perfctr->kstate->start.ctr[0] );
    return 0;
}

int perfctr_control(const struct vperfctr *perfctr,
		    struct perfctr_control *control)
{
    return ioctl(perfctr->fd, VPERFCTR_CONTROL, control);
}

int perfctr_stop(const struct vperfctr *perfctr)
{
    return ioctl(perfctr->fd, VPERFCTR_STOP, NULL);
}

int perfctr_unlink(const struct vperfctr *perfctr)
{
    return ioctl(perfctr->fd, VPERFCTR_UNLINK, NULL);
}

void perfctr_close(struct vperfctr *perfctr)
{
    if( perfctr->fd >= 0 ) {
	munmap((void*)perfctr->kstate, PAGE_SIZE);
	close(perfctr->fd);
	perfctr->kstate = NULL;
	perfctr->fd = -1;
    }
    free(perfctr);
}

/*
 * Operations on the process' own perfctrs.
 */

static struct vperfctr *perfctr_attach_self(int writable)
{
    return perfctr_attach_hairy(0, NULL, writable);
}

struct vperfctr *perfctr_attach_rdonly_self(void)
{
    return perfctr_attach_self(0);
}

struct vperfctr *perfctr_attach_rdwr_self(void)
{
    return perfctr_attach_self(1);
}

#define rdtscl(low)	\
	__asm__ __volatile__("rdtsc" : "=a"(low) : : "edx")
#define rdpmcl(ctr,low)	\
	__asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")

int perfctr_read_self(const struct vperfctr *self, struct vperfctr_state *state)
{
    unsigned int start_tsc;
    unsigned int diff;
    struct perfctr_low_ctrs now;
    int i, nrctrs;

    if( !have_rdpmc )		/* Intel pre-MMX P5 lossage */
	perfctr_syscall(VPERFCTR_SAMPLE, 0);
    /* cannot use RDPMC unless the counters actually are running.. */
    nrctrs = self->kstate->status;
    if( !have_rdpmc || nrctrs <= 0 )
	return perfctr_read(self, state);
    do {
	start_tsc = self->kstate->start.ctr[0];
	/* read current register contents */
	if( have_rdtsc )	/* WinChip lossage */
	    rdtscl(now.ctr[0]);
	for(i = 1; i < nrctrs; ++i)
	    rdpmcl(i-1, now.ctr[i]);
	/* accumulate */
	if( have_rdtsc ) {	/* WinChip lossage */
	    diff = now.ctr[0] - start_tsc;
	    state->sum.ctr[0] = self->kstate->sum.ctr[0] + diff;
	}
	for(i = 1; i < nrctrs; ++i) {
	    diff = now.ctr[i] - self->kstate->start.ctr[i];
	    state->sum.ctr[i] = self->kstate->sum.ctr[i] + diff;
	}
    } while( start_tsc != self->kstate->start.ctr[0] );
    state->control_id = self->kstate->control_id;
    state->control = self->kstate->control;
    state->children = self->kstate->children;
    state->status = self->kstate->status;
    return 0;
}

int perfctr_control_self(const struct vperfctr *self,
			 struct perfctr_control *control)
{
    return perfctr_control(self, control);
}

int perfctr_stop_self(const struct vperfctr *self)
{
    return perfctr_stop(self);
}

int perfctr_unlink_self(void)
{
    return perfctr_syscall(VPERFCTR_UNLINK, 0);
}

void perfctr_close_self(struct vperfctr *self)
{
    return perfctr_close(self);
}

/*
 * Operations on global-mode perfctrs.
 */

int perfctr_global_control(struct gperfctr_control *arg)
{
    return perfctr_syscall(GPERFCTR_CONTROL, (long)arg);
}

int perfctr_global_read(struct gperfctr_state *arg)
{
    return perfctr_syscall(GPERFCTR_READ, (long)arg);
}

int perfctr_global_stop(void)
{
    return perfctr_syscall(GPERFCTR_STOP, 0);
}

/*
 * Miscellaneous operations.
 */

int perfctr_info(struct perfctr_info *info)
{
    return perfctr_syscall(PERFCTR_INFO, (long)info);
}

unsigned perfctr_cpu_nrctrs(void)
{
    if( check_init() < 0 )
	return 0;
    switch( cpu_type ) {
      case PERFCTR_X86_GENERIC:
	return 1;
      case PERFCTR_X86_AMD_K7:
	return 5;
      default:
	return 3;
    }
}

const char *perfctr_cpu_name(void)
{
    if( check_init() < 0 )
	return NULL;
    switch( cpu_type ) {
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

unsigned perfctr_evntsel_num_insns(void)
{
    /* This is terribly naive. Assumes only one event will be
     * selected, at CPL > 0.
     */
    if( check_init() < 0 )
	return (unsigned)-1;
    switch( cpu_type ) {
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
        fprintf(stderr, __FUNCTION__ ": cpu_type %u not recognized\n", cpu_type);
        exit(1);
    }
}
