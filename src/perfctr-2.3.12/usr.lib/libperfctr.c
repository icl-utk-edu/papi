/* $Id$
 * Library interface to Linux x86 Performance-Monitoring Counters.
 *
 * Copyright (C) 1999-2001  Mikael Pettersson
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

/*
 * Operations on the process' own virtual-mode perfctrs.
 */

struct vperfctr {
    /* XXX: point to &vperfctr_state.cpu_state instead? */
    volatile const struct vperfctr_state *kstate;
    int fd;
    unsigned char have_rdpmc;
};

struct vperfctr *vperfctr_open(void)
{
    struct vperfctr *perfctr;
    struct perfctr_info info;
    int isnew;

    perfctr = malloc(sizeof(*perfctr));
    if( !perfctr )
	return NULL;
    isnew = 1;
    perfctr->fd = open("/proc/self/perfctr", O_RDONLY|O_CREAT);
    if( perfctr->fd < 0 ) {
	isnew = 0;
	perfctr->fd = open("/proc/self/perfctr", O_RDONLY);
	if( perfctr->fd < 0 )
	    goto out_perfctr;
    }
    if( ioctl(perfctr->fd, PERFCTR_INFO, &info) != 0 )
	goto out_fd;
    perfctr->have_rdpmc = (info.cpu_features & PERFCTR_FEATURE_RDPMC) != 0;
    perfctr->kstate = mmap(NULL, PAGE_SIZE, PROT_READ,
			   MAP_SHARED, perfctr->fd, 0);
    if( perfctr->kstate != MAP_FAILED ) {
	if( perfctr->kstate->magic == VPERFCTR_MAGIC )
	    return perfctr;
	fprintf(stderr, __FILE__ ":" __FUNCTION__
		": kstate version mismatch, kernel %#x, expected %#x\n",
		perfctr->kstate->magic, VPERFCTR_MAGIC);
	munmap((void*)perfctr->kstate, PAGE_SIZE);
    }
 out_fd:
    if( isnew )
	vperfctr_unlink(perfctr);
    close(perfctr->fd);
 out_perfctr:
    free(perfctr);
    return NULL;
}

int vperfctr_info(const struct vperfctr *vperfctr, struct perfctr_info *info)
{
    return ioctl(vperfctr->fd, PERFCTR_INFO, info);
}

#define rdtscl(low)	\
	__asm__ __volatile__("rdtsc" : "=a"(low) : : "edx")
#define rdpmcl(ctr,low)	\
	__asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")

#if (__GNUC__ < 2) ||  (__GNUC__ == 2 && __GNUC_MINOR__ < 96)
#define __builtin_expect(x, expected_value) (x)
#endif

unsigned long long vperfctr_read_tsc(const struct vperfctr *self)
{
    unsigned long long sum;
    unsigned int tsc0, tsc1, now;
    volatile const struct vperfctr_state *kstate;

    kstate = self->kstate;
    if( __builtin_expect(kstate->cpu_state.cstatus != 0, 1) ) {
	tsc0 = kstate->cpu_state.start.tsc;
    retry:
	rdtscl(now);
	sum = kstate->cpu_state.sum.tsc;
	tsc1 = kstate->cpu_state.start.tsc;
	if( __builtin_expect(tsc1 == tsc0, 1) )
	    return sum += (now - tsc0);
	tsc0 = tsc1;
	goto retry; /* better gcc code than with a do{}while() loop */
    }
    return kstate->cpu_state.sum.tsc;
}

unsigned long long vperfctr_read_pmc(const struct vperfctr *self, unsigned i)
{
    unsigned long long sum;
    unsigned int start, now;
    unsigned int tsc0, tsc1;
    volatile const struct vperfctr_state *kstate;
    unsigned int cstatus;

    kstate = self->kstate;
    cstatus = kstate->cpu_state.cstatus;
    /* gcc 3.0 generates crap code for __builtin_expect(E1 && E2) :-( */
    if( perfctr_cstatus_has_tsc(cstatus) && self->have_rdpmc ) {
	 tsc0 = kstate->cpu_state.start.tsc;
    retry:
	 rdpmcl(kstate->cpu_state.control.pmc_map[i], now);
	 start = kstate->cpu_state.start.pmc[i];
	 sum = kstate->cpu_state.sum.pmc[i];
	 tsc1 = kstate->cpu_state.start.tsc;
	 if( __builtin_expect(tsc1 == tsc0, 1) ) {
	      return sum += (now - start);
	 }
	 tsc0 = tsc1;
	 goto retry;
    }
    if( cstatus != 0 )
	ioctl(self->fd, VPERFCTR_SAMPLE, NULL);
    return kstate->cpu_state.sum.pmc[i];
}

void vperfctr_read_ctrs(const struct vperfctr *self,
			struct perfctr_sum_ctrs *sum)
{
    unsigned int tsc0, tsc1, now;
    unsigned int cstatus, nrctrs;
    volatile const struct vperfctr_state *kstate;
    int i;

    /* Fast path is impossible if the TSC isn't being sampled (bad idea,
       but on WinChip you don't have a choice), or at least one PMC is
       enabled but the CPU doesn't have RDPMC. */
    kstate = self->kstate;
    cstatus = kstate->cpu_state.cstatus;
    nrctrs = perfctr_cstatus_nractrs(cstatus);
    if( perfctr_cstatus_has_tsc(cstatus) && (!nrctrs || self->have_rdpmc) ) {
    retry:
	tsc0 = kstate->cpu_state.start.tsc;
	rdtscl(now);
	sum->tsc = kstate->cpu_state.sum.tsc + (now - tsc0);
	for(i = nrctrs; --i >= 0;) {
	    rdpmcl(kstate->cpu_state.control.pmc_map[i], now);
	    sum->pmc[i] = kstate->cpu_state.sum.pmc[i] + (now - kstate->cpu_state.start.pmc[i]);
	}
	if( __builtin_expect(tsc0 == kstate->cpu_state.start.tsc, 1) )
	    return;
	goto retry;
    }
    ioctl(self->fd, VPERFCTR_SAMPLE, NULL);
    tsc1 = kstate->cpu_state.start.tsc;
    do {
	tsc0 = tsc1;
	sum->tsc = kstate->cpu_state.sum.tsc;
	for(i = 0; i < nrctrs; ++i)
	    sum->pmc[i] = kstate->cpu_state.sum.pmc[i];
	tsc1 = kstate->cpu_state.start.tsc;
    } while( tsc1 != tsc0 );
}

int vperfctr_read_state(const struct vperfctr *self, struct perfctr_sum_ctrs *sum,
			struct vperfctr_control *control)
{
    unsigned int prev_tsc, next_tsc;
    volatile const struct vperfctr_state *kstate;

    ioctl(self->fd, VPERFCTR_SAMPLE, NULL);
    kstate = self->kstate;
    next_tsc = kstate->cpu_state.start.tsc;
    do {
	prev_tsc = next_tsc;
	/* XXX: this copies more than necessary */
	if( sum )
	    *sum = kstate->cpu_state.sum;
	if( control ) {
	    control->si_signo = kstate->si_signo;
	    control->cpu_control = kstate->cpu_state.control;
	}
	next_tsc = kstate->cpu_state.start.tsc;
    } while( next_tsc != prev_tsc );
    return 0;
}

int vperfctr_control(const struct vperfctr *perfctr,
		     struct vperfctr_control *control)
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
    munmap((void*)perfctr->kstate, PAGE_SIZE);
    close(perfctr->fd);
    free(perfctr);
}

/*
 * Operations on global-mode perfctrs.
 */

struct gperfctr {
    int fd;
};

struct gperfctr *gperfctr_open(void)
{
    struct gperfctr *gperfctr;

    gperfctr = malloc(sizeof(*gperfctr));
    if( !gperfctr )
	return NULL;
    gperfctr->fd = open("/dev/perfctr", O_RDONLY);
    if( gperfctr->fd >= 0 ) {
	return gperfctr;
    }
    free(gperfctr);
    return NULL;
}

void gperfctr_close(struct gperfctr *gperfctr)
{
    close(gperfctr->fd);
    free(gperfctr);
}

int gperfctr_control(const struct gperfctr *gperfctr,
		     struct gperfctr_control *arg)
{
    return ioctl(gperfctr->fd, GPERFCTR_CONTROL, (long)arg);
}

int gperfctr_read(const struct gperfctr *gperfctr, struct gperfctr_state *arg)
{
    return ioctl(gperfctr->fd, GPERFCTR_READ, (long)arg);
}

int gperfctr_stop(const struct gperfctr *gperfctr)
{
    return ioctl(gperfctr->fd, GPERFCTR_STOP, 0);
}

int gperfctr_info(const struct gperfctr *gperfctr, struct perfctr_info *info)
{
    return ioctl(gperfctr->fd, PERFCTR_INFO, info);
}

/*
 * Miscellaneous operations.
 */

unsigned perfctr_cpu_nrctrs(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_X86_GENERIC:
	return 0;
      case PERFCTR_X86_VIA_C3:
	return 1;
      case PERFCTR_X86_AMD_K7:
	return 4;
      case PERFCTR_X86_INTEL_P4:
	return 18;
      default:
	return 2;
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
	return "AMD K7";
      case PERFCTR_X86_VIA_C3:
	return "VIA C3";
      case PERFCTR_X86_INTEL_P4:
	return "Intel Pentium 4";
      default:
        return "?";
    }
}

unsigned perfctr_evntsel_num_insns(const struct perfctr_info *info)
{
    /* This is terribly naive. Assumes only one event will be
     * selected, at CPL > 0.
     */
    switch( info->cpu_type ) {
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
      case PERFCTR_X86_VIA_C3:
	return 0xC0;
      default:
	return 0;
    }
}

void perfctr_print_info(const struct perfctr_info *info)
{
    static const char * const features[] = { "rdpmc", "rdtsc", "pcint" };
    int fi, comma;

    printf("driver_version\t\t%s\n", info->version);
    printf("nrcpus\t\t\t%u\n", info->nrcpus);
    printf("cpu_type\t\t%u (%s)\n", info->cpu_type, perfctr_cpu_name(info));
    printf("cpu_features\t\t%#x (", info->cpu_features);
    for(comma = 0, fi = 0; fi < sizeof features / sizeof features[0]; ++fi) {
	unsigned fmask = 1 << fi;
	if( info->cpu_features & fmask ) {
	    if( comma )
		printf(",");
	    printf("%s", features[fi]);
	    comma = 1;
	}
    }
    printf(")\n");
    printf("cpu_khz\t\t\t%lu\n", info->cpu_khz);
    printf("cpu_nrctrs\t\t%u\n", perfctr_cpu_nrctrs(info));
}
