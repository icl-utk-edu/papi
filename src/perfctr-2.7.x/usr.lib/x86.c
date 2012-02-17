/* $Id: x86.c,v 1.23 2007/10/06 13:02:07 mikpe Exp $
 * x86-specific perfctr library procedures.
 *
 * Copyright (C) 1999-2007  Mikael Pettersson
 */
#include <errno.h>
#include <asm/unistd.h>
#include <stdio.h>
#include <string.h>	/* memset() */
#include "libperfctr.h"
#include "x86.h"
#include "x86_cpuinfo.h"

static unsigned int __NR_vperfctr_open;
#define __NR_vperfctr_control	(__NR_vperfctr_open+1)
#define __NR_vperfctr_write	(__NR_vperfctr_open+2)
#define __NR_vperfctr_read	(__NR_vperfctr_open+3)

#include <unistd.h>

static void init_sys_vperfctr(void)
{
    if (!__NR_vperfctr_open) {
	unsigned int nr;
	unsigned int kver = perfctr_linux_version_code();

#if defined(__x86_64__)
	if (kver >= PERFCTR_KERNEL_VERSION(2,6,18))
	    nr = 286;
	else if (kver >= PERFCTR_KERNEL_VERSION(2,6,16))
	    nr = 280;
	else
	    nr = 257;
#elif defined(__i386__)
	if (kver >= PERFCTR_KERNEL_VERSION(2,6,18))
	    nr = 325;
	else if (kver >= PERFCTR_KERNEL_VERSION(2,6,16))
	    nr = 318;
	else
	    nr = 296;
#endif
	__NR_vperfctr_open = nr;
    }
}

/*
 * The actual syscalls.
 */

int _sys_vperfctr_open(int fd_unused, int tid, int creat)
{
    init_sys_vperfctr();
    return syscall(__NR_vperfctr_open, tid, creat);
}

static int _sys_vperfctr_control(int fd, unsigned int cmd)
{
    init_sys_vperfctr();
    return syscall(__NR_vperfctr_control, fd, cmd);
}

static int _sys_vperfctr_write(int fd, unsigned int domain, const void *arg, unsigned int argbytes)
{
    init_sys_vperfctr();
    return syscall(__NR_vperfctr_write, fd, domain, arg, argbytes);
}

static int _sys_vperfctr_read(int fd, unsigned int domain, void *arg, unsigned int argbytes)
{
    init_sys_vperfctr();
    return syscall(__NR_vperfctr_read, fd, domain, arg, argbytes);
}

/*
 * Simple syscall wrappers.
 */

int _sys_vperfctr_read_sum(int fd, struct perfctr_sum_ctrs *arg)
{
    return _sys_vperfctr_read(fd, VPERFCTR_DOMAIN_SUM, arg, sizeof(*arg));
}

int _sys_vperfctr_read_children(int fd, struct perfctr_sum_ctrs *arg)
{
    return _sys_vperfctr_read(fd, VPERFCTR_DOMAIN_CHILDREN, arg, sizeof(*arg));
}

int _sys_vperfctr_unlink(int fd)
{
    return _sys_vperfctr_control(fd, VPERFCTR_CONTROL_UNLINK);
}

int _sys_vperfctr_iresume(int fd)
{
    return _sys_vperfctr_control(fd, VPERFCTR_CONTROL_RESUME);
}

/*
 * Complex syscall wrappers, for transmitting control data
 * in CPU family specific formats.
 */

#define MSR_P5_CESR		0x11
#define MSR_P6_PERFCTR0		0xC1		/* .. 0xC2 */
#define MSR_P6_EVNTSEL0		0x186		/* .. 0x187 */
#define MSR_K7_EVNTSEL0		0xC0010000	/* .. 0xC0010003 */
#define MSR_K7_PERFCTR0		0xC0010004	/* .. 0xC0010007 */
#define MSR_P4_PERFCTR0		0x300		/* .. 0x311 */
#define MSR_P4_CCCR0		0x360		/* .. 0x371 */
#define MSR_P4_ESCR0		0x3A0		/* .. 0x3E1, with some gaps */
#define MSR_P4_PEBS_ENABLE	0x3F1
#define MSR_P4_PEBS_MATRIX_VERT	0x3F2
#define P4_CCCR_ESCR_SELECT(X)	(((X) >> 13) & 0x7)
#define P4_FAST_RDPMC		0x80000000

#if 0
static void show_regs(const struct perfctr_cpu_reg *regs, unsigned int n)
{
    unsigned int i;

    fprintf(stderr, "CPU Register Values:\n");
    for(i = 0; i < n; ++i)
	fprintf(stderr, "MSR %#x\t0x%08x\n", regs[i].nr, regs[i].value);
}
#else
#define show_regs(regs, n) do{}while(0)
#endif

static int read_packet(int fd, unsigned int domain, void *arg, unsigned int argbytes)
{
    int ret;

    ret = _sys_vperfctr_read(fd, domain, arg, argbytes);
    if (ret != argbytes && ret >= 0) {
	errno = EPROTO;
	return -1;
    }
    return ret;
}

#if !defined(__x86_64__)
static int p5_write_regs(int fd, const struct perfctr_cpu_control *arg)
{
    struct perfctr_cpu_reg reg;
    unsigned short cesr_half[2];
    unsigned int i, pmc;

    if (!arg->nractrs)
	return 0;
    memset(cesr_half, 0, sizeof cesr_half);
    for(i = 0; i < arg->nractrs; ++i) {
	pmc = arg->pmc_map[i];
	if (pmc > 1) {
	    errno = EINVAL;
	    return -1;
	}
	cesr_half[pmc] = arg->evntsel[i];
    }
    reg.nr = MSR_P5_CESR;
    reg.value = (cesr_half[1] << 16) | cesr_half[0];
    show_regs(&reg, 1);
    return _sys_vperfctr_write(fd, PERFCTR_DOMAIN_CPU_REGS, &reg, sizeof reg);
}

static int p5_read_regs(int fd, struct perfctr_cpu_control *arg)
{
    struct perfctr_cpu_reg reg;
    unsigned short cesr_half[2];
    unsigned int i, pmc;
    int ret;

    if (!arg->nractrs)
	return 0;
    reg.nr = MSR_P5_CESR;
    ret = read_packet(fd, PERFCTR_DOMAIN_CPU_REGS, &reg, sizeof reg);
    if (ret < 0)
	return ret;
    show_regs(&reg, 1);
    cesr_half[0] = reg.value & 0xffff;
    cesr_half[1] = reg.value >> 16;
    for(i = 0; i < arg->nractrs; ++i) {
	pmc = arg->pmc_map[i];
	if (pmc > 1) {
	    errno = EINVAL;
	    return -1;
	}
	arg->evntsel[i] = cesr_half[pmc];
    }
    return 0;
}
#endif

static int p6_like_read_write_regs(int fd, struct perfctr_cpu_control *control,
				   unsigned int msr_evntsel0, unsigned int msr_perfctr0,
				   int do_write)
{
    struct perfctr_cpu_reg regs[4+4];
    unsigned int nrctrs, nractrs, pmc_mask, nr_regs, i, pmc;
    int ret;

    nractrs = control->nractrs;
    nrctrs = nractrs + control->nrictrs;
    if (nrctrs < nractrs || nrctrs > 4) {
	errno = EINVAL;
	return -1;
    }

    if (!nrctrs)
	return 0;

    nr_regs = 0;
    pmc_mask = 0;
    for(i = 0; i < nrctrs; ++i) {
	pmc = control->pmc_map[i];
	if (pmc >= 4 || (pmc_mask & (1<<pmc))) {
	    errno = EINVAL;
	    return -1;
	}
	pmc_mask |= (1<<pmc);
	regs[nr_regs].nr = msr_evntsel0 + pmc;
	regs[nr_regs].value = control->evntsel[i];
	++nr_regs;
	if (i >= nractrs) {
	    regs[nr_regs].nr = msr_perfctr0 + pmc;
	    regs[nr_regs].value = control->ireset[i];
	    ++nr_regs;
	}
    }
    if (do_write) {
	show_regs(regs, nr_regs);
	return _sys_vperfctr_write(fd, PERFCTR_DOMAIN_CPU_REGS, regs, nr_regs*sizeof(regs[0]));
    }
    ret = read_packet(fd, PERFCTR_DOMAIN_CPU_REGS, regs, nr_regs*sizeof(regs[0]));
    if (ret < 0)
	return ret;
    show_regs(regs, nr_regs);
    nr_regs = 0;
    for(i = 0; i < nrctrs; ++i) {
	control->evntsel[i] = regs[nr_regs].value;
	++nr_regs;
	if (i >= nractrs) {
	    control->ireset[i] = regs[nr_regs].value;
	    ++nr_regs;
	}
    }
    return 0;
}

/*
 * Table 15-4 in the IA32 Volume 3 manual contains a 18x8 entry mapping
 * from counter/CCCR number (0-17) and ESCR SELECT value (0-7) to the
 * actual ESCR MSR number. This mapping contains some repeated patterns,
 * so we can compact it to a 4x8 table of MSR offsets:
 *
 * 1. CCCRs 16 and 17 are mapped just like CCCRs 13 and 14, respectively.
 *    Thus, we only consider the 16 CCCRs 0-15.
 * 2. The CCCRs are organised in pairs, and both CCCRs in a pair use the
 *    same mapping. Thus, we only consider the 8 pairs 0-7.
 * 3. In each pair of pairs, the second odd-numbered pair has the same domain
 *    as the first even-numbered pair, and the range is 1+ the range of the
 *    the first even-numbered pair. For example, CCCR(0) and (1) map ESCR
 *    SELECT(7) to 0x3A0, and CCCR(2) and (3) map it to 0x3A1.
 *    The only exception is that pair (7) [CCCRs 14 and 15] does not have
 *    ESCR SELECT(3) in its domain, like pair (6) [CCCRs 12 and 13] has.
 *    NOTE: Revisions of IA32 Volume 3 older than #245472-007 had an error
 *    in this table: CCCRs 12, 13, and 16 had their mappings for ESCR SELECT
 *    values 2 and 3 swapped.
 * 4. All MSR numbers are on the form 0x3??. Instead of storing these as
 *    16-bit numbers, the table only stores the 8-bit offsets from 0x300.
 */

static const unsigned char p4_cccr_escr_map[4][8] = {
	/* 0x00 and 0x01 as is, 0x02 and 0x03 are +1 */
	[0x00/4] {	[7] 0xA0,
			[6] 0xA2,
			[2] 0xAA,
			[4] 0xAC,
			[0] 0xB2,
			[1] 0xB4,
			[3] 0xB6,
			[5] 0xC8, },
	/* 0x04 and 0x05 as is, 0x06 and 0x07 are +1 */
	[0x04/4] {	[0] 0xC0,
			[2] 0xC2,
			[1] 0xC4, },
	/* 0x08 and 0x09 as is, 0x0A and 0x0B are +1 */
	[0x08/4] {	[1] 0xA4,
			[0] 0xA6,
			[5] 0xA8,
			[2] 0xAE,
			[3] 0xB0, },
	/* 0x0C, 0x0D, and 0x10 as is,
	   0x0E, 0x0F, and 0x11 are +1 except [3] is not in the domain */
	[0x0C/4] {	[4] 0xB8,
			[5] 0xCC,
			[6] 0xE0,
			[0] 0xBA,
			[2] 0xBC,
			[3] 0xBE,
			[1] 0xCA, },
};

static unsigned int p4_escr_addr(unsigned int pmc, unsigned int cccr_val)
{
	unsigned int escr_select, pair, escr_offset;

	escr_select = P4_CCCR_ESCR_SELECT(cccr_val);
	if (pmc > 0x11)
		return 0;	/* pmc range error */
	if (pmc > 0x0F)
		pmc -= 3;	/* 0 <= pmc <= 0x0F */
	pair = pmc / 2;		/* 0 <= pair <= 7 */
	escr_offset = p4_cccr_escr_map[pair / 2][escr_select];
	if (!escr_offset || (pair == 7 && escr_select == 3))
		return 0;	/* ESCR SELECT range error */
	return escr_offset + (pair & 1) + 0x300;
};

static int p4_read_write_regs(int fd, struct perfctr_cpu_control *control, int do_write)
{
    struct perfctr_cpu_reg regs[18*3+2];
    unsigned int nrctrs, nractrs, pmc_mask, nr_regs, i, pmc, escr_addr;
    int ret;

    nractrs = control->nractrs;
    nrctrs = nractrs + control->nrictrs;
    if (nrctrs < nractrs || nrctrs > 18) {
	errno = EINVAL;
	return -1;
    }

    if (!nrctrs)
	return 0;

    nr_regs = 0;
    pmc_mask = 0;
    for(i = 0; i < nrctrs; ++i) {
	pmc = control->pmc_map[i] & ~P4_FAST_RDPMC;
	if (pmc >= 18 || (pmc_mask & (1<<pmc))) {
	    errno = EINVAL;
	    return -1;
	}
	pmc_mask |= (1<<pmc);
	regs[nr_regs].nr = MSR_P4_CCCR0 + pmc;
	regs[nr_regs].value = control->evntsel[i];
	++nr_regs;
	escr_addr = p4_escr_addr(pmc, control->evntsel[i]);
	if (!escr_addr) {
	    errno = EINVAL;
	    return -1;
	}
	regs[nr_regs].nr = escr_addr;
	regs[nr_regs].value = control->p4.escr[i];
	++nr_regs;
	if (i >= nractrs) {
	    regs[nr_regs].nr = MSR_P4_PERFCTR0 + pmc;
	    regs[nr_regs].value = control->ireset[i];
	    ++nr_regs;
	}
    }
    regs[nr_regs].nr = MSR_P4_PEBS_ENABLE;
    regs[nr_regs].value = control->p4.pebs_enable;
    ++nr_regs;
    regs[nr_regs].nr = MSR_P4_PEBS_MATRIX_VERT;
    regs[nr_regs].value = control->p4.pebs_matrix_vert;
    ++nr_regs;
    if (do_write) {
	show_regs(regs, nr_regs);
	return _sys_vperfctr_write(fd, PERFCTR_DOMAIN_CPU_REGS, regs, nr_regs*sizeof(regs[0]));
    }
    ret = read_packet(fd, PERFCTR_DOMAIN_CPU_REGS, regs, nr_regs*sizeof(regs[0]));
    if (ret < 0)
	return ret;
    show_regs(regs, nr_regs);
    nr_regs = 0;
    for(i = 0; i < nrctrs; ++i) {
	control->evntsel[i] = regs[nr_regs].value;
	++nr_regs;
	control->p4.escr[i] = regs[nr_regs].value;
	++nr_regs;
	if (i >= nractrs) {
	    control->ireset[i] = regs[nr_regs].value;
	    ++nr_regs;
	}
    }
    control->p4.pebs_enable = regs[nr_regs].value;
    ++nr_regs;
    control->p4.pebs_matrix_vert = regs[nr_regs].value;
    ++nr_regs;
    return 0;
}

static int write_cpu_regs(int fd, unsigned int cpu_type, struct perfctr_cpu_control *arg)
{
    switch (cpu_type) {
      case PERFCTR_X86_GENERIC:
	return 0;
#if !defined(__x86_64__)
      case PERFCTR_X86_INTEL_P5:
      case PERFCTR_X86_INTEL_P5MMX:
      case PERFCTR_X86_CYRIX_MII:
      case PERFCTR_X86_WINCHIP_C6:
      case PERFCTR_X86_WINCHIP_2:
	return p5_write_regs(fd, arg);
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
      case PERFCTR_X86_INTEL_PENTM:
      case PERFCTR_X86_VIA_C3:
	return p6_like_read_write_regs(fd, arg, MSR_P6_EVNTSEL0, MSR_P6_PERFCTR0, 1);
      case PERFCTR_X86_AMD_K7:
#endif
      case PERFCTR_X86_AMD_K8:
      case PERFCTR_X86_AMD_K8C:
	return p6_like_read_write_regs(fd, arg, MSR_K7_EVNTSEL0, MSR_K7_PERFCTR0, 1);
#if !defined(__x86_64__)
      case PERFCTR_X86_INTEL_P4:
      case PERFCTR_X86_INTEL_P4M2:
#endif
      case PERFCTR_X86_INTEL_P4M3:
	return p4_read_write_regs(fd, arg, 1);
	break;
      default:
	fprintf(stderr, "unable to write control registers for cpu type %u\n",
		cpu_type);
	errno = EINVAL;
	return -1;
    }
}

int _sys_vperfctr_write_control(int fd, unsigned int cpu_type, const struct vperfctr_control *control)
{
    union {
	struct vperfctr_control_kernel control;
	struct perfctr_cpu_control_header header;
    } u;
    unsigned int nrctrs;
    int ret;
    
    ret = _sys_vperfctr_control(fd, VPERFCTR_CONTROL_CLEAR);
    if (ret < 0)
	return ret;

    u.control.si_signo = control->si_signo;
    u.control.preserve = control->preserve;
    ret = _sys_vperfctr_write(fd, VPERFCTR_DOMAIN_CONTROL,
			      &u.control, sizeof u.control);
    if (ret < 0)
	return ret;

    u.header.tsc_on = control->cpu_control.tsc_on;
    u.header.nractrs = control->cpu_control.nractrs;
    u.header.nrictrs = control->cpu_control.nrictrs;
    ret = _sys_vperfctr_write(fd, PERFCTR_DOMAIN_CPU_CONTROL,
			      &u.header, sizeof u.header);
    if (ret < 0)
	return ret;

    nrctrs = control->cpu_control.nractrs + control->cpu_control.nrictrs;
    ret = _sys_vperfctr_write(fd, PERFCTR_DOMAIN_CPU_MAP,
			      &control->cpu_control.pmc_map,
			      nrctrs * sizeof control->cpu_control.pmc_map[0]);
    if (ret < 0)
	return ret;

    ret = write_cpu_regs(fd, cpu_type, (struct perfctr_cpu_control*)&control->cpu_control);
    if (ret < 0)
	return ret;

    return _sys_vperfctr_control(fd, VPERFCTR_CONTROL_RESUME);
}

static int read_cpu_regs(int fd, unsigned int cpu_type, struct perfctr_cpu_control *arg)
{
    switch (cpu_type) {
      case PERFCTR_X86_GENERIC:
	return 0;
#if !defined(__x86_64__)
      case PERFCTR_X86_INTEL_P5:
      case PERFCTR_X86_INTEL_P5MMX:
      case PERFCTR_X86_CYRIX_MII:
      case PERFCTR_X86_WINCHIP_C6:
      case PERFCTR_X86_WINCHIP_2:
	return p5_read_regs(fd, arg);
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
      case PERFCTR_X86_INTEL_PENTM:
      case PERFCTR_X86_VIA_C3:
	return p6_like_read_write_regs(fd, arg, MSR_P6_EVNTSEL0, MSR_P6_PERFCTR0, 0);
      case PERFCTR_X86_AMD_K7:
#endif
      case PERFCTR_X86_AMD_K8:
      case PERFCTR_X86_AMD_K8C:
	return p6_like_read_write_regs(fd, arg, MSR_K7_EVNTSEL0, MSR_K7_PERFCTR0, 0);
#if !defined(__x86_64__)
      case PERFCTR_X86_INTEL_P4:
      case PERFCTR_X86_INTEL_P4M2:
#endif
      case PERFCTR_X86_INTEL_P4M3:
	return p4_read_write_regs(fd, arg, 0);
	break;
      default:
	fprintf(stderr, "unable to read control registers for cpu type %u\n",
		cpu_type);
	errno = EINVAL;
	return -1;
    }
}

int _sys_vperfctr_read_control(int fd, unsigned int cpu_type, struct vperfctr_control *control)
{
    union {
	struct vperfctr_control_kernel control;
	struct perfctr_cpu_control_header header;
    } u;
    unsigned int nrctrs;
    int ret;
    
    memset(control, 0, sizeof *control);

    ret = read_packet(fd, VPERFCTR_DOMAIN_CONTROL,
		      &u.control, sizeof u.control);
    if (ret < 0)
	return ret;
    control->si_signo = u.control.si_signo;
    control->preserve = u.control.preserve;

    ret = read_packet(fd, PERFCTR_DOMAIN_CPU_CONTROL,
		      &u.header, sizeof u.header);
    if (ret < 0)
	return ret;
    control->cpu_control.tsc_on = u.header.tsc_on;
    control->cpu_control.nractrs = u.header.nractrs;
    control->cpu_control.nrictrs = u.header.nrictrs;

    nrctrs = control->cpu_control.nractrs + control->cpu_control.nrictrs;
    ret = read_packet(fd, PERFCTR_DOMAIN_CPU_MAP,
		      &control->cpu_control.pmc_map,
		      nrctrs * sizeof control->cpu_control.pmc_map[0]);
    if (ret < 0)
	return ret;

    return read_cpu_regs(fd, cpu_type, &control->cpu_control);
}

static int intel_init(const struct cpuinfo *cpuinfo,
		      struct perfctr_info *info)
{
    unsigned int family, model, stepping;

    if (!cpu_has(cpuinfo, X86_FEATURE_TSC))
	return -1;
    family = cpu_family(cpuinfo);
    model = cpu_model(cpuinfo);
    stepping = cpu_stepping(cpuinfo);
    switch (family) {
      case 5:
	if (cpu_has(cpuinfo, X86_FEATURE_MMX)) {
	    /* Avoid Pentium Erratum 74. */
	    if (model == 4 &&
		(stepping == 4 ||
		 (stepping == 3 &&
		  cpu_type(cpuinfo) == 1)))
		info->cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	    return PERFCTR_X86_INTEL_P5MMX;
	} else {
	    info->cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	    return PERFCTR_X86_INTEL_P5;
	}
      case 6:
	if (model == 9 || model == 13)
	    return PERFCTR_X86_INTEL_PENTM;
	else if (model >= 7)
	    return PERFCTR_X86_INTEL_PIII;
	else if (model >= 3)
	    return PERFCTR_X86_INTEL_PII;
	else {
	    /* Avoid Pentium Pro Erratum 26. */
	    if (stepping < 9)
		info->cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	    return PERFCTR_X86_INTEL_P6;
	}
      case 15:
	if (model >= 3)
	    return PERFCTR_X86_INTEL_P4M3;
	else if (model >= 2)
	    return PERFCTR_X86_INTEL_P4M2;
	else
	    return PERFCTR_X86_INTEL_P4;
    }
    return -1;
}

static int amd_init(const struct cpuinfo *cpuinfo,
		    struct perfctr_info *info)
{
    unsigned int family, model, stepping;

    if (!cpu_has(cpuinfo, X86_FEATURE_TSC))
	return -1;
    family = cpu_family(cpuinfo);
    model = cpu_model(cpuinfo);
    stepping = cpu_stepping(cpuinfo);
    switch (family) {
      case 15:
	if (model > 5 || (model >= 4 && stepping >= 8))
	    return PERFCTR_X86_AMD_K8C;
	else
	    return PERFCTR_X86_AMD_K8;
      case 6:
	return PERFCTR_X86_AMD_K7;
    }
    return -1;
}

static int cyrix_init(const struct cpuinfo *cpuinfo,
		      struct perfctr_info *info)
{
    if (!cpu_has(cpuinfo, X86_FEATURE_TSC))
	return -1;
    switch (cpu_family(cpuinfo)) {
      case 6: /* 6x86MX, MII, or III */
	return PERFCTR_X86_CYRIX_MII;
    }
    return -1;
}

static int centaur_init(const struct cpuinfo *cpuinfo,
			struct perfctr_info *info)
{
    unsigned int family, model;

    family = cpu_family(cpuinfo);
    model = cpu_model(cpuinfo);
    switch (family) {
      case 5:
	if (cpu_has(cpuinfo, X86_FEATURE_TSC))
	    return -1;
	info->cpu_features &= ~PERFCTR_FEATURE_RDTSC;
	switch (model) {
	  case 4:
	    return PERFCTR_X86_WINCHIP_C6;
	  case 8: /* WinChip 2, 2A, or 2B */
	  case 9: /* WinChip 3 */
	    return PERFCTR_X86_WINCHIP_2;
	  default:
	    return -1;
	}
      case 6:
	if (!cpu_has(cpuinfo, X86_FEATURE_TSC))
	    return -1;
	switch (model) {
	  case 6: /* Cyrix III */
	  case 7: /* Samuel 2 */
	  case 8: /* Ezra-T */
	  case 9: /* Antaur/Nehemiah */
	    return PERFCTR_X86_VIA_C3;
	  default:
	    return -1;
	}
    }
    return -1;
}

static int generic_init(const struct cpuinfo *cpuinfo,
			struct perfctr_info *info)
{
    if (!cpu_has(cpuinfo, X86_FEATURE_TSC))
	return -1;
    info->cpu_features &= ~PERFCTR_FEATURE_RDPMC;
    return PERFCTR_X86_GENERIC;
}

void perfctr_info_cpu_init(struct perfctr_info *info)
{
    struct cpuinfo cpuinfo;
    int cpu_type;

    identify_cpu(&cpuinfo);
    cpu_type = -1; /* binary compat prevents using 0 for "unknown" */
    if (cpu_has(&cpuinfo, X86_FEATURE_MSR)) {
	switch (cpuinfo.vendor) {
	  case X86_VENDOR_INTEL:
	    cpu_type = intel_init(&cpuinfo, info);
	    break;
	  case X86_VENDOR_AMD:
	    cpu_type = amd_init(&cpuinfo, info);
	    break;
	  case X86_VENDOR_CYRIX:
	    cpu_type = cyrix_init(&cpuinfo, info);
	    break;
	  case X86_VENDOR_CENTAUR:	
	    cpu_type = centaur_init(&cpuinfo, info);
	    break;
	}
    }
    if (cpu_type < 0)
	cpu_type = generic_init(&cpuinfo, info);
    info->cpu_type = cpu_type;
}

unsigned int perfctr_info_nrctrs(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
#if !defined(__x86_64__)
      case PERFCTR_X86_INTEL_P5:
      case PERFCTR_X86_INTEL_P5MMX:
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
      case PERFCTR_X86_CYRIX_MII:
      case PERFCTR_X86_WINCHIP_C6:
      case PERFCTR_X86_WINCHIP_2:
      case PERFCTR_X86_INTEL_PENTM:
	return 2;
      case PERFCTR_X86_AMD_K7:
	return 4;
      case PERFCTR_X86_VIA_C3:
	return 1;
      case PERFCTR_X86_INTEL_P4:
      case PERFCTR_X86_INTEL_P4M2:
	return 18;
#endif
      case PERFCTR_X86_INTEL_P4M3:
	return 18;
      case PERFCTR_X86_AMD_K8:
      case PERFCTR_X86_AMD_K8C:
	return 4;
      case PERFCTR_X86_GENERIC:
      default:
	return 0;
    }
}

const char *perfctr_info_cpu_name(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_X86_GENERIC:
	return "Generic x86 with TSC";
#if !defined(__x86_64__)
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
      case PERFCTR_X86_INTEL_P4M2:
	return "Intel Pentium 4 Model 2";
      case PERFCTR_X86_INTEL_PENTM:
	return "Intel Pentium M";
#endif
      case PERFCTR_X86_INTEL_P4M3:
	return "Intel Pentium 4 Model 3";
      case PERFCTR_X86_AMD_K8:
	return "AMD K8";
      case PERFCTR_X86_AMD_K8C:
	return "AMD K8 Revision C";
      default:
        return "?";
    }
}

void perfctr_cpu_control_print(const struct perfctr_cpu_control *control)
{
    unsigned int i, nractrs, nrictrs, nrctrs;

    nractrs = control->nractrs;
    nrictrs = control->nrictrs;
    nrctrs = control->nractrs + nrictrs;

    printf("tsc_on\t\t\t%u\n", control->tsc_on);
    printf("nractrs\t\t\t%u\n", nractrs);
    if( nrictrs )
	printf("nrictrs\t\t\t%u\n", nrictrs);
    for(i = 0; i < nrctrs; ++i) {
        if( control->pmc_map[i] >= 18 ) /* for P4 'fast rdpmc' cases */
            printf("pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i]);
        else
            printf("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
        printf("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
        if( control->p4.escr[i] )
            printf("escr[%u]\t\t\t0x%08X\n", i, control->p4.escr[i]);
	if( i >= nractrs )
	    printf("ireset[%u]\t\t%d\n", i, control->ireset[i]);
    }
    if( control->p4.pebs_enable )
	printf("pebs_enable\t\t0x%08X\n", control->p4.pebs_enable);
    if( control->p4.pebs_matrix_vert )
	printf("pebs_matrix_vert\t0x%08X\n", control->p4.pebs_matrix_vert);
}
