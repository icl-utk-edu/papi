/* 
 * PPC64-specific perfctr library procedures.
 * Copyright (C) 2004, 2007  Mikael Pettersson
 * Copyright (C) 2004  Maynard Johnson
 *
 */
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <asm/unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include "libperfctr.h"
#include "ppc64.h"

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

	if (kver >= PERFCTR_KERNEL_VERSION(2,6,18))
	    nr = 310;
	else if (kver >= PERFCTR_KERNEL_VERSION(2,6,16))
	    nr = 301;
	else
	    nr = 280;
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

int _sys_vperfctr_control(int fd, unsigned int cmd)
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

#define	SPRN_PVR	0x11F	/* Processor Version Register */
#define SPRN_MMCRA	786
#define SPRN_MMCR0	795
#define SPRN_MMCR1	798
#define SPRN_PMC1	787
#define SPRN_PMC2	788
#define SPRN_PMC3	789
#define SPRN_PMC4	790
#define SPRN_PMC5	791
#define SPRN_PMC6	792
#define SPRN_PMC7	793
#define SPRN_PMC8	794

#if 0
static void show_regs(const struct perfctr_cpu_reg *regs, unsigned int n)
{
    unsigned int i;

    fprintf(stderr, "CPU Register Values:\n");
    for(i = 0; i < n; ++i)
	fprintf(stderr, "SPR %#x\t0x%08x\n", regs[i].nr, regs[i].value);
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

static unsigned int pmc_to_spr(unsigned int pmc)
{
    switch (pmc) {
    default: /* impossible, but silences gcc warning */
    case (1-1): return SPRN_PMC1;
    case (2-1): return SPRN_PMC2;
    case (3-1): return SPRN_PMC3;
    case (4-1): return SPRN_PMC4;
    case (5-1): return SPRN_PMC5;
    case (6-1): return SPRN_PMC6;
    case (7-1): return SPRN_PMC7;
    case (8-1): return SPRN_PMC8;
    }
}

static int write_cpu_regs(int fd, const struct perfctr_cpu_control *control)
{
    struct perfctr_cpu_reg regs[3+8];
    unsigned int nrctrs, nractrs, pmc_mask, nr_regs, i, pmc;

    nractrs = control->nractrs;
    nrctrs = nractrs + control->nrictrs;
    if (nrctrs < nractrs || nrctrs > 8) {
	errno = EINVAL;
	return -1;
    }

    if (!nrctrs)
	return 0;

    nr_regs = 0;
    pmc_mask = 0;
    for (i = 0; i < nrctrs; ++i) {
	pmc = control->pmc_map[i];
	if (pmc >= 8 || (pmc_mask & (1<<pmc))) {
	    errno = EINVAL;
	    return -1;
	}
	pmc_mask |= (1<<pmc);
	if (i >= nractrs) {
	    unsigned int j = 3 + (i - nractrs);
	    regs[j].nr = pmc_to_spr(pmc);
	    regs[j].value = control->ireset[i];
	}
    }
    regs[0].nr = SPRN_MMCR0;
    regs[0].value = control->ppc64.mmcr0;
    regs[1].nr = SPRN_MMCR1;
    regs[1].value = control->ppc64.mmcr1;
    regs[2].nr = SPRN_MMCRA;
    regs[2].value = control->ppc64.mmcra;
    nr_regs = 3 + (nrctrs - nractrs);
    show_regs(regs, nr_regs);
    return _sys_vperfctr_write(fd, PERFCTR_DOMAIN_CPU_REGS, regs, nr_regs*sizeof(regs[0]));
}

static int read_cpu_regs(int fd, struct perfctr_cpu_control *control)
{
    struct perfctr_cpu_reg regs[3+8];
    unsigned int nrctrs, nractrs, pmc_mask, nr_regs, i, pmc;
    int ret;

    nractrs = control->nractrs;
    nrctrs = nractrs + control->nrictrs;
    if (nrctrs < nractrs || nrctrs > 8) {
	errno = EINVAL;
	return -1;
    }

    if (!nrctrs)
	return 0;

    nr_regs = 0;
    pmc_mask = 0;
    for(i = 0; i < nrctrs; ++i) {
	pmc = control->pmc_map[i];
	if (pmc >= 8 || (pmc_mask & (1<<pmc))) {
	    errno = EINVAL;
	    return -1;
	}
	pmc_mask |= (1<<pmc);
	if (i >= nractrs) {
	    unsigned int j = 3 + (i - nractrs);
	    regs[j].nr = pmc_to_spr(pmc);
	}
    }
    regs[0].nr = SPRN_MMCR0;
    regs[1].nr = SPRN_MMCR1;
    regs[2].nr = SPRN_MMCRA;
    nr_regs = 3 + (nrctrs - nractrs);
    ret = read_packet(fd, PERFCTR_DOMAIN_CPU_REGS, regs, nr_regs*sizeof(regs[0]));
    if (ret < 0)
	return ret;
    show_regs(regs, nr_regs);
    for(i = 0; i < nrctrs; ++i) {
	if (i >= nractrs)
	    control->ireset[i] = regs[3 + (i - nractrs)].value;
    }
    control->ppc64.mmcr0 = regs[0].value;
    control->ppc64.mmcr1 = regs[1].value;
    control->ppc64.mmcra = regs[2].value;

    return 0;
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

    ret = write_cpu_regs(fd, &control->cpu_control);
    if (ret < 0)
	return ret;

    return _sys_vperfctr_control(fd, VPERFCTR_CONTROL_RESUME);
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

    return read_cpu_regs(fd, &control->cpu_control);
}

#define PVR_VER(pvr)	(((pvr) >>  16) & 0xFFFF)	/* Version field */
#define PVR_REV(pvr)	(((pvr) >>   0) & 0xFFFF)	/* Revison field */

/* Processor Version Numbers */
#define PV_NORTHSTAR	0x0033
#define PV_PULSAR	0x0034
#define PV_POWER4	0x0035
#define PV_ICESTAR	0x0036
#define PV_SSTAR	0x0037
#define PV_POWER4p	0x0038
#define PV_970		0x0039
#define PV_POWER5	0x003A
#define PV_POWER5p	0x003B
#define PV_970FX	0x003C
#define PV_630        	0x0040
#define PV_630p	        0x0041
#define PV_970MP	0x0044
#define PV_970GX	0x0045

static unsigned int mfpvr(void)
{
    unsigned long pvr;

    asm("mfspr	%0,%1" : "=r"(pvr) : "i"(SPRN_PVR));
    return pvr;
}

void perfctr_info_cpu_init(struct perfctr_info *info)
{
    unsigned int pvr = mfpvr();
    int cpu_type;

    switch (PVR_VER(pvr)) {
      case PV_POWER4:
	cpu_type = PERFCTR_PPC64_POWER4;
	break;
      case PV_POWER4p:
	cpu_type = PERFCTR_PPC64_POWER4p;
	break;
      case PV_970:
      case PV_970FX:
	cpu_type = PERFCTR_PPC64_970;
	break;
      case PV_970MP:
	cpu_type = PERFCTR_PPC64_970MP;
	break;
      case PV_POWER5:
      case PV_POWER5p: 
	cpu_type = PERFCTR_PPC64_POWER5;
	break;

      default:
	cpu_type = PERFCTR_PPC64_GENERIC;
	break;
    }

    info->cpu_type = cpu_type;
    return;
}

unsigned int perfctr_info_nrctrs(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_PPC64_POWER4:
      case PERFCTR_PPC64_POWER4p:
      case PERFCTR_PPC64_970:
      case PERFCTR_PPC64_970MP:
      	return 8;
      case PERFCTR_PPC64_POWER5:
      	return 6;
      default:
	return 0;
    }
}

const char *perfctr_info_cpu_name(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_PPC64_GENERIC:
	return "Generic PowerPC64";
      case PERFCTR_PPC64_POWER4:
	return "POWER4";
      case PERFCTR_PPC64_POWER4p:
	return "POWER4+";
      case PERFCTR_PPC64_970:
	return "PowerPC 970";
      case PERFCTR_PPC64_970MP:
	return "PowerPC 970MP";
      case PERFCTR_PPC64_POWER5:
	return "POWER5";
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
	printf("pmc[%u].map\t\t%u\n", i, control->pmc_map[i]);
	if( i >= nractrs )
	    printf("pmc[%u].ireset\t\t%d\n", i, control->ireset[i]);
    }
    if( control->ppc64.mmcr0 )
	printf("mmcr0\t\t\t0x%08X\n", control->ppc64.mmcr0);
    if( control->ppc64.mmcr1 )
	printf("mmcr1\t\t\t0x%016llX\n",
	       (unsigned long long)control->ppc64.mmcr1);
    if( control->ppc64.mmcra )
	printf("mmcra\t\t\t0x%08X\n", control->ppc64.mmcra);
}

