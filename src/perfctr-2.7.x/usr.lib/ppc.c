/* $Id: ppc.c,v 1.20 2007/10/06 13:02:07 mikpe Exp $
 * PPC32-specific perfctr library procedures.
 *
 * Copyright (C) 2004-2007  Mikael Pettersson
 */
#include <errno.h>
#include <asm/unistd.h>
#include <stdio.h>
#include <string.h>	/* memset() */
#include "libperfctr.h"
#include "ppc.h"

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
#define SPRN_MMCR0      0x3B8   /* Monitor Mode Control Register 0 (604 and up) */
#define SPRN_MMCR1      0x3BC   /* Monitor Mode Control Register 1 (604e and up) */
#define SPRN_MMCR2      0x3B0   /* Monitor Mode Control Register 2 (7400 and up) */
#define SPRN_PMC1       0x3B9   /* Performance Counter Register 1 (604 and up) */
#define SPRN_PMC2       0x3BA   /* Performance Counter Register 2 (604 and up) */
#define SPRN_PMC3       0x3BD   /* Performance Counter Register 3 (604e and up) */
#define SPRN_PMC4       0x3BE   /* Performance Counter Register 4 (604e and up) */
#define SPRN_PMC5       0x3B1   /* Performance Counter Register 5 (7450 and up) */
#define SPRN_PMC6       0x3B2   /* Performance Counter Register 6 (7450 and up) */

#define MMCR0_PMC1SEL           0x00001FC0 /* PMC1 event selector, 7 bits. */
#define MMCR0_PMC2SEL           0x0000003F /* PMC2 event selector, 6 bits. */

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
        }
}

static int write_cpu_regs(int fd, const struct perfctr_cpu_control *control)
{
    struct perfctr_cpu_reg regs[3+6];
    unsigned int evntsel[6];
    unsigned int nrctrs, nractrs, pmc_mask, nr_regs, i, pmc;

    nractrs = control->nractrs;
    nrctrs = nractrs + control->nrictrs;
    if (nrctrs < nractrs || nrctrs > 6) {
	errno = EINVAL;
	return -1;
    }

    if (!nrctrs)
	return 0;

    nr_regs = 0;
    pmc_mask = 0;
    memset(evntsel, 0, sizeof evntsel);
    for(i = 0; i < nrctrs; ++i) {
	pmc = control->pmc_map[i];
	if (pmc >= 6 || (pmc_mask & (1<<pmc))) {
	    errno = EINVAL;
	    return -1;
	}
	pmc_mask |= (1<<pmc);
	evntsel[pmc] = control->evntsel[i];
	if (i >= nractrs) {
	    unsigned int j = 3 + (i - nractrs);
	    regs[j].nr = pmc_to_spr(pmc);
	    regs[j].value = control->ireset[i];
	}
    }
    regs[0].nr = SPRN_MMCR0;
    regs[0].value = (control->ppc.mmcr0
		     | (evntsel[1-1] << (31-25))
		     | (evntsel[2-1] << (31-31)));
    regs[1].nr = SPRN_MMCR1;
    regs[1].value = ((  evntsel[3-1] << (31-4))
		     | (evntsel[4-1] << (31-9))
		     | (evntsel[5-1] << (31-14))
		     | (evntsel[6-1] << (31-20)));
    regs[2].nr = SPRN_MMCR2;
    regs[2].value = control->ppc.mmcr2;
    nr_regs = 3 + (nrctrs - nractrs);
    show_regs(regs, nr_regs);
    return _sys_vperfctr_write(fd, PERFCTR_DOMAIN_CPU_REGS, regs, nr_regs*sizeof(regs[0]));
}

static int read_cpu_regs(int fd, struct perfctr_cpu_control *control)
{
    struct perfctr_cpu_reg regs[3+6];
    unsigned int evntsel[6];
    unsigned int nrctrs, nractrs, pmc_mask, nr_regs, i, pmc;
    int ret;

    nractrs = control->nractrs;
    nrctrs = nractrs + control->nrictrs;
    if (nrctrs < nractrs || nrctrs > 6) {
	errno = EINVAL;
	return -1;
    }

    if (!nrctrs)
	return 0;

    nr_regs = 0;
    pmc_mask = 0;
    for(i = 0; i < nrctrs; ++i) {
	pmc = control->pmc_map[i];
	if (pmc >= 6 || (pmc_mask & (1<<pmc))) {
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
    regs[2].nr = SPRN_MMCR2;
    nr_regs = 3 + (nrctrs - nractrs);
    ret = read_packet(fd, PERFCTR_DOMAIN_CPU_REGS, regs, nr_regs*sizeof(regs[0]));
    if (ret < 0)
	return ret;
    show_regs(regs, nr_regs);
    evntsel[1-1] = (regs[0].value >> (31-25)) & 0x7F;
    evntsel[2-1] = (regs[0].value >> (31-31)) & 0x3F;
    evntsel[3-1] = (regs[1].value >> (31- 4)) & 0x1F;
    evntsel[4-1] = (regs[1].value >> (31- 9)) & 0x1F;
    evntsel[5-1] = (regs[1].value >> (31-14)) & 0x1F;
    evntsel[6-1] = (regs[1].value >> (31-20)) & 0x3F;
    for(i = 0; i < nrctrs; ++i) {
	pmc = control->pmc_map[i];
	control->evntsel[i] = evntsel[pmc];
	if (i >= nractrs)
	    control->ireset[i] = regs[3 + (i - nractrs)].value;
    }
    control->ppc.mmcr0 = regs[0].value & ~(MMCR0_PMC1SEL | MMCR0_PMC2SEL);
    control->ppc.mmcr2 = regs[2].value;

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

void perfctr_info_cpu_init(struct perfctr_info *info)
{
    unsigned int pvr = mfspr(SPRN_PVR); /* trapped & emulated by the kernel */
    unsigned int cpu_type;

    switch( PVR_VER(pvr) ) {
      case 0x0004: /* 604 */
	cpu_type = PERFCTR_PPC_604;
	break;
      case 0x0009: /* 604e */
      case 0x000A: /* 604ev */
	cpu_type = PERFCTR_PPC_604e;
	break;
      case 0x0008: /* 750/740 */
      case 0x7000: case 0x7001: /* 750FX */
      case 0x7002: /* 750GX */
	cpu_type = PERFCTR_PPC_750;
	break;
      case 0x000C: /* 7400 */
      case 0x800C: /* 7410 */
	cpu_type = PERFCTR_PPC_7400;
	break;
      case 0x8000: /* 7451/7441 */
      case 0x8001: /* 7455/7445 */
      case 0x8002: /* 7457/7447 */
      case 0x8003: /* 7447A */
      case 0x8004: /* 7448 */
	cpu_type = PERFCTR_PPC_7450;
	break;
      default:
	cpu_type = PERFCTR_PPC_GENERIC;
    }
    info->cpu_type = cpu_type;
}

unsigned int perfctr_info_nrctrs(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_PPC_604:
	return 2;
      case PERFCTR_PPC_604e:
      case PERFCTR_PPC_750:
      case PERFCTR_PPC_7400:
	return 4;
      case PERFCTR_PPC_7450:
	return 6;
      default:
	return 0;
    }
}

const char *perfctr_info_cpu_name(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_PPC_GENERIC:
	return "Generic PowerPC with TB";
      case PERFCTR_PPC_604:
	return "PowerPC 604";
      case PERFCTR_PPC_604e:
	return "PowerPC 604e";
      case PERFCTR_PPC_750:
	return "PowerPC 750";
      case PERFCTR_PPC_7400:
	return "PowerPC 7400";
      case PERFCTR_PPC_7450:
	return "PowerPC 7450";
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
	printf("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
        printf("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
	if( i >= nractrs )
	    printf("ireset[%u]\t\t%d\n", i, control->ireset[i]);
    }
    if( control->ppc.mmcr0 )
	printf("mmcr0\t\t\t0x%08X\n", control->ppc.mmcr0);
    if( control->ppc.mmcr2 )
	printf("mmcr2\t\t\t0x%08X\n", control->ppc.mmcr2);
}
