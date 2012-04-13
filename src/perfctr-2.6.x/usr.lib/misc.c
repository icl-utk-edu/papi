/* $Id: misc.c,v 1.20.2.1 2005/12/22 22:44:49 mikpe Exp $
 * Miscellaneous perfctr operations.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "libperfctr.h"
#include "marshal.h"
#include "arch.h"

int _perfctr_abi_check_fd(int fd, unsigned int user_abi_version)
{
    unsigned int driver_abi_version;
    
    if( ioctl(fd, PERFCTR_ABI, &driver_abi_version) < 0 ) {
	perror("perfctr_abi_check");
	return -1;
    }
    if( (driver_abi_version ^ user_abi_version) & 0xFF00FF00 ) {
	fprintf(stderr, "Error: perfctr ABI major version mismatch: "
		"driver ABI 0x%08X, user ABI 0x%08X\n",
		driver_abi_version, user_abi_version);
	errno = EPROTO;
	return -1;
    }
    return 0;
}

int perfctr_info(int fd, struct perfctr_info *info)
{
    int err = perfctr_ioctl_r(fd, PERFCTR_INFO, info, &perfctr_info_sdesc);
    if( err < 0 )
	return err;
    perfctr_info_cpu_init(info);
    return 0;
}

int perfctr_get_info(struct perfctr_info *info)
{
    int fd, ret;

    fd = open("/dev/perfctr", O_RDONLY);
    if (fd < 0)
	return -1;
    ret = perfctr_info(fd, info);
    close(fd);
    return ret;
}

struct perfctr_cpus_info *perfctr_cpus_info(int fd)
{
    struct perfctr_cpu_mask dummy;
    struct perfctr_cpus_info *info;
    unsigned int cpu_mask_bytes;

    dummy.nrwords = 0;
    if( ioctl(fd, PERFCTR_CPUS, &dummy) >= 0 ||
	errno != EOVERFLOW ||
	dummy.nrwords == 0 ) {
	perror("PERFCTR_CPUS");
	return NULL;
    }
    cpu_mask_bytes = offsetof(struct perfctr_cpu_mask, mask[dummy.nrwords]);
    info = malloc(sizeof(struct perfctr_cpus_info) + 2*cpu_mask_bytes);
    if( !info ) {
	perror("malloc");
	return NULL;
    }
    info->cpus = (struct perfctr_cpu_mask*)(info + 1);
    info->cpus->nrwords = dummy.nrwords;
    info->cpus_forbidden = (struct perfctr_cpu_mask*)((char*)(info + 1) + cpu_mask_bytes);
    info->cpus_forbidden->nrwords = dummy.nrwords;
    if( ioctl(fd, PERFCTR_CPUS, info->cpus) < 0 ||
	ioctl(fd, PERFCTR_CPUS_FORBIDDEN, info->cpus_forbidden) < 0 ) {
	perror("PERFCTR_CPUS");
	free(info);
	return NULL;
    }
    return info;
}

void perfctr_info_print(const struct perfctr_info *info)
{
    static const char * const features[] = { "rdpmc", "rdtsc", "pcint" };
    int fi, comma;

    printf("abi_version\t\t0x%08X\n", info->abi_version);
    printf("driver_version\t\t%s\n", info->driver_version);
    printf("cpu_type\t\t%u (%s)\n", info->cpu_type, perfctr_info_cpu_name(info));
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
    printf("cpu_khz\t\t\t%u\n", info->cpu_khz);
    printf("tsc_to_cpu_mult\t\t%u%s\n",
	   info->tsc_to_cpu_mult,
	   info->tsc_to_cpu_mult ? "" : " (unspecified, assume 1)");
    printf("cpu_nrctrs\t\t%u\n", perfctr_info_nrctrs(info));
}

static void print_cpus(const struct perfctr_cpu_mask *cpus)
{
    unsigned int nrcpus, nr, i, cpumask, bitmask;

    printf("[");
    nrcpus = 0;
    for(i = 0; i < cpus->nrwords; ++i) {
	cpumask = cpus->mask[i];
	nr = i * 8 * sizeof(int);
	for(bitmask = 1; cpumask != 0; ++nr, bitmask <<= 1) {
	    if( cpumask & bitmask ) {
		cpumask &= ~bitmask;
		if( nrcpus )
		    printf(",");
		++nrcpus;
		printf("%u", nr);
	    }
	}
    }
    printf("], total: %u\n", nrcpus);
}

void perfctr_cpus_info_print(const struct perfctr_cpus_info *info)
{
    printf("cpus\t\t\t"); print_cpus(info->cpus);
    printf("cpus_forbidden\t\t"); print_cpus(info->cpus_forbidden);
}
