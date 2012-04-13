/* $Id: misc.c,v 1.26 2006/08/27 07:34:41 mikpe Exp $
 * Miscellaneous perfctr operations.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */

#include <sys/utsname.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libperfctr.h"
#include "arch.h"

#define SYS_CLASS_PERFCTR	"/sys/class/perfctr/"

static int read_uint(const char *path, unsigned int *dst)
{
    FILE *fp;
    int ret;

    fp = fopen(path, "r");
    if (!fp)
	return -1;
    ret = fscanf(fp, "%i", (int*)dst);
    fclose(fp);
    return (ret == 1) ? 0 : -1;
}

static int read_string(const char *path, char *dst, size_t dstlen)
{
    FILE *fp;
    int len;

    fp = fopen(path, "r");
    if (!fp)
	return -1;
    len = 0;
    while (len < dstlen) {
	int ch = fgetc(fp);
	switch (ch) {
	  case '\n':
	  case '\0':
	    break;
	  default:
	    dst[len++] = ch;
	    continue;
	}
	break;
    }
    if (len < dstlen)
	dst[len] = '\0';
    fclose(fp);
    return 0;
}

static int read_cpumask(const char *path, struct perfctr_cpu_mask *mask)
{
    unsigned int nrwords;

    if (!mask)
	return 0;

    /* XXX: fake */
    nrwords = mask->nrwords;
    mask->nrwords = 1;
    if (nrwords) {
	mask->mask[0] = 0;
	return 0;
    } else {
	errno = EOVERFLOW;
	return -1;
    }
}

static int read_info(struct perfctr_info *info)
{
    int err = 0;

    if (!info)
	return 0;
#if 0
    err |= read_uint(SYS_CLASS_PERFCTR "abi_version", &info->abi_version);
#else
    info->abi_version = PERFCTR_ABI_VERSION;
#endif
    // err |= read_uint(SYS_CLASS_PERFCTR "cpu_type", &info->cpu_type);
    err |= read_uint(SYS_CLASS_PERFCTR "cpu_features", &info->cpu_features);
    err |= read_uint(SYS_CLASS_PERFCTR "cpu_khz", &info->cpu_khz);
    err |= read_uint(SYS_CLASS_PERFCTR "tsc_to_cpu_mult", &info->tsc_to_cpu_mult);
    err |= read_string(SYS_CLASS_PERFCTR "driver_version",
		       info->driver_version, sizeof(info->driver_version));
    return err;
}

int _perfctr_get_state_user_offset(void)
{
    unsigned int offset;
    
    if (read_uint(SYS_CLASS_PERFCTR "state_user_offset", &offset) != 0)
	return -1;
    return offset;
}

int _sys_perfctr_info(int fd_unused,
		      struct perfctr_info *info,
		      struct perfctr_cpu_mask *cpus,
		      struct perfctr_cpu_mask *forbidden)
{
    int err = 0;

    err |= read_info(info);
    err |= read_cpumask(SYS_CLASS_PERFCTR "cpus_online", cpus);
    err |= read_cpumask(SYS_CLASS_PERFCTR "cpus_forbidden", forbidden);
    return err;
}

int _perfctr_abi_check_fd(int fd, unsigned int user_abi_version)
{
    struct perfctr_info info;
    
    if( _sys_perfctr_info(fd, &info, NULL, NULL) < 0 ) {
	perror("perfctr_abi_check");
	return -1;
    }
    if( (info.abi_version ^ user_abi_version) & 0xFF00FF00 ) {
	fprintf(stderr, "Error: perfctr ABI major version mismatch: "
		"driver ABI 0x%08X, user ABI 0x%08X\n",
		info.abi_version, user_abi_version);
	errno = EPROTO;
	return -1;
    }
    return 0;
}

int perfctr_info(int fd, struct perfctr_info *info)
{
    if( _sys_perfctr_info(fd, info, NULL, NULL) < 0 )
	return -1;
    perfctr_info_cpu_init(info);
    return 0;
}

struct perfctr_cpus_info *perfctr_cpus_info(int fd)
{
    struct perfctr_cpu_mask dummy;
    struct perfctr_cpus_info *info;
    unsigned int cpu_mask_bytes;

    dummy.nrwords = 0;
    if( _sys_perfctr_info(fd, NULL, &dummy, NULL) >= 0 ||
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
    if( _sys_perfctr_info(fd, NULL, info->cpus, info->cpus_forbidden) < 0 ) {
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

unsigned int perfctr_linux_version_code(void)
{
    struct utsname utsname;
    unsigned int version, patchlevel, sublevel;

    if (uname(&utsname) < 0) {
	fprintf(stderr, "uname: %s\n", strerror(errno));
	return 0;
    }
    if (sscanf(utsname.release, "%u.%u.%u", &version, &patchlevel, &sublevel) != 3) {
	fprintf(stderr, "uname: unexpected release '%s'\n", utsname.release);
	return 0;
    }
    return PERFCTR_KERNEL_VERSION(version,patchlevel,sublevel);
}
