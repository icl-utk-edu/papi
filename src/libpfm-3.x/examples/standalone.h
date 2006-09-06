/*
 * standalone.h - common header file for all the *_standalone.c examples
 *
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 */
#ifndef __STANDALONE_H__
#define __STANDALONE_H__ 1

#ifdef __x86_64__
#define __NR_pfm_create_context		279
#elif defined(__i386__)
#define __NR_pfm_create_context		317
#elif defined( __ia64__)
#define __NR_pfm_create_context		1303
#elif defined(__mips__)
/* Add 12 */
#if (_MIPS_SIM == _ABIN32) || (_MIPS_SIM == _MIPS_SIM_NABI32)
#define __NR_Linux 6000
#define __NR_pfm_create_context		__NR_Linux+250
#elif (_MIPS_SIM == _ABI32) || (_MIPS_SIM == _MIPS_SIM_ABI32)
#define __NR_Linux 4000
#define __NR_pfm_create_context		__NR_Linux+287
#elif (_MIPS_SIM == _ABI64) || (_MIPS_SIM == _MIPS_SIM_ABI64)
#define __NR_Linux 5000
#define __NR_pfm_create_context		__NR_Linux+246
#endif
#else
#error "you need to define based syscall number"
#endif

#define __NR_pfm_write_pmcs		(__NR_pfm_create_context+1)
#define __NR_pfm_write_pmds		(__NR_pfm_create_context+2)
#define __NR_pfm_read_pmds		(__NR_pfm_create_context+3)
#define __NR_pfm_load_context		(__NR_pfm_create_context+4)
#define __NR_pfm_start			(__NR_pfm_create_context+5)
#define __NR_pfm_stop			(__NR_pfm_create_context+6)
#define __NR_pfm_restart		(__NR_pfm_create_context+7)
#define __NR_pfm_create_evtsets		(__NR_pfm_create_context+8)
#define __NR_pfm_getinfo_evtsets	(__NR_pfm_create_context+9)
#define __NR_pfm_delete_evtsets		(__NR_pfm_create_context+10)
#define __NR_pfm_unload_context		(__NR_pfm_create_context+11)

inline int
pfm_create_context(pfarg_ctx_t *ctx, void *smpl_arg, size_t smpl_size)
{
  return syscall(__NR_pfm_create_context, ctx, smpl_arg, smpl_size);
}

inline int
pfm_write_pmcs(int ctx_fd, pfarg_pmc_t *pmcs, int count)
{
	return syscall(__NR_pfm_write_pmcs, ctx_fd, pmcs, count);
}

inline int
pfm_write_pmds(int ctx_fd, pfarg_pmd_t *pmds, int count)
{
	return syscall(__NR_pfm_write_pmds, ctx_fd, pmds, count);
}

inline int
pfm_read_pmds(int ctx_fd, pfarg_pmd_t *pmds, int count)
{
	return syscall(__NR_pfm_read_pmds, ctx_fd, pmds, count);
}

inline int
pfm_load_context(int ctx_fd, pfarg_load_t *load)
{
	return syscall(__NR_pfm_load_context, ctx_fd, load);
}

inline int
pfm_start(int ctx_fd, pfarg_start_t *start)
{
	return syscall(__NR_pfm_start, ctx_fd, start);
}

inline int
pfm_stop(int ctx_fd)
{
	return syscall(__NR_pfm_stop, ctx_fd);
}

inline int
pfm_restart(int ctx_fd)
{
	return syscall(__NR_pfm_restart, ctx_fd);
}

inline int
pfm_create_evtsets(int ctx_fd, pfarg_setdesc_t *setd, int count)
{
	return syscall(__NR_pfm_create_evtsets, ctx_fd, setd, count);
}

inline int
pfm_delete_evtsets(int ctx_fd, pfarg_setdesc_t *setd, int count)
{
	return syscall(__NR_pfm_delete_evtsets, ctx_fd, setd, count);
}

inline int
pfm_getinfo_evtsets(int ctx_fd, pfarg_setinfo_t *info, int count)
{
	return syscall(__NR_pfm_getinfo_evtsets, ctx_fd, info, count);
}

inline int
pfm_unload_context(int ctx_fd)
{
	return syscall(__NR_pfm_unload_context, ctx_fd);
}

#define STANDALONE_MIPS20K	0
#define STANDALONE_MIPS5K	1
#define STANDALONE_P4		2

static inline int cpu_detect(void)
{
	pfarg_ctx_t ctx;
	FILE *fp;	
	char buffer[256];
	char *p;
	int ret;

	memset(buffer, 0, sizeof(buffer));

	/*
	 * check if PMU mapping exist. that is a sign that there is
	 * a PMU description module loaded, i.e., the PMU is supported by
	 * the kernel.
	 *
	 * If the description does not exists, create a context to trigger
	 * PMu description module auto-insertion. If that works then we
	 * can determine PMU model. If that fails, then PMU is not
	 * automatically supported. The system administrator may be able
	 * to manually insert the correct PMU description module.
	 */
	ret = access("/sys/kernel/perfmon/pmu_desc", F_OK);
	if (ret == -1) {
		memset(&ctx, 0, sizeof(ctx));
		ret = pfm_create_context(&ctx, NULL, 0);
		if (ret == -1)
			return -1;
		close(ctx.ctx_fd);
	}
	fp = fopen("/sys/kernel/perfmon/pmu_desc/model", "r");
	if (fp == NULL) return -1;

	p  = fgets(buffer, 255, fp);

	fclose(fp);

	if (p == NULL)
		return -1;

	/* remove trailing \n */
	buffer[strlen(buffer)-1] = '\0';

	if (!strcmp(buffer, "MIPS20K"))
		return STANDALONE_MIPS20K;
	if (!strcmp(buffer, "MIPS5K"))
		return STANDALONE_MIPS5K;
	if (!strcmp(buffer, "Intel P4/Xeon/EM64T"))
		return STANDALONE_P4;

	return -1;
}

#endif /* __STANDALONE_H__ */
