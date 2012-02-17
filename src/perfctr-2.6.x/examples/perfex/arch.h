/* $Id: arch.h,v 1.2.2.3 2010/06/08 20:48:55 mikpe Exp $
 * Architecture-specific support code.
 *
 * Copyright (C) 2004-2010  Mikael Pettersson
 */

#define ARRAY_SIZE(x)	(sizeof(x) / sizeof((x)[0]))

extern void do_print(FILE *resfile,
		     const struct perfctr_info *info,
		     const struct perfctr_cpu_control *cpu_control,
		     const struct perfctr_sum_ctrs *sum);

extern void do_arch_usage(void);

/* Hack while phasing out an old number parsing bug. */
extern unsigned long my_strtoul(const char *nptr, char **endptr);

extern unsigned int do_event_spec(unsigned int n,
				  const char *arg,
				  struct perfctr_cpu_control *cpu_control);

extern int do_arch_option(int ch,
			  const char *arg,
			  struct perfctr_cpu_control *cpu_control);

#if defined(__i386__) || defined(__x86_64__)
#include "x86.h"
#elif defined(__powerpc__)
#include "ppc.h"
#elif defined(__arm__)
#include "arm.h"
#endif
