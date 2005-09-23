/* $Id$
 * x86-specific code for performance counters library.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */
#ifndef __LIB_PERFCTR_X86_H
#define __LIB_PERFCTR_X86_H

#define rdtscl(low)	\
	__asm__ __volatile__("rdtsc" : "=a"(low) : : "edx")
#define rdpmcl(ctr,low)	\
	__asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")

#if defined(__x86_64__)
#define vperfctr_has_rdpmc(vperfctr)	(1)
#else
#define vperfctr_has_rdpmc(vperfctr)	((vperfctr)->have_rdpmc)
#endif

extern void perfctr_info_cpu_init(struct perfctr_info*);

#endif /* __LIB_PERFCTR_X86_H */
