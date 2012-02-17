/* $Id: x86.h,v 1.1 2004/01/26 13:21:41 mikpe Exp $
 * x86-specific code for performance counters library.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */
#ifndef __LIB_PERFCTR_X86_H
#define __LIB_PERFCTR_X86_H

#define PAGE_SIZE	4096

#define rdtscl(low)	\
	__asm__ __volatile__("rdtsc" : "=a"(low) : : "edx")
#define rdpmcl(ctr,low)	\
	__asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")

#if defined(__x86_64__)
#define vperfctr_has_rdpmc(vperfctr)	(1)
#else
#define vperfctr_has_rdpmc(vperfctr)	((vperfctr)->have_rdpmc)
#endif

#define perfctr_info_cpu_init(info)	do{}while(0)

#endif /* __LIB_PERFCTR_X86_H */
