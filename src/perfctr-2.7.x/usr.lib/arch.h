/* $Id$
 * Architecture-specific code for performance counters library.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */
#ifndef __LIB_PERFCTR_ARCH_H
#define __LIB_PERFCTR_ARCH_H

#if defined(__i386__) || defined(__x86_64__)
#include "x86.h"
#elif defined(__powerpc64__) || defined(PPC64)
#include "ppc64.h"
#elif defined(__powerpc__)
#include "ppc.h"
#endif

#endif /* __LIB_PERFCTR_ARCH_H */
