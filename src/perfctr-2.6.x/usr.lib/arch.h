/* $Id: arch.h,v 1.1.2.1 2007/02/11 20:15:03 mikpe Exp $
 * Architecture-specific code for performance counters library.
 *
 * Copyright (C) 2004-2007  Mikael Pettersson
 */
#ifndef __LIB_PERFCTR_ARCH_H
#define __LIB_PERFCTR_ARCH_H

#if defined(__i386__) || defined(__x86_64__)
#include "x86.h"
#elif defined(__powerpc__)
#include "ppc.h"
#elif defined(__arm__)
#include "arm.h"
#endif

#endif /* __LIB_PERFCTR_ARCH_H */
