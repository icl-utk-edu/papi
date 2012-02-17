/* $Id: ppc_tests.h,v 1.1.2.1 2004/06/21 22:33:35 mikpe Exp $
 * Performance-monitoring counters driver.
 * Optional PPC32-specific init-time tests.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_INIT_TESTS
extern void perfctr_ppc_init_tests(int have_mmcr1);
#else
static inline void perfctr_ppc_init_tests(int have_mmcr1) { }
#endif
