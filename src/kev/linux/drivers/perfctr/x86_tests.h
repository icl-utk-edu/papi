/* $Id$
 * Performance-monitoring counters driver.
 * Optional x86/x86_64-specific init-time tests.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_INIT_TESTS
extern void perfctr_x86_init_tests(void);
#else
#define perfctr_x86_init_tests()
#endif
