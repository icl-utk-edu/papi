/* $Id$
 * Performance-monitoring counters driver.
 * Optional x86_64-specific init-time tests.
 *
 * Copyright (C) 2003  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_INIT_TESTS
extern void perfctr_k8_init_tests(void);
extern void perfctr_generic_init_tests(void);
#else
#define perfctr_k8_init_tests()
#define perfctr_generic_init_tests()
#endif
