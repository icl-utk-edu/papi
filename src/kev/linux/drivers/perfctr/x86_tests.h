/* $Id$
 * Performance-monitoring counters driver.
 * Optional x86-specific init-time tests.
 *
 * Copyright (C) 1999-2002  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_INIT_TESTS
extern void perfctr_p5_init_tests(void);
extern void perfctr_p6_init_tests(void);
extern void perfctr_k7_init_tests(void);
extern void perfctr_c6_init_tests(void);
extern void perfctr_vc3_init_tests(void);
extern void perfctr_p4_init_tests(void);
extern void perfctr_generic_init_tests(void);
#else
#define perfctr_p5_init_tests()
#define perfctr_p6_init_tests()
#define perfctr_k7_init_tests()
#define perfctr_c6_init_tests()
#define perfctr_vc3_init_tests()
#define perfctr_p4_init_tests()
#define perfctr_generic_init_tests()
#endif
#define perfctr_mii_init_tests()	perfctr_p5_init_tests()
