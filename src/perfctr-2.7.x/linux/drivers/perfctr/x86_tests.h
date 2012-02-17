/* $Id: x86_tests.h,v 1.10 2004/05/22 20:48:57 mikpe Exp $
 * Performance-monitoring counters driver.
 * Optional x86/x86_64-specific init-time tests.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */

/* 'enum perfctr_x86_tests_type' classifies CPUs according
   to relevance for perfctr_x86_init_tests(). */
enum perfctr_x86_tests_type {
	PTT_UNKNOWN,
	PTT_GENERIC,
	PTT_P5,
	PTT_P6,
	PTT_P4,
	PTT_AMD,
	PTT_WINCHIP,
	PTT_VC3,
};

extern enum perfctr_x86_tests_type perfctr_x86_tests_type;

static inline void perfctr_set_tests_type(enum perfctr_x86_tests_type t)
{
#ifdef CONFIG_PERFCTR_INIT_TESTS
	perfctr_x86_tests_type = t;
#endif
}

extern void perfctr_x86_init_tests(void);
