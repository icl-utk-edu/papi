/* $Id$
 * Global-mode performance-monitoring counters.
 *
 * Copyright (C) 2000-2002  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_GLOBAL
extern int gperfctr_init(void);
extern void gperfctr_exit(void);
#else
static inline int gperfctr_init(void) { return 0; }
static inline void gperfctr_exit(void) { }
#endif
