/* $Id: virtual.h,v 1.13 2004/05/31 18:18:55 mikpe Exp $
 * Virtual per-process performance counters.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_VIRTUAL
extern int vperfctr_init(void);
extern void vperfctr_exit(void);
#else
static inline int vperfctr_init(void) { return 0; }
static inline void vperfctr_exit(void) { }
#endif
