/* $Id$
 * Virtual per-process performance counters.
 *
 * Copyright (C) 1999-2003  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_VIRTUAL
extern int vperfctr_attach(int, int);
extern int vperfctr_init(void);
extern void vperfctr_exit(void);
#else
static inline int vperfctr_attach(int tid, int creat) { return -EINVAL; }
static inline int vperfctr_init(void) { return 0; }
static inline void vperfctr_exit(void) { }
#endif
