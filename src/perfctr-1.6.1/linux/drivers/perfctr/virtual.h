/* $Id$
 * Virtual per-process performance counters.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_VIRTUAL
extern int vperfctr_attach_current(void);
extern int vperfctr_init(void);
extern void vperfctr_exit(void);
#else
static __inline__ int vperfctr_attach_current(void) { return -ENOSYS; }
static __inline__ int vperfctr_init(void) { return 0; }
static __inline__ void vperfctr_exit(void) { }
#endif
