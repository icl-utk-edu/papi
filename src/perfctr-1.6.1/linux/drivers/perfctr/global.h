/* $Id$
 * Global-mode performance-monitoring counters.
 *
 * Copyright (C) 2000  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_GLOBAL

extern int gperfctr_control(struct gperfctr_control *arg);
extern int gperfctr_read(struct gperfctr_state *arg);
extern int gperfctr_stop(void);
extern void gperfctr_init(void);

#else

static __inline__ int gperfctr_control(struct gperfctr_control *arg)
{
	return -ENOSYS;
}
static __inline__ int gperfctr_read(struct gperfctr_state *arg)
{
	return -ENOSYS;
}
static __inline__ int gperfctr_stop(void)
{
	return -ENOSYS;
}
static __inline__ void gperfctr_init(void) { }

#endif
