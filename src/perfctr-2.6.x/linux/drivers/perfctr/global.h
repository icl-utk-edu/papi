/* $Id: global.h,v 1.7.2.1 2005/01/22 14:04:03 mikpe Exp $
 * Global-mode performance-monitoring counters.
 *
 * Copyright (C) 2000-2005  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_GLOBAL
extern int gperfctr_ioctl(struct file*, unsigned int, unsigned long);
extern void gperfctr_init(void);
#else
extern int gperfctr_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	return -EINVAL;
}
static inline void gperfctr_init(void) { }
#endif
