/* $Id$
 * Global-mode performance-monitoring counters.
 *
 * Copyright (C) 2000-2003  Mikael Pettersson
 */

#ifdef CONFIG_PERFCTR_GLOBAL
extern int gperfctr_ioctl(struct inode*, struct file*, unsigned int, unsigned long);
extern void gperfctr_init(void);
#else
extern int gperfctr_ioctl(struct inode *inode, struct file *filp,
			  unsigned int cmd, unsigned long arg)
{
	return -EINVAL;
}
static inline void gperfctr_init(void) { }
#endif
