/* $Id$
 * Performance-monitoring counters driver.
 * Top-level initialisation code and implementation of /dev/perfctr.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/miscdevice.h>
#include <linux/smp.h>
#include <linux/sched.h>
#include <linux/perfctr.h>

#include <asm/uaccess.h>

#include "compat.h"
#include "virtual.h"
#include "global.h"

#define VERSION_MAJOR	1
#define VERSION_MINOR	6
#define VERSION_MICRO	0

static struct perfctr_info perfctr_info;

int sys_perfctr_info(struct perfctr_info *argp)
{
	if( copy_to_user(argp, &perfctr_info, sizeof perfctr_info) )
		return -EFAULT;
	return 0;
}

static int dev_perfctr_ioctl(struct inode *inode, struct file *filp,
			     unsigned int cmd, unsigned long arg)
{
	switch( cmd ) {
	case PERFCTR_INFO:
		return sys_perfctr_info((struct perfctr_info*)arg);
	case VPERFCTR_ATTACH:
		return vperfctr_attach_current();
	case GPERFCTR_CONTROL:
		return gperfctr_control((struct gperfctr_control*)arg);
	case GPERFCTR_READ:
		return gperfctr_read((struct gperfctr_state*)arg);
	case GPERFCTR_STOP:
		return gperfctr_stop();
	}
	return -EINVAL;
}

#ifdef NEED_MOD_INC_OPEN	/* kernel < 2.4.0-test3 */
static int dev_perfctr_open(struct inode *inode, struct file *filp)
{
	MOD_INC_USE_COUNT;
	return 0;
}

static int dev_perfctr_release(struct inode *inode, struct file *filp)
{
	MOD_DEC_USE_COUNT;
	return 0;
}
#endif

static struct file_operations dev_perfctr_file_ops = {
	OWNER_THIS_MODULE
	MOD_INC_OPEN(dev_perfctr_open)
	MOD_DEC_RELEASE(dev_perfctr_release)
	.ioctl = dev_perfctr_ioctl,
};

static struct miscdevice dev_perfctr = {
	.minor = 182,
	.name = "perfctr",
	.fops = &dev_perfctr_file_ops,
};

#define STR2(X) #X
#define STR(X)	STR2(X)

int __init perfctr_init(void)
{
	int err;
	if( (err = perfctr_cpu_init()) != 0 ) {
		printk(KERN_INFO "perfctr: not supported by this processor\n");
		return err;
	}
	if( (err = vperfctr_init()) != 0 )
		return err;
	gperfctr_init();
	if( (err = misc_register(&dev_perfctr)) != 0 ) {
		printk(KERN_ERR "/dev/perfctr: failed to register, errno %d\n",
		       -err);
		return err;
	}
	perfctr_info.version_major = VERSION_MAJOR;
	perfctr_info.version_minor = VERSION_MINOR;
	perfctr_info.version_micro = VERSION_MICRO;
	perfctr_info.nrcpus = smp_num_cpus;
	perfctr_info.cpu_type = perfctr_cpu_type;
	perfctr_info.cpu_features = perfctr_cpu_features;
	perfctr_info.cpu_khz = perfctr_cpu_khz;
	printk(KERN_INFO "perfctr: driver version %u.%u%s\n",
	       VERSION_MAJOR, VERSION_MINOR,
	       VERSION_MICRO ? "." STR(VERSION_MICRO) : "");
	printk(KERN_INFO "perfctr: cpu type %s at %lu kHz\n",
	       perfctr_cpu_name[perfctr_cpu_type], perfctr_cpu_khz);
	return 0;
}

void __exit perfctr_exit(void)
{
	vperfctr_exit();
	perfctr_cpu_exit();
	misc_deregister(&dev_perfctr);
}

module_init(perfctr_init)
module_exit(perfctr_exit)
