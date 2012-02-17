/* $Id: init.c,v 1.83 2007/10/06 13:02:07 mikpe Exp $
 * Performance-monitoring counters driver.
 * Top-level initialisation code.
 *
 * Copyright (C) 1999-2007  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/device.h>
#include <linux/perfctr.h>

#include "cpumask.h"
#include "virtual.h"
#include "version.h"

struct perfctr_info perfctr_info;

static ssize_t
driver_version_show(struct class *class, char *buf)
{
	return sprintf(buf, "%s\n", VERSION);
}

static ssize_t
cpu_features_show(struct class *class, char *buf)
{
	return sprintf(buf, "%#x\n", perfctr_info.cpu_features);
}

static ssize_t
cpu_khz_show(struct class *class, char *buf)
{
	return sprintf(buf, "%u\n", perfctr_info.cpu_khz);
}

static ssize_t
tsc_to_cpu_mult_show(struct class *class, char *buf)
{
	return sprintf(buf, "%u\n", perfctr_info.tsc_to_cpu_mult);
}

static ssize_t
state_user_offset_show(struct class *class, char *buf)
{
	return sprintf(buf, "%u\n", (unsigned int)offsetof(struct perfctr_cpu_state, user));
}

static ssize_t
cpus_online_show(struct class *class, char *buf)
{
	int ret = cpumask_scnprintf(buf, PAGE_SIZE-1, cpu_online_map);
	buf[ret++] = '\n';
	return ret;
}

static ssize_t
cpus_forbidden_show(struct class *class, char *buf)
{
	int ret = cpumask_scnprintf(buf, PAGE_SIZE-1, perfctr_cpus_forbidden_mask);
	buf[ret++] = '\n';
	return ret;
}

static struct class_attribute perfctr_class_attrs[] = {
	__ATTR_RO(driver_version),
	__ATTR_RO(cpu_features),
	__ATTR_RO(cpu_khz),
	__ATTR_RO(tsc_to_cpu_mult),
	__ATTR_RO(state_user_offset),
	__ATTR_RO(cpus_online),
	__ATTR_RO(cpus_forbidden),
	__ATTR_NULL
};

static struct class perfctr_class = {
	.name		= "perfctr",
	.class_attrs	= perfctr_class_attrs,
};

char *perfctr_cpu_name __initdata;

static int __init perfctr_init(void)
{
	int err;

	err = perfctr_cpu_init();
	if (err) {
		printk(KERN_INFO "perfctr: not supported by this processor\n");
		return err;
	}
	err = vperfctr_init();
	if (err)
		return err;
	err = class_register(&perfctr_class);
	if (err) {
		printk(KERN_ERR "perfctr: class initialisation failed\n");
		return err;
	}
	printk(KERN_INFO "perfctr: driver %s, cpu type %s at %u kHz\n",
	       VERSION,
	       perfctr_cpu_name,
	       perfctr_info.cpu_khz);
	return 0;
}

static void __exit perfctr_exit(void)
{
	vperfctr_exit();
	perfctr_cpu_exit();
}

module_init(perfctr_init)
module_exit(perfctr_exit)
