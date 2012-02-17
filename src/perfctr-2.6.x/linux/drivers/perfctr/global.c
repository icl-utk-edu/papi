/* $Id: global.c,v 1.38.2.7 2009/06/11 08:11:31 mikpe Exp $
 * Global-mode performance-monitoring counters via /dev/perfctr.
 *
 * Copyright (C) 2000-2006, 2008, 2009  Mikael Pettersson
 *
 * XXX: Doesn't do any authentication yet. Should we limit control
 * to root, or base it on having write access to /dev/perfctr?
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/perfctr.h>

#include <asm/uaccess.h>

#include "compat.h"
#include "global.h"
#include "marshal.h"

static const char this_service[] = __FILE__;
static int hardware_is_ours = 0;
static struct timer_list sampling_timer;
static DEFINE_MUTEX(control_mutex);
static unsigned int nr_active_cpus = 0;

struct gperfctr {
	struct perfctr_cpu_state cpu_state;
	spinlock_t lock;
} ____cacheline_aligned;

static struct gperfctr per_cpu_gperfctr[NR_CPUS] __cacheline_aligned;

static int reserve_hardware(void)
{
	const char *other;

	if (hardware_is_ours)
		return 0;
	other = perfctr_cpu_reserve(this_service);
	if (other) {
		printk(KERN_ERR __FILE__ ":%s: failed because hardware is taken by '%s'\n",
		       __FUNCTION__, other);
		return -EBUSY;
	}
	hardware_is_ours = 1;
	__module_get(THIS_MODULE);
	return 0;
}

static void release_hardware(void)
{
	int i;

	nr_active_cpus = 0;
	if (hardware_is_ours) {
		hardware_is_ours = 0;
		if (sampling_timer.data)
			del_timer(&sampling_timer);
		sampling_timer.data = 0;
		perfctr_cpu_release(this_service);
		module_put(THIS_MODULE);
		for(i = 0; i < NR_CPUS; ++i)
			per_cpu_gperfctr[i].cpu_state.cstatus = 0;
	}
}

static void sample_this_cpu(void *unused)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	struct gperfctr *perfctr;

	perfctr = &per_cpu_gperfctr[smp_processor_id()];
	if (!perfctr_cstatus_enabled(perfctr->cpu_state.cstatus))
		return;
	spin_lock(&perfctr->lock);
	perfctr_cpu_sample(&perfctr->cpu_state);
	spin_unlock(&perfctr->lock);
}

static void sample_all_cpus(void)
{
	on_each_cpu(sample_this_cpu, NULL, 1);
}

static void do_sample_one_cpu(void *info)
{
	unsigned int cpu = (unsigned long)info;

	if (cpu == smp_processor_id())
		sample_this_cpu(NULL);
}

static void sample_one_cpu(unsigned int cpu)
{
	on_each_cpu(do_sample_one_cpu, (void*)(unsigned long)cpu, 1);
}

static void sampling_timer_function(unsigned long interval)
{	
	sample_all_cpus();
	sampling_timer.expires = jiffies + interval;
	add_timer(&sampling_timer);
}

static unsigned long usectojiffies(unsigned long usec)
{
	usec += 1000000 / HZ - 1;
	usec /= 1000000 / HZ;
	return usec;
}

static void start_sampling_timer(unsigned long interval_usec)
{
	if (interval_usec > 0) {
		unsigned long interval = usectojiffies(interval_usec);
		init_timer(&sampling_timer);
		sampling_timer.function = sampling_timer_function;
		sampling_timer.data = interval;
		sampling_timer.expires = jiffies + interval;
		add_timer(&sampling_timer);
	}
}

static void start_this_cpu(void *unused)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	struct gperfctr *perfctr;

	perfctr = &per_cpu_gperfctr[smp_processor_id()];
	if (perfctr_cstatus_enabled(perfctr->cpu_state.cstatus))
		perfctr_cpu_resume(&perfctr->cpu_state);
}

static void start_all_cpus(void)
{
	on_each_cpu(start_this_cpu, NULL, 1);
}

static int gperfctr_control(struct perfctr_struct_buf *argp)
{
	int ret;
	struct gperfctr *perfctr;
	struct gperfctr_cpu_control cpu_control;

	ret = perfctr_copy_from_user(&cpu_control, argp, &gperfctr_cpu_control_sdesc);
	if (ret)
		return ret;
	if (cpu_control.cpu >= NR_CPUS ||
	    !cpu_online(cpu_control.cpu) ||
	    perfctr_cpu_is_forbidden(cpu_control.cpu))
		return -EINVAL;
	/* we don't permit i-mode counters */
	if (cpu_control.cpu_control.nrictrs != 0)
		return -EPERM;
	mutex_lock(&control_mutex);
	ret = -EBUSY;
	if (hardware_is_ours)
		goto out_unlock;	/* you have to stop them first */
	perfctr = &per_cpu_gperfctr[cpu_control.cpu];
	spin_lock(&perfctr->lock);
	perfctr->cpu_state.tsc_start = 0;
	perfctr->cpu_state.tsc_sum = 0;
	memset(&perfctr->cpu_state.pmc, 0, sizeof perfctr->cpu_state.pmc);
	perfctr->cpu_state.control = cpu_control.cpu_control;
	ret = perfctr_cpu_update_control(&perfctr->cpu_state, NULL);
	spin_unlock(&perfctr->lock);
	if (ret < 0)
		goto out_unlock;
	if (perfctr_cstatus_enabled(perfctr->cpu_state.cstatus))
		++nr_active_cpus;
	ret = nr_active_cpus;
 out_unlock:
	mutex_unlock(&control_mutex);
	return ret;
}

static int gperfctr_start(unsigned int interval_usec)
{
	int ret;

	if (interval_usec && interval_usec < 10000)
		return -EINVAL;
	mutex_lock(&control_mutex);
	ret = nr_active_cpus;
	if (ret > 0) {
		if (reserve_hardware() < 0) {
			ret = -EBUSY;
		} else {
			start_all_cpus();
			start_sampling_timer(interval_usec);
		}
	}
	mutex_unlock(&control_mutex);
	return ret;
}

static int gperfctr_stop(void)
{
	mutex_lock(&control_mutex);
	release_hardware();
	mutex_unlock(&control_mutex);
	return 0;
}

static int gperfctr_read(struct perfctr_struct_buf *argp)
{
	struct gperfctr *perfctr;
	struct gperfctr_cpu_state state;
	int err;

	err = perfctr_copy_from_user(&state, argp, &gperfctr_cpu_state_only_cpu_sdesc);
	if (err)
		return err;
	if (state.cpu >= NR_CPUS || !cpu_online(state.cpu))
		return -EINVAL;
	if (!sampling_timer.data)
		sample_one_cpu(state.cpu);
	perfctr = &per_cpu_gperfctr[state.cpu];
	spin_lock(&perfctr->lock);
	state.cpu_control = perfctr->cpu_state.control;
	//state.sum = perfctr->cpu_state.sum;
	{
		int j;
		state.sum.tsc = perfctr->cpu_state.tsc_sum;
		for(j = 0; j < ARRAY_SIZE(state.sum.pmc); ++j)
			state.sum.pmc[j] = perfctr->cpu_state.pmc[j].sum;
	}
	spin_unlock(&perfctr->lock);
	return perfctr_copy_to_user(argp, &state, &gperfctr_cpu_state_sdesc);
}

int gperfctr_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	switch (cmd) {
	case GPERFCTR_CONTROL:
		return gperfctr_control((struct perfctr_struct_buf*)arg);
	case GPERFCTR_READ:
		return gperfctr_read((struct perfctr_struct_buf*)arg);
	case GPERFCTR_STOP:
		return gperfctr_stop();
	case GPERFCTR_START:
		return gperfctr_start(arg);
	}
	return -EINVAL;
}

void __init gperfctr_init(void)
{
	int i;

	for(i = 0; i < NR_CPUS; ++i)
		per_cpu_gperfctr[i].lock = SPIN_LOCK_UNLOCKED;
}
