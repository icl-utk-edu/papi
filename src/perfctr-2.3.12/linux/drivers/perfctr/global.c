/* $Id$
 * Global-mode performance-monitoring counters via /dev/perfctr.
 *
 * Copyright (C) 2000-2001  Mikael Pettersson
 *
 * XXX: Doesn't do any authentication yet. Should we limit control
 * to root, or base it on having write access to /dev/perfctr?
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/perfctr.h>

#include <asm/uaccess.h>

#include "compat.h"
#include "global.h"

static const char this_service[] = __FILE__;
static int hardware_is_ours = 0;
static struct timer_list sampling_timer;

static unsigned int nr_active_cpus = 0;

struct gperfctr {
	struct perfctr_cpu_state cpu_state;
	spinlock_t lock;
} __attribute__((__aligned__(SMP_CACHE_BYTES)));

static struct gperfctr per_cpu_gperfctr[NR_CPUS] __cacheline_aligned;

static int reserve_hardware(void)
{
	const char *other;

	if( hardware_is_ours )
		return 0;
	other = perfctr_cpu_reserve(this_service);
	if( other ) {
		printk(KERN_ERR __FILE__ ": " __FUNCTION__
		       ": failed because hardware is taken by '%s'\n",
		       other);
		return -EBUSY;
	}
	hardware_is_ours = 1;
	MOD_INC_USE_COUNT;
	return 0;
}

static void release_hardware(void)
{
	nr_active_cpus = 0;
	if( hardware_is_ours ) {
		hardware_is_ours = 0;
		del_timer(&sampling_timer);
		sampling_timer.data = 0;
		perfctr_cpu_release(this_service);
		MOD_DEC_USE_COUNT;
	}
}

static void sample_this_cpu(void *unused)
{
	struct gperfctr *perfctr;

	perfctr = &per_cpu_gperfctr[smp_processor_id()];
	if( !perfctr_cstatus_enabled(perfctr->cpu_state.cstatus) )
		return;
	spin_lock(&perfctr->lock);
	perfctr_cpu_sample(&perfctr->cpu_state);
	spin_unlock(&perfctr->lock);
}

static void sample_all_cpus(void)
{
	smp_call_function(sample_this_cpu, NULL, 1, 1);
	sample_this_cpu(NULL);
}

static void sampling_timer_function(unsigned long interval)
{	
	sample_all_cpus();
	sampling_timer.expires = jiffies + interval;
	add_timer(&sampling_timer);
}

static unsigned long usectojiffies(unsigned long usec)
{
	/* based on kernel/itimer.c:tvtojiffies() */
	usec += 1000000 / HZ - 1;
	usec /= 1000000 / HZ;
	return usec;
}

static void start_sampling_timer(unsigned long interval_usec)
{
	if( interval_usec > 0 ) {
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
	struct gperfctr *perfctr;

	perfctr = &per_cpu_gperfctr[smp_processor_id()];
	if( perfctr_cstatus_enabled(perfctr->cpu_state.cstatus) )
		perfctr_cpu_resume(&perfctr->cpu_state);
}

static void start_all_cpus(void)
{
	smp_call_function(start_this_cpu, NULL, 1, 1);
	start_this_cpu(NULL);
}

static int gperfctr_control(struct gperfctr_control *argp)
{
	unsigned long interval_usec;
	unsigned int nrcpus, i;
	int last_active, ret;
	struct gperfctr *perfctr;
	struct perfctr_cpu_control cpu_control;
	static DECLARE_MUTEX(control_mutex);

	if( nr_active_cpus > 0 )
		return -EBUSY;	/* you have to stop them first */
	if( get_user(interval_usec, &argp->interval_usec) )
		return -EFAULT;
	if( get_user(nrcpus, &argp->nrcpus) )
		return -EFAULT;
	if( nrcpus > smp_num_cpus )
		return -EINVAL;
	down(&control_mutex);
	last_active = -1;
	for(i = 0; i < nrcpus; ++i) {
		ret = -EFAULT;
		if( copy_from_user(&cpu_control,
				   &argp->cpu_control[i],
				   sizeof cpu_control) )
			goto out_up;
		/* we don't permit i-mode counters */
		ret = -EPERM;
		if( cpu_control.nrictrs != 0 )
			goto out_up;
		perfctr = &per_cpu_gperfctr[cpu_logical_map(i)];
		spin_lock(&perfctr->lock);
		perfctr->cpu_state.control = cpu_control;
		memset(&perfctr->cpu_state.sum, 0, sizeof perfctr->cpu_state.sum);
		ret = perfctr_cpu_update_control(&perfctr->cpu_state);
		spin_unlock(&perfctr->lock);
		if( ret < 0 )
			goto out_up;
		if( perfctr_cstatus_enabled(perfctr->cpu_state.cstatus) )
			last_active = i;
	}
	for(; i < smp_num_cpus; ++i) {
		perfctr = &per_cpu_gperfctr[cpu_logical_map(i)];
		memset(&perfctr->cpu_state, 0, sizeof perfctr->cpu_state);
	}
	nr_active_cpus = ret = last_active + 1;
	if( ret > 0 ) {
		if( reserve_hardware() < 0 ) {
			nr_active_cpus = 0;
			ret = -EBUSY;
		} else {
			start_all_cpus();
			start_sampling_timer(interval_usec);
		}
	}
 out_up:
	up(&control_mutex);
	return ret;
}

static int gperfctr_read(struct gperfctr_state *arg)
{
	unsigned nrcpus, i;
	struct gperfctr *perfctr;
	struct gperfctr_cpu_state state;

	if( get_user(nrcpus, &arg->nrcpus) )
		return -EFAULT;
	if( nrcpus > smp_num_cpus )
		nrcpus = smp_num_cpus;
	if( sampling_timer.data == 0 )	/* no timer; sample now */
		sample_all_cpus();
	for(i = 0; i < nrcpus; ++i) {
		perfctr = &per_cpu_gperfctr[cpu_logical_map(i)];
		spin_lock(&perfctr->lock);
		state.cpu_control = perfctr->cpu_state.control;
		state.sum = perfctr->cpu_state.sum;
		spin_unlock(&perfctr->lock);
		if( copy_to_user(&arg->cpu_state[i], &state, sizeof state) )
			return -EFAULT;
	}
	return nr_active_cpus;
}

static int gperfctr_stop(void)
{
	release_hardware();
	return 0;
}

static int dev_perfctr_ioctl(struct inode *inode, struct file *filp,
			     unsigned int cmd, unsigned long arg)
{
	switch( cmd ) {
	case PERFCTR_INFO:
		return sys_perfctr_info((struct perfctr_info*)arg);
	case GPERFCTR_CONTROL:
		return gperfctr_control((struct gperfctr_control*)arg);
	case GPERFCTR_READ:
		return gperfctr_read((struct gperfctr_state*)arg);
	case GPERFCTR_STOP:
		return gperfctr_stop();
	}
	return -EINVAL;
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)
static struct file_operations dev_perfctr_file_ops = {
	.owner = THIS_MODULE,
	.ioctl = dev_perfctr_ioctl,
};
#else
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
static struct file_operations dev_perfctr_file_ops = {
	.open = dev_perfctr_open,
	.release = dev_perfctr_release,
	.ioctl = dev_perfctr_ioctl,
};
#endif

static struct miscdevice dev_perfctr = {
	.minor = 182,
	.name = "perfctr",
	.fops = &dev_perfctr_file_ops,
};

int __init gperfctr_init(void)
{
	int i, err;

	if( (err = misc_register(&dev_perfctr)) != 0 ) {
		printk(KERN_ERR "/dev/perfctr: failed to register, errno %d\n",
		       -err);
		return err;
	}
	for(i = 0; i < smp_num_cpus; ++i)
		per_cpu_gperfctr[i].lock = SPIN_LOCK_UNLOCKED;
	return 0;
}

void gperfctr_exit(void)
{
	misc_deregister(&dev_perfctr);
}
