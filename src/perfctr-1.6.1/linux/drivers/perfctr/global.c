/* $Id$
 * Global-mode performance-monitoring counters.
 *
 * Copyright (C) 2000  Mikael Pettersson
 *
 * XXX: Doesn't do any authentication yet. Should we limit control
 * to root, or base it on having write access to /dev/perfctr?
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/init.h>
#include <linux/perfctr.h>

#include <asm/uaccess.h>

#include "compat.h"
#include "global.h"

static const char this_service[] = __FILE__;
static int hardware_is_ours = 0;
static struct timer_list sampling_timer;

unsigned int nr_active_cpus = 0;

struct gperfctr {
	struct gperfctr_cpu_state state;
	struct perfctr_low_ctrs start;
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

static __inline__ void sample_this_cpu(void *unused)
{
	struct gperfctr *perfctr;
	struct perfctr_low_ctrs now;
	int nrctrs, i;

	perfctr = &per_cpu_gperfctr[smp_processor_id()];
	nrctrs = perfctr->state.nrctrs;
	if( nrctrs <= 0 )
		return;
	perfctr_cpu_read_counters(nrctrs, &now);
	spin_lock(&perfctr->lock);
	for(i = nrctrs; --i >= 0;) {
		perfctr->state.sum.ctr[i] += now.ctr[i] - perfctr->start.ctr[i];
		perfctr->start.ctr[i] = now.ctr[i];
	}
	spin_unlock(&perfctr->lock);
}

static __inline__ void sample_all_cpus(void)
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

static __inline__ unsigned long usectojiffies(unsigned long usec)
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
	int nrctrs;

	perfctr = &per_cpu_gperfctr[smp_processor_id()];
	nrctrs = perfctr->state.nrctrs;
	if( nrctrs <= 0 )
		return;
	perfctr_cpu_write_control(nrctrs, &perfctr->state.control);
	perfctr_cpu_read_counters(nrctrs, &perfctr->start);
}

static void start_all_cpus(void)
{
	smp_call_function(start_this_cpu, NULL, 1, 1);
	start_this_cpu(NULL);
}

int gperfctr_control(struct gperfctr_control *arg)
{
	unsigned long interval_usec;
	unsigned int nrcpus, i;
	int last_active, nrctrs, ret;
	struct gperfctr *perfctr;
	struct perfctr_control control;
	static DECLARE_MUTEX(control_mutex);

	if( nr_active_cpus > 0 )
		return -EBUSY;	/* you have to stop them first */
	if( get_user(interval_usec, &arg->interval_usec) )
		return -EFAULT;
	if( get_user(nrcpus, &arg->nrcpus) )
		return -EFAULT;
	if( nrcpus > smp_num_cpus )
		return -EINVAL;
	down(&control_mutex);
	last_active = -1;
	for(i = 0; i < nrcpus; ++i) {
		if( copy_from_user(&control, &arg->cpu_control[i],
				   sizeof control) ) {
			ret = -EFAULT;
			goto out_up;
		}
		nrctrs = perfctr_cpu_check_control(&control);
		if( nrctrs < 0 ) {
			ret = nrctrs;
			goto out_up;
		}
		if( nrctrs > 0 )
			last_active = i;
		perfctr = &per_cpu_gperfctr[cpu_logical_map(i)];
		spin_lock(&perfctr->lock);
		perfctr->state.nrctrs = nrctrs;
		perfctr->state.control = control;
		memset(&perfctr->state.sum, 0, sizeof perfctr->state.sum);
		spin_unlock(&perfctr->lock);
	}
	for(; i < smp_num_cpus; ++i)
		per_cpu_gperfctr[cpu_logical_map(i)].state.nrctrs = 0;
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

int gperfctr_read(struct gperfctr_state *arg)
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
		state = perfctr->state;
		spin_unlock(&perfctr->lock);
		if( copy_to_user(&arg->cpu_state[i], &state, sizeof state) )
			return -EFAULT;
	}
	return nr_active_cpus;
}

int gperfctr_stop(void)
{
	release_hardware();
	return 0;
}

void __init gperfctr_init(void)
{
	int i;

	for(i = 0; i < smp_num_cpus; ++i)
		per_cpu_gperfctr[i].lock = SPIN_LOCK_UNLOCKED;
}
