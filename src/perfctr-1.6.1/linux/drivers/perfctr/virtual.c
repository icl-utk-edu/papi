/* $Id$
 * Virtual per-process performance counters.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/perfctr.h>

#include <asm/io.h>
#include <asm/uaccess.h>

#include "compat.h"
#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,4,0)) && defined(CONFIG_VPERFCTR_PROC) && !defined(MODULE)
#include <linux/proc_fs.h>
#endif
#include "virtual.h"

/****************************************************************
 *								*
 * Data types and macros.					*
 *								*
 ****************************************************************/

struct vperfctr {
/* User-visible fields: (must be first for mmap()) */
	struct vperfctr_state state;
/* Kernel-private fields: */
	atomic_t count;
	struct file *filp;
};
#define IS_RUNNING(perfctr)	((perfctr)->state.status > 0)

/****************************************************************
 *								*
 * Resource management.						*
 *								*
 ****************************************************************/

/* XXX: perhaps relax this to number of _live_ perfctrs */
/* XXX: nrctrs should be an atomic_t, but <asm-i386/atomic.h> doesn't
   support atomic_inc_return, and its atomic_inc_and_test tests for
   the wrong condition (-1 --> 0, not 0 --> +1). */
static spinlock_t nrctrs_lock = SPIN_LOCK_UNLOCKED;
int nrctrs = 0;
static const char this_service[] = __FILE__;

static int inc_nrctrs(void)
{
	const char *other;

	other = NULL;
	spin_lock(&nrctrs_lock);
	if( ++nrctrs == 1 )
		other = perfctr_cpu_reserve(this_service);
	spin_unlock(&nrctrs_lock);
	if( other ) {
		printk(KERN_ERR __FILE__
		       ": cannot operate, perfctr hardware taken by '%s'\n",
		       other);
		return -EBUSY;
	}
	return 0;
}

static void dec_nrctrs(void)
{
	spin_lock(&nrctrs_lock);
	if( --nrctrs == 0 )
		perfctr_cpu_release(this_service);
	spin_unlock(&nrctrs_lock);
}

static struct vperfctr *vperfctr_alloc(void)
{
	unsigned long page;

	if( inc_nrctrs() != 0 )
		return NULL;
	page = get_zeroed_page(GFP_KERNEL);
	if( !page ) {
		dec_nrctrs();
		return NULL;
	}
	SetPageReserved(virt_to_page(page));
	return (struct vperfctr*) page;
}

static void vperfctr_free(struct vperfctr *perfctr)
{
	ClearPageReserved(virt_to_page(perfctr));
	free_page((unsigned long)perfctr);
	dec_nrctrs();
}

static struct vperfctr *get_empty_vperfctr(void)
{
	struct vperfctr *perfctr = vperfctr_alloc();
	if( perfctr ) {
		atomic_set(&perfctr->count, 1);
		perfctr->filp = NULL;
	}
	return perfctr;
}

static void put_vperfctr(struct vperfctr *perfctr)
{
	if( atomic_dec_and_test(&perfctr->count) )
		vperfctr_free(perfctr);
}

static unsigned int new_control_id(void)
{
	/* XXX: `atomic_inc_return' would have been nice here ... */
	static spinlock_t lock = SPIN_LOCK_UNLOCKED;
	static unsigned int counter;
	int id;

	spin_lock(&lock);
	id = ++counter;
	spin_unlock(&lock);
	return id;
}

/****************************************************************
 *								*
 * Basic counter operations.					*
 *								*
 ****************************************************************/

/* Sample the CPU's registers to *now. Update accumulated counters. */
static __inline__ void
vperfctr_accumulate(struct vperfctr *perfctr, struct perfctr_low_ctrs *now)
{
	int i;

	perfctr_cpu_read_counters(perfctr->state.status, now);
	for(i = perfctr->state.status; --i >= 0;)
		perfctr->state.sum.ctr[i] +=
			now->ctr[i] - perfctr->state.start.ctr[i];
}

/* PRE: perfctr == TASK_VPERFCTR(current) && IS_RUNNING(perfctr)
 * Suspend the counters.
 */
static __inline__ void vperfctr_suspend(struct vperfctr *perfctr)
{
	struct perfctr_low_ctrs now;
	vperfctr_accumulate(perfctr, &now);
	perfctr_cpu_disable_rdpmc();
}

/* PRE: perfctr == TASK_VPERFCTR(current) && IS_RUNNING(perfctr)
 * Restart the counters.
 */
static __inline__ void vperfctr_resume(struct vperfctr *perfctr)
{
	perfctr_cpu_enable_rdpmc();
	perfctr_cpu_write_control(perfctr->state.status, &perfctr->state.control);
	perfctr_cpu_read_counters(perfctr->state.status, &perfctr->state.start);
}

/****************************************************************
 *								*
 * Process management callbacks.				*
 *								*
 ****************************************************************/

/* do_fork() --> copy_thread() --> perfctr_copy_thread() --> __vperfctr_copy()
 * Copy parent's perfctr setup to a new child.
 * Note: do not inherit interrupt-mode perfctrs (when implemented).
 */
struct vperfctr *__vperfctr_copy(struct vperfctr *parent)
{
	struct vperfctr *child;
	if( (child = get_empty_vperfctr()) != NULL ) {
		child->state.status = parent->state.status;
		child->state.control_id = parent->state.control_id;
		child->state.control = parent->state.control;
		/* child's counters start from zero */
	}
	return child;
}

/* Called from from exit_thread() or sys_vperfctr_unlink().
 * PRE: perfctr == TASK_VPERFCTR(current)
 * Current halts its vperfctr, by exiting or by request.
 * If the counters are running, stop them and sample their final values.
 * Mark this perfctr as dead.
 */
void __vperfctr_exit(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) )
		vperfctr_suspend(perfctr);
	perfctr->state.status = VPERFCTR_STATUS_DEAD;
}

/* sys_wait4() --> .. --> release_thread() --> .. --> __vperfctr_release().
 * PRE: parent == TASK_VPERFCTR(current).
 * Parent waits for and finds terminated child.
 * If child's perfctr was inherited from parent [same control_id],
 * add child's self and children counts to parent's children counts.
 */
void __vperfctr_release(struct vperfctr *parent, struct vperfctr *child)
{
	if( parent && parent->state.control_id == child->state.control_id ) {
		int i;
		for(i = parent->state.status; --i >= 0;)
			parent->state.children.ctr[i] +=
				child->state.sum.ctr[i] +
				child->state.children.ctr[i];
	}
	put_vperfctr(child);
}

/* schedule() --> switch_to() --> .. --> __vperfctr_suspend().
 * If the counters are running, sample their values.
 */
void __vperfctr_suspend(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) )
		vperfctr_suspend(perfctr);
}

/* schedule() --> switch_to() --> .. --> __vperfctr_resume().
 * PRE: perfctr == TASK_VPERFCTR(current)
 * If the counters are runnable, resume them.
 */
void __vperfctr_resume(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) )
		vperfctr_resume(perfctr);
}

/****************************************************************
 *								*
 * Virtual perfctr "system calls".				*
 *								*
 ****************************************************************/

/* PRE: perfctr == TASK_VPERFCTR(current) */
static int sys_vperfctr_stop(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) ) {
		vperfctr_suspend(perfctr);
		perfctr->state.status = VPERFCTR_STATUS_STOPPED;
	}
	return 0;
}

/* PRE: perfctr == TASK_VPERFCTR(current) */
static int
sys_vperfctr_control(struct vperfctr *perfctr, struct perfctr_control *argp)
{
	struct perfctr_control control;
	int status;
	int prev_status, i;
	unsigned int prev_start_tsc;

	if( copy_from_user(&control, argp, sizeof control) )
		return -EFAULT;
	if( (status = perfctr_cpu_check_control(&control)) < 0 )
		return status;
	if( status == 0 )
		return sys_vperfctr_stop(perfctr);

	prev_status = perfctr->state.status;
	perfctr->state.status = status;
	perfctr->state.control_id = new_control_id();
	perfctr->state.control = control;

	/*
	 * Clear the perfctr sums and restart the perfctrs.
	 *
	 * If the counters were running before this control call,
	 * then don't clear the time-stamp counter's sum and don't
	 * overwrite its current start value.
	 */
	if( prev_status == 0 )
		perfctr->state.sum.ctr[0] = 0;
	for(i = 1; i < ARRAY_SIZE(perfctr->state.sum.ctr); ++i)
		perfctr->state.sum.ctr[i] = 0;
	prev_start_tsc = perfctr->state.start.ctr[0];
	vperfctr_resume(perfctr);	/* clobbers start.ctr[0] :-( */
	if( prev_status > 0 )
		perfctr->state.start.ctr[0] = prev_start_tsc;

	return 0;
}

/* PRE: perfctr == TASK_VPERFCTR(current) */
static int sys_vperfctr_unlink(struct vperfctr *perfctr)
{
	TASK_VPERFCTR(current) = NULL;
	__vperfctr_exit(perfctr);
	put_vperfctr(perfctr);
	return 0;
}

/* PRE: perfctr == TASK_VPERFCTR(current)
 * Sample the current process' counters and update state.
 * This operation is used on processors like the pre-MMX Intel P5,
 * which cannot sample the counter registers in user-mode.
 */
int sys_vperfctr_sample(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) ) {
		struct perfctr_low_ctrs now;
		vperfctr_accumulate(perfctr, &now);
		perfctr->state.start = now;
	}
	return 0;
}

/****************************************************************
 *								*
 * Virtual perfctr file operations.				*
 *								*
 ****************************************************************/

static int vperfctr_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct vperfctr *perfctr;

	/* Only allow read-only mapping of first page. */
	if( (vma->vm_end - vma->vm_start) != PAGE_SIZE ||
	    vma_pgoff(vma) != 0 ||
	    (pgprot_val(vma->vm_page_prot) & _PAGE_RW) ||
	    (vma->vm_flags & (VM_WRITE | VM_MAYWRITE)) )
		return -EPERM;
	perfctr = filp->private_data;
	return remap_page_range(vma->vm_start, virt_to_phys(perfctr),
				PAGE_SIZE, vma->vm_page_prot);
}

static int vperfctr_release(struct inode *inode, struct file *filp)
{
	struct vperfctr *perfctr = filp->private_data;
	/*
	 * With the addition of /proc/self/perfctr, there is no longer
	 * a one-to-one perfctr <--> filp correspondence. This may
	 * change again when /proc/self/perfctr is reimplemented as
	 * a symbolic link to "perfctr:[...]" and the /dev/perfctr
	 * interface is dropped.
	 */
	if( filp == perfctr->filp )
		perfctr->filp = NULL;
	filp->private_data = NULL;
	put_vperfctr(perfctr);
	return 0;
}

static int vperfctr_ioctl(struct inode *inode, struct file *filp,
			  unsigned int cmd, unsigned long arg)
{
	struct vperfctr *perfctr = filp->private_data;

	if( perfctr != TASK_VPERFCTR(current) )
		return -EPERM;
	switch( cmd ) {
	case PERFCTR_INFO:
		return sys_perfctr_info((struct perfctr_info*)arg);
	case VPERFCTR_CONTROL:
		return sys_vperfctr_control(perfctr, (struct perfctr_control*)arg);
	case VPERFCTR_STOP:
		return sys_vperfctr_stop(perfctr);
	case VPERFCTR_UNLINK:
		return sys_vperfctr_unlink(perfctr);
	case VPERFCTR_SAMPLE:
		return sys_vperfctr_sample(perfctr);
	}
	return -EINVAL;
}

#ifdef CONFIG_VPERFCTR_PROC
static int vperfctr_open(struct inode *inode, struct file *filp)
{
	struct task_struct *tsk;
	struct vperfctr *perfctr;

	tsk = current;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)
	if( tsk != inode->u.proc_i.task )
		return -EPERM;
#else
	if( tsk->pid != (inode->i_ino >> 16) )
		return -EPERM;
#endif
	perfctr = TASK_VPERFCTR(tsk);
	if( !perfctr ) {
		perfctr = get_empty_vperfctr();
		if( !perfctr )
			return -ENOMEM;
		perfctr->filp = filp;
	}
	filp->private_data = perfctr;
	atomic_inc(&perfctr->count);
	if( !TASK_VPERFCTR(tsk) )
		TASK_VPERFCTR(tsk) = perfctr;
	return 0;
}
#endif

static struct file_operations vperfctr_file_ops = {
	OWNER_THIS_MODULE
	.mmap = vperfctr_mmap,
	.release = vperfctr_release,
	.ioctl = vperfctr_ioctl,
#ifdef CONFIG_VPERFCTR_PROC
	.open = vperfctr_open,
#endif
};

/* XXX: inode->u.proc_i.op.proc_get_link() ought to take a fourth 'follow'
   parameter indicating whether readlink() or follow_link() is the caller */

#if defined(CONFIG_VPERFCTR_PROC) && !defined(MODULE)

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

void perfctr_set_proc_pid_ops(struct inode *inode)
{
	inode->i_fop = &vperfctr_file_ops;
}

#else	/* 2.2 */

struct inode_operations perfctr_proc_pid_inode_operations = {
	.default_file_ops = &vperfctr_file_ops,
	.permission = proc_permission,
};

#endif	/* LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0) */

#endif	/* defined(CONFIG_VPERFCTR_PROC) && !defined(MODULE) */

/****************************************************************
 *								*
 * Virtual perfctr file system (based on fs/pipe.c)		*
 *								*
 ****************************************************************/

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

static struct super_block *
vperfctr_fs_read_super(struct super_block *sb, void *data, int silent)
{
	static const struct qstr qstr = { "vperfctr:", 9, 0 };
	struct inode *root = new_inode(sb);
	if( !root )
		return NULL;
	root->i_mode = S_IFDIR | S_IRUSR | S_IWUSR;
	root->i_uid = root->i_gid = 0;
	root->i_atime = root->i_mtime = root->i_ctime = CURRENT_TIME;
	sb->s_blocksize = 0;
	sb->s_blocksize_bits = 0;
	sb->s_magic = 0;
	sb->s_op = NULL;
	sb->s_root = d_alloc(NULL, &qstr);
	if( !sb->s_root ) {
		iput(root);
		return NULL;
	}
	sb->s_root->d_sb = sb;
	sb->s_root->d_parent = sb->s_root;
	d_instantiate(sb->s_root, root);
	return sb;
}

static DECLARE_FSTYPE(vperfctr_fs_type, "vperfctr", vperfctr_fs_read_super,
		      FS_NOMOUNT|FS_SINGLE);

static int __init vperfctr_fs_init(void)
{
	int err = register_filesystem(&vperfctr_fs_type);
	if( !err ) {
		struct vfsmount *mnt = kern_mount(&vperfctr_fs_type);
		if( !IS_ERR(mnt) )
			return 0;
		err = PTR_ERR(mnt);
		unregister_filesystem(&vperfctr_fs_type);
	}
	printk(KERN_ERR "vperfctr: failed to register file system (errno %d)\n",
	       -err);
	return err;
}

static void __exit vperfctr_fs_exit(void)
{
	unregister_filesystem(&vperfctr_fs_type);
	kern_umount(vperfctr_fs_type.kern_mnt);
}

#define set_inode_vperfctr_ops(inode)	\
	((inode)->i_fop = &vperfctr_file_ops)
#define set_inode_vperfctr_sb(inode)	\
	((inode)->i_sb = vperfctr_fs_type.kern_mnt->mnt_sb)
#define set_filp_vperfctr_vfsmnt(filp)	\
	((filp)->f_vfsmnt = mntget(vperfctr_fs_type.kern_mnt))

static int vperfctr_fs_d_delete(struct dentry *dentry)
{
	return 1;
}

static struct dentry_operations vperfctr_fs_dentry_ops = {
	.d_delete = vperfctr_fs_d_delete,
};

static struct dentry *d_alloc_vperfctr_root(struct inode *inode)
{
	struct qstr this;
	char name[32];
	struct dentry *dentry;

	sprintf(name, "[%lu]", inode->i_ino);
	this.name = name;
	this.len = strlen(name);
	this.hash = inode->i_ino;
	dentry = d_alloc(vperfctr_fs_type.kern_mnt->mnt_sb->s_root, &this);
	if( dentry ) {
		dentry->d_op = &vperfctr_fs_dentry_ops;
		d_add(dentry, inode);
	}
	return dentry;
}

#else	/* 2.4 simulation for 2.2; things were simpler then... */

#define vperfctr_fs_init()	(0)
#define vperfctr_fs_exit()	do { } while( 0 )
static struct inode_operations vperfctr_inode_ops = {
	.default_file_ops = &vperfctr_file_ops,
};
#define set_inode_vperfctr_ops(inode)	((inode)->i_op = &vperfctr_inode_ops)
#define set_inode_vperfctr_sb(inode)	do { } while( 0 )
#define set_filp_vperfctr_vfsmnt(filp)	do { } while( 0 )
#define d_alloc_vperfctr_root(inode)	d_alloc_root((inode), NULL)

#endif

/****************************************************************
 *								*
 * /dev/perfctr ioctl commands for virtual perfctrs.		*
 *								*
 ****************************************************************/

static struct file *get_vperfctr_filp(struct vperfctr *perfctr)
{
	struct inode *inode;
	struct file *filp;
	struct dentry *dentry;

	filp = get_empty_filp();
	if( !filp )
		return NULL;
	inode = get_empty_inode();
	if( !inode )
		goto out_filp;
	set_inode_vperfctr_ops(inode);
	set_inode_vperfctr_sb(inode);
	inode->i_state = I_DIRTY;
	inode->i_mode = S_IFCHR | S_IRUSR | S_IWUSR;
	inode->i_uid = current->fsuid;
	inode->i_gid = current->fsgid;
	inode->i_atime = inode->i_mtime = inode->i_ctime = CURRENT_TIME;
	inode->i_blksize = 0;

	dentry = d_alloc_vperfctr_root(inode);
	if( !dentry )
		goto out_inode;
	set_filp_vperfctr_vfsmnt(filp);
	filp->f_dentry = dentry;

	filp->f_pos = 0;
	filp->f_flags = 0;
	filp->f_mode = FMODE_READ;
	filp->f_op = fops_get(&vperfctr_file_ops);

	filp->private_data = perfctr;
	perfctr->filp = filp;
	atomic_inc(&perfctr->count);

	return filp;

 out_inode:
	iput(inode);
 out_filp:
	put_filp(filp);
	return NULL;
}

int vperfctr_attach_current(void)
{
	struct task_struct *tsk = current;
	struct vperfctr *perfctr;
	struct file *filp;
	int ret;

	perfctr = TASK_VPERFCTR(tsk);
	if( !perfctr ) {
		perfctr = get_empty_vperfctr();
		if( !perfctr )
			return -ENOMEM;
	}
	filp = perfctr->filp;
	if( filp ) {
		get_file(filp);
	} else {
		filp = get_vperfctr_filp(perfctr);
		ret = -ENFILE;
		if( !filp )
			goto out_perfctr;
	}
	ret = get_unused_fd();
	if( ret < 0 )
		goto out_filp;
	fd_install(ret, filp);
	if( !TASK_VPERFCTR(tsk) )
		TASK_VPERFCTR(tsk) = perfctr;
	return ret;
 out_filp:
	fput(filp);
 out_perfctr:
	if( !TASK_VPERFCTR(tsk) )
		vperfctr_free(perfctr);
	return ret;
}

/****************************************************************
 *								*
 * module_init/exit						*
 *								*
 ****************************************************************/

#ifdef MODULE
static struct vperfctr_stub off;

static void vperfctr_stub_init(void)
{
	write_lock(&vperfctr_stub_lock);
	off = vperfctr_stub;
	vperfctr_stub.copy = __vperfctr_copy;
	vperfctr_stub.exit = __vperfctr_exit;
	vperfctr_stub.release = __vperfctr_release;
	vperfctr_stub.suspend = __vperfctr_suspend;
	vperfctr_stub.resume = __vperfctr_resume;
#ifdef CONFIG_VPERFCTR_PROC
	vperfctr_stub.file_ops = &vperfctr_file_ops;
#endif
	write_unlock(&vperfctr_stub_lock);
}

static void vperfctr_stub_exit(void)
{
	write_lock(&vperfctr_stub_lock);
	vperfctr_stub = off;
	write_unlock(&vperfctr_stub_lock);
}
#else
static __inline__ void vperfctr_stub_init(void) { }
static __inline__ void vperfctr_stub_exit(void) { }
#endif	/* MODULE */

int __init vperfctr_init(void)
{
	int err = vperfctr_fs_init();
	if( !err )
		vperfctr_stub_init();
	return err;
}

void __exit vperfctr_exit(void)
{
	vperfctr_fs_exit();
	vperfctr_stub_exit();
}
