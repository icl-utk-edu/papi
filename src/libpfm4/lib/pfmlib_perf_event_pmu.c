/*
 * pfmlib_perf_pmu.c: support for perf_events event table
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@google.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#define _GNU_SOURCE /* getline() */
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <fcntl.h>
#include <syscall.h> /* for openat() */
#include <sys/param.h>

/*
 * looks like several distributions do not have
 * the latest libc with openat support, so disable
 * for now
 */
#undef HAS_OPENAT

#include "pfmlib_priv.h"

#define PERF_MAX_UMASKS	8

typedef struct {
	char		*uname;	/* unit mask name */
	char		*udesc;	/* unit mask desc */
	uint64_t	uid;	/* unit mask id */
} perf_umask_t;
	
typedef struct {
	char		*name;			/* name */
	char		*desc;			/* description */
	uint64_t	id;			/* perf_hw_id or equivalent */
	int		modmsk;			/* modifiers bitmask */
	int		type;			/* perf_type_id */
	int		numasks;		/* number of unit masls */
	unsigned long	umask_ovfl_idx;		/* base index of overflow unit masks */
	perf_umask_t	umasks[PERF_MAX_UMASKS];/* first unit masks */
} perf_event_t;

#define PCL_EVT(f, t, m)	\
	{ .name = #f,		\
	  .id = (f),		\
	  .type = (t),		\
	  .desc = #f,		\
	  .numasks = 0,		\
	  .modmsk = (m)	\
	}


static char debugfs_mnt[MAXPATHLEN];

#define PERF_ATTR_U	0 /* user level (interpretation is arch-specific)*/
#define PERF_ATTR_K	1 /* kernel level (interpretation is arch-specific) */
#define PERF_ATTR_H	2 /* hypervisor level (interpretation is arch-specific) */

#define _PERF_ATTR_U	(1 << PERF_ATTR_U)
#define _PERF_ATTR_K	(1 << PERF_ATTR_K)
#define _PERF_ATTR_H	(1 << PERF_ATTR_H)

static const pfmlib_attr_desc_t perf_mods[]={
	PFM_ATTR_B("u", "monitor at priv level 1, 2, 3"),	/* monitor priv levels 1, 2, 3 */
	PFM_ATTR_B("k", "monitor at priv level 0"),		/* montor priv level 0 */
	PFM_ATTR_B("h", "monitor in hypervisor"),		/* virtualization host */
	PFM_ATTR_NULL
};
#define modx(a, z) (perf_mods[(a)].z)

#define PERF_ATTR_HW (_PERF_ATTR_U|_PERF_ATTR_K|_PERF_ATTR_H)
#define PERF_ATTR_SW (_PERF_ATTR_U|_PERF_ATTR_K)

#include "events/perf_events.h"

pfmlib_pmu_t perf_event_support;
#define perf_nevents (perf_event_support.pme_count)

static perf_event_t *perf_pe;
static perf_umask_t *perf_um;

static void *perf_pe_free, *perf_pe_end;
static void *perf_um_free, *perf_um_end;
static size_t perf_pe_sz, perf_um_sz;

static inline int
perf_attr2mod(int pidx, int attr_idx)
{
	int x, n;

	n = attr_idx - perf_pe[pidx].numasks;

	pfmlib_for_each_bit(x, perf_pe[pidx].modmsk) {
		if (n == 0)
			break;
		n--;
	}
	return x;
}


/*
 * figure out the mount point of the debugfs filesystem
 *
 * returns -1 if none is found
 */
static int
get_debugfs_mnt(void)
{
	FILE *fp;
	char *buffer = NULL;
	size_t len = 0;
	char *q, *mnt, *fs;
	int res = -1;

	fp = fopen("/proc/mounts", "r");
	if (!fp)
		return -1;

	while(getline(&buffer, &len, fp) != -1) {

		q = strchr(buffer, ' ');
		if (!q)
			continue;
		mnt = ++q;
		q = strchr(q, ' ');
		if (!q)
			continue;
		*q = '\0';

		fs = ++q;
		q = strchr(q, ' ');
		if (!q)
			continue;
		*q = '\0';

		if (!strcmp(fs, "debugfs")) {
			strncpy(debugfs_mnt, mnt, MAXPATHLEN);
			debugfs_mnt[MAXPATHLEN-1]= '\0';
			res = 0;
			break;
		}
	}
	if (buffer)
		free(buffer);

	fclose(fp);

	return res;
}

#define PERF_ALLOC_EVCHUNK	(512)
#define PERF_ALLOC_UMCHUNK	(1024)

/*
 * clone static event table into a  dynamic
 * event table
 *
 * Used for tracepoints
 */
static void *
perf_table_clone(void)
{
	void *addr;
	size_t sz, stat_sz;

	stat_sz = perf_nevents * sizeof(perf_event_t);

	sz = perf_pe_sz = stat_sz + PERF_ALLOC_EVCHUNK;

	addr = malloc(sz);
	if (addr) {
		memcpy(addr, perf_static_events, stat_sz);
		perf_pe_free = addr + stat_sz;
		perf_pe_end = perf_pe_free + PERF_ALLOC_EVCHUNK;
		perf_pe = addr;
	}
	return addr;
}

/*
 * allocate space for one new event in event table
 *
 * returns NULL if out-of-memory
 *
 * may realloc existing table if necessary for growth
 */
static void *
perf_table_alloc_event(void)
{
	void *new_pe;
	size_t sz;

retry:
	sz = sizeof(perf_event_t);
	if ((perf_pe_free+sz) < perf_pe_end) {
		perf_pe_free += sz;
		return perf_pe_free - sz;
	}
	perf_pe_sz += PERF_ALLOC_EVCHUNK;
	
	new_pe = realloc(perf_pe, perf_pe_sz);
	if (!new_pe) 
		return NULL;
	
	perf_pe_free = new_pe + (perf_pe_free - (void *)perf_pe);
	perf_pe_end = perf_pe_free + PERF_ALLOC_EVCHUNK;
	perf_pe = new_pe;

	goto retry;
}

/*
 * allocate space for overflow new unit masks
 *
 * Each event can hold up to PERF_MAX_UMASKS.
 * But gievn we can dynamically add events
 * which may have more unit masks, then we
 * put them into a separate overflow unit
 * masks table which can grow on demand.
 * In that case the first PERF_MAX_UMASKS
 * are in the event, the rest in the overflow
 * table at index pointed to by event->umask_ovfl_idx
 * All unit masks for an event are contiguous in the
 * overflow table.
 */
static void *
perf_table_alloc_umask(void)
{
	void *new_um;
	size_t sz;

retry:
	sz = sizeof(perf_umask_t);
	if ((perf_um_free+sz) < perf_um_end) {
		perf_um_free += sz;
		return perf_um_free - sz;
	}
	perf_um_sz += PERF_ALLOC_UMCHUNK;
	
	new_um = realloc(perf_um, perf_um_sz);
	if (!new_um) 
		return NULL;
	
	perf_um_free = new_um + (perf_um_free - (void *)perf_um);
	perf_um_end = perf_um_free + PERF_ALLOC_UMCHUNK;
	perf_um = new_um;

	goto retry;
}

static inline unsigned long
perf_get_ovfl_umask_idx(perf_umask_t *um)
{
	return um - perf_um;
}

static inline perf_umask_t *
perf_get_ovfl_umask(int pidx)
{
	return perf_um+perf_pe[pidx].umask_ovfl_idx;
}

static void
gen_tracepoint_table(void)
{
	DIR *dir1, *dir2;
	struct dirent *d1, *d2;
	perf_event_t *p;
	perf_umask_t *um;
	char d2path[MAXPATHLEN];
	char idpath[MAXPATHLEN];
	char id_str[32];
	uint64_t id;
	int fd, err;
	int dir2_fd, reuse_event = 0;
	int numasks;

	err = get_debugfs_mnt();
	if (err == -1)
		return;

	strncat(debugfs_mnt, "/tracing/events", MAXPATHLEN);
	debugfs_mnt[MAXPATHLEN-1]= '\0';

	dir1 = opendir(debugfs_mnt);
	if (!dir1)
		return;

	p = perf_table_clone();

	err = 0;
	while((d1 = readdir(dir1)) && err >= 0) {

		if (!strcmp(d1->d_name, "."))
			continue;

		if (!strcmp(d1->d_name, ".."))
			continue;

		snprintf(d2path, MAXPATHLEN, "%s/%s", debugfs_mnt, d1->d_name);

		/* fails if d2path is not a directory */
		dir2 = opendir(d2path);
		if (!dir2)
			continue;

		dir2_fd = dirfd(dir2);

		
		/*
 		 * if a subdir did not fit our expected
 		 * tracepoint format, then we reuse the
 		 * allocatoed space (with have no free)
 		 */
		if (!reuse_event)
			p = perf_table_alloc_event();

		if (p)
			p->name = strdup(d1->d_name);

		if (!(p && p->name)) {
			closedir(dir2);
			err = -1;
			continue;
		}

		p->desc = "tracepoint";
		p->id = -1;
		p->type = PERF_TYPE_TRACEPOINT;
		p->umask_ovfl_idx = 0;
		p->modmsk = 0,

		numasks = 0;
		reuse_event = 0;

		while((d2 = readdir(dir2))) {
			if (!strcmp(d2->d_name, "."))
				continue;

			if (!strcmp(d2->d_name, ".."))
				continue;

#ifdef HAS_OPENAT
                        snprintf(idpath, MAXPATHLEN, "%s/id", d2->d_name);
                        fd = openat(dir2_fd, idpath, O_RDONLY);
#else
                        snprintf(idpath, MAXPATHLEN, "%s/%s/id", d2path, d2->d_name);
                        fd = open(idpath, O_RDONLY);
#endif
			if (fd == -1)
				continue;

			err = read(fd, id_str, sizeof(id_str));

			close(fd);

			if (err < 0)
				continue;

			id = strtoull(id_str, NULL, 0);

			if (numasks < PERF_MAX_UMASKS)
				um = p->umasks+numasks;
			else {
				um = perf_table_alloc_umask();
				if (numasks == PERF_MAX_UMASKS)
					p->umask_ovfl_idx = perf_get_ovfl_umask_idx(um);
			}

			if (!um) {
				err = -1;
				break;
			}

			/*
			 * tracepoint have no event codes
			 * the code is in the unit masks
			 */
			p->id = 0;

			um->uname = strdup(d2->d_name);
			if (!um->uname) {
				close(fd);
				err = -1;
				break;
			}
			um->udesc = um->uname;
			um->uid   = id;
			DPRINT("idpath=%s:%s id=%"PRIu64"\n", p->name, um->uname, id);
			numasks++;
		}
		p->numasks = numasks;

		closedir(dir2);

		/*
		 * directory was not pointing
		 * to a tree structure we know about
		 */
		if (!numasks) {
			free(p->name);
			reuse_event =1;
			continue;
		}

		/*
 		 * update total number of events
 		 * only when no error is reported
 		 */
		if (err >= 0)
			perf_nevents++;
		reuse_event = 0;
	}
	closedir(dir1);
}

static int
pfm_perf_detect(void *this)
{
	/* ought to find a better way of detecting PERF */
#define PERF_OLD_PROC_FILE "/proc/sys/kernel/perf_counter_paranoid"
#define PERF_PROC_FILE "/proc/sys/kernel/perf_event_paranoid"
	return !(access(PERF_PROC_FILE, F_OK)
		  && access(PERF_OLD_PROC_FILE, F_OK)) ? PFM_SUCCESS: PFM_ERR_NOTSUPP;
}

static int
pfm_perf_init(void *this)
{

	perf_pe = perf_static_events;

	gen_tracepoint_table();

	/* must dynamically add tracepoints */
	return PFM_SUCCESS;
}

static int
pfm_perf_get_event_first(void *this)
{
	return 0;
}

static int
pfm_perf_get_event_next(void *this, int idx)
{
	if (idx < 0 || idx >= (perf_nevents-1))
		return -1;

	return idx+1;
}

static int
pfmlib_perf_encode_tp(pfmlib_event_desc_t *e, uint64_t *codes, int *count)
{
	perf_umask_t *um;
	pfmlib_attr_t *a;
	int i, nu = 0;

	e->fstr[0] = '\0';
	evt_strcat(e->fstr, "%s", perf_pe[e->event].name);
	/*
	 * look for tracepoints
	 */
	for(i=0; i < e->nattrs; i++) {
		a = e->attrs+i;
		if (a->type == PFM_ATTR_UMASK) {
			/*
			 * tracepoint unit masks cannot be combined
			 */
			if (++nu > 1)
				return PFM_ERR_FEATCOMB;

			if (a->id < PERF_MAX_UMASKS) {
				*codes = perf_pe[e->event].umasks[a->id].uid;
				evt_strcat(e->fstr, ":%s", perf_pe[e->event].umasks[a->id].uname);
			} else {
				um = perf_get_ovfl_umask(e->event);
				*codes = um[a->id - PERF_MAX_UMASKS].uid;
				evt_strcat(e->fstr, ":%s", um[a->id - PERF_MAX_UMASKS].uname);
			}
		}
	}
	return PFM_SUCCESS;
}

static int
pfmlib_perf_encode_hw_cache(pfmlib_event_desc_t *e, uint64_t *codes, int *count)
{
	pfmlib_attr_t *a;
	int i;

	*codes = perf_pe[e->event].id;
	*count = 1;

	e->fstr[0] = '\0';
	evt_strcat(e->fstr, "%s", perf_pe[e->event].name);

	for(i=0; i < e->nattrs; i++) {
		a = e->attrs+i;
		if (a->type == PFM_ATTR_UMASK) {
			*codes |= perf_pe[e->event].umasks[a->id].uid;
			evt_strcat(e->fstr, ":%s", perf_pe[e->event].umasks[a->id].uname);
		}
	}
	return PFM_SUCCESS;
}

static int
pfm_perf_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{
	pfmlib_attr_t *a;
	int i, ret;

	switch(perf_pe[e->event].type) {
	case PERF_TYPE_TRACEPOINT:
		ret = pfmlib_perf_encode_tp(e, codes, count);
		break;
	case PERF_TYPE_HW_CACHE:
		ret = pfmlib_perf_encode_hw_cache(e, codes, count);
		break;
	case PERF_TYPE_HARDWARE:
	case PERF_TYPE_SOFTWARE:
		ret = PFM_SUCCESS;
		*codes = perf_pe[e->event].id;
		*count = 1;
		e->fstr[0] = '\0';
		evt_strcat(e->fstr, "%s", perf_pe[e->event].name);
		break;
	default:
		DPRINT("unsupported event type=%d\n", perf_pe[e->event].type);
		return PFM_ERR_NOTSUPP;
	}

	if (ret != PFM_SUCCESS)
		return ret;

	if (!attrs)
		return PFM_SUCCESS;

	/*
	 * propagate priv level to caller attrs struct
	 */
	for(i=0; i < e->nattrs; i++) {
		a = e->attrs+i;

		if (a->type == PFM_ATTR_UMASK)
			continue;

		switch(a->id - perf_pe[e->event].numasks) {
		case PERF_ATTR_U:
			if (a->ival)
				attrs->plm |= PFM_PLM3;
			break;	
		case PERF_ATTR_K:
			if (a->ival)
				attrs->plm |= PFM_PLM0;
			break;	
		case PERF_ATTR_H:
			if (a->ival)
				attrs->plm |= PFM_PLMH;
			break;	
		}
	
	}
	return PFM_SUCCESS;
}

static int
pfm_perf_event_is_valid(void *this, int idx)
{
	return idx >= 0 && idx < perf_nevents;
}

static int
pfm_perf_get_event_perf_type(void *this, int pidx)
{
	return perf_pe[pidx].type;
}

int
pfm_perf_get_event_attr_info(void *this, int idx, int attr_idx, pfm_event_attr_info_t *info)
{
	perf_umask_t *um;
	int m;

	if (attr_idx < perf_pe[idx].numasks) {
		if (attr_idx < PERF_MAX_UMASKS) {
			um = &perf_pe[idx].umasks[attr_idx];
		} else {
			um  = perf_get_ovfl_umask(idx);
			um += attr_idx - PERF_MAX_UMASKS;
		}
		info->name = um->uname;
		info->desc = um->udesc;
		info->equiv= NULL;
		info->code = um->uid;
		info->type = PFM_ATTR_UMASK;
	} else {
		m = perf_attr2mod(idx, attr_idx);
		info->name = modx(m, name);
		info->desc = modx(m, desc);
		info->equiv= NULL;
		info->code = m;
		info->type = modx(m, type);
	}
	info->is_dfl = 0;
	info->idx = attr_idx;
	info->dfl_val64 = 0;

	return PFM_SUCCESS;
}

int
pfm_perf_get_event_info(void *this, int idx, pfm_event_info_t *info)
{
	info->name  = perf_pe[idx].name;
	info->desc  = perf_pe[idx].desc;
	info->code  = perf_pe[idx].id;

	/* unit masks + modifiers */
	info->nattrs  = perf_pe[idx].numasks;
	info->nattrs += pfmlib_popcnt((unsigned long)perf_pe[idx].modmsk);

	return PFM_SUCCESS;
}

pfmlib_pmu_t perf_event_support={
	.desc			= "perf_events generic PMU",
	.name			= "perf",
	.pmu			= PFM_PMU_PERF_EVENT,
	.pme_count		= PME_PERF_EVENT_COUNT,
	.max_encoding		= 1,
	.pmu_detect		= pfm_perf_detect,
	.pmu_init		= pfm_perf_init,
	.get_event_encoding	= pfm_perf_get_encoding,
	.get_event_first	= pfm_perf_get_event_first,
	.get_event_next		= pfm_perf_get_event_next,
	.event_is_valid		= pfm_perf_event_is_valid,
	.get_event_perf_type	= pfm_perf_get_event_perf_type,
	.get_event_info		= pfm_perf_get_event_info,
	.get_event_attr_info	= pfm_perf_get_event_attr_info,
};
