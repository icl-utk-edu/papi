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
#include <unistd.h>
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
	const char	*uname;	/* unit mask name */
	const char	*udesc;	/* unit mask desc */
	uint64_t	uid;	/* unit mask id */
	int		uflags;	/* umask options */
	int		grpid;	/* group identifier */
} perf_umask_t;
	
typedef struct {
	const char	*name;			/* name */
	const char	*desc;			/* description */
	uint64_t	id;			/* perf_hw_id or equivalent */
	int		modmsk;			/* modifiers bitmask */
	int		type;			/* perf_type_id */
	int		numasks;		/* number of unit masls */
	int		ngrp;			/* number of umasks groups */
	unsigned long	umask_ovfl_idx;		/* base index of overflow unit masks */
	perf_umask_t	umasks[PERF_MAX_UMASKS];/* first unit masks */
} perf_event_t;

/*
 * umask options: uflags
 */
#define PERF_FL_DEFAULT	0x1	/* umask is default for group */

#define PCL_EVT(f, t, m)	\
	{ .name = #f,		\
	  .id = (f),		\
	  .type = (t),		\
	  .desc = #f,		\
	  .numasks = 0,		\
	  .modmsk = (m),	\
	  .ngrp = 0,		\
	  .umask_ovfl_idx = -1,	\
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

#define PERF_ATTR_HW (_PERF_ATTR_U|_PERF_ATTR_K|_PERF_ATTR_H)
#define PERF_ATTR_SW (_PERF_ATTR_U|_PERF_ATTR_K)

#include "events/perf_events.h"

pfmlib_pmu_t perf_event_support;
#define perf_nevents (perf_event_support.pme_count)

static perf_event_t *perf_pe = perf_static_events;
static perf_event_t  *perf_pe_free, *perf_pe_end;
static perf_umask_t *perf_um, *perf_um_free, *perf_um_end;
static int perf_pe_count, perf_um_count;

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

#define PERF_ALLOC_EVENT_COUNT	(512)
#define PERF_ALLOC_UMASK_COUNT	(1024)

/*
 * clone static event table into a  dynamic
 * event table
 *
 * Used for tracepoints
 */
static perf_event_t *
perf_table_clone(void)
{
	perf_event_t *addr;

	perf_pe_count = perf_nevents + PERF_ALLOC_EVENT_COUNT;

	addr = malloc(perf_pe_count * sizeof(perf_event_t));
	if (addr) {
		memcpy(addr, perf_static_events, perf_nevents * sizeof(perf_event_t));
		perf_pe_free = addr + perf_nevents;
		perf_pe_end = perf_pe_free + PERF_ALLOC_EVENT_COUNT;
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
static perf_event_t *
perf_table_alloc_event(void)
{
	perf_event_t *new_pe;

retry:
	if (perf_pe_free < perf_pe_end)
		return perf_pe_free++;

	perf_pe_count += PERF_ALLOC_EVENT_COUNT;
	
	new_pe = realloc(perf_pe, perf_pe_count * sizeof(perf_event_t));
	if (!new_pe) 
		return NULL;
	
	perf_pe_free = new_pe + (perf_pe_free - perf_pe);
	perf_pe_end = perf_pe_free + PERF_ALLOC_EVENT_COUNT;
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
static perf_umask_t *
perf_table_alloc_umask(void)
{
	perf_umask_t *new_um;

retry:
	if (perf_um_free < perf_um_end)
		return perf_um_free++;

	perf_um_count += PERF_ALLOC_UMASK_COUNT;
	
	new_um = realloc(perf_um, perf_um_count * sizeof(*new_um));
	if (!new_um) 
		return NULL;
	
	perf_um_free = new_um + (perf_um_free - perf_um);
	perf_um_end = perf_um_free + PERF_ALLOC_UMASK_COUNT;
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
	char *tracepoint_name;

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
		 * allocated space (with have no free)
 		 */
		if (!reuse_event)
			p = perf_table_alloc_event();

		if (!p)
			break;

		if (p)
			p->name = tracepoint_name = strdup(d1->d_name);

		if (!(p && p->name)) {
			closedir(dir2);
			err = -1;
			continue;
		}

		p->desc = "tracepoint";
		p->id = -1;
		p->type = PERF_TYPE_TRACEPOINT;
		p->umask_ovfl_idx = -1;
		p->modmsk = 0,
		p->ngrp = 1;

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
			um->grpid = 0;
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
			free(tracepoint_name);
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
pfm_perf_add_defaults(pfmlib_event_desc_t *e, unsigned int msk, uint64_t *umask)
{
	perf_event_t *ent;
	perf_umask_t *um;
	int i, j, k, added;

	k = e->nattrs;
	ent = perf_pe+e->event;

	for(i=0; msk; msk >>=1, i++) {

		if (!(msk & 0x1))
			continue;

		added = 0;

		for(j=0; j < ent->numasks; j++) {

			if (j < PERF_MAX_UMASKS) {
				um = &perf_pe[e->event].umasks[j];
			} else {
				um = perf_get_ovfl_umask(e->event);
			}
			if (um->grpid != i)
				continue;

			if (um->uflags & PERF_FL_DEFAULT) {
				DPRINT("added default %s for group %d\n", um->uname, i);

				*umask |= um->uid;

				e->attrs[k].id = j;
				e->attrs[k].ival = 0;
				e->attrs[k].type = PFM_ATTR_UMASK;
				k++;

				added++;
			}
		}
		if (!added) {
			DPRINT("no default found for event %s unit mask group %d\n", ent->name, i);
			return PFM_ERR_UMASK;
		}
	}
	e->nattrs = k;
	return PFM_SUCCESS;
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
	perf_event_t *ent;
	unsigned int msk, grpmsk;
	uint64_t umask = 0;
	int i, ret;

	grpmsk = (1 << perf_pe[e->event].ngrp)-1;

	ent = perf_pe + e->event;

	*codes = ent->id;
	*count = 1;

	e->fstr[0] = '\0';

	for(i=0; i < e->nattrs; i++) {
		a = e->attrs+i;
		if (a->type == PFM_ATTR_UMASK) {
			*codes |= ent->umasks[a->id].uid;

			msk = 1 << ent->umasks[a->id].grpid;
			/* umask cannot be combined in each group */
			if ((grpmsk & msk) == 0)
				return PFM_ERR_UMASK;
			grpmsk &= ~msk;
		}
	}

	/* check for missing default umasks */
	if (grpmsk) {
		ret = pfm_perf_add_defaults(e, grpmsk, &umask);
		if (ret != PFM_SUCCESS)
			return ret;
		*codes |= umask;
	}

	/*
	 * reorder all the attributes such that the fstr appears always
	 * the same regardless of how the attributes were submitted.
	 *
	 * cannot sort attr until after we have added the default umasks
	 */
	evt_strcat(e->fstr, "%s", ent->name);
	pfmlib_sort_attr(e);
	for(i=0; i < e->nattrs; i++) {
		if (e->attrs[i].type == PFM_ATTR_UMASK)
			evt_strcat(e->fstr, ":%s", ent->umasks[e->attrs[i].id].uname);
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
	evt_strcat(e->fstr, ":k=%lu", !!(attrs->plm & PFM_PLM0));
	evt_strcat(e->fstr, ":u=%lu", !!(attrs->plm & PFM_PLM3));
	evt_strcat(e->fstr, ":h=%lu", !!(attrs->plm & PFM_PLMH));

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
		info->name = modx(perf_mods, m, name);
		info->desc = modx(perf_mods, m, desc);
		info->equiv= NULL;
		info->code = m;
		info->type = modx(perf_mods, m, type);
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
	info->equiv = NULL;

	/* unit masks + modifiers */
	info->nattrs  = perf_pe[idx].numasks;
	info->nattrs += pfmlib_popcnt((unsigned long)perf_pe[idx].modmsk);

	return PFM_SUCCESS;
}

void
pfm_perf_terminate(void *this)
{
	perf_event_t *p;
	size_t i, j;

	if (!(perf_pe && perf_um))
		return;

	/*
	 * free tracepoints name + unit mask names
	 * which are dynamically allocated
	 */
	for (i=0; i < perf_nevents; i++) {
		p = &perf_pe[i];

		if (p->type != PERF_TYPE_TRACEPOINT)
			continue;

		/* cast to keep compiler happy, we are
		 * freeing the dynamically allocated clone
		 * table, not the static one. We do not want
		 * to create a specific data type
		 */
		free((void *)p->name);

		/*
		 * first PERF_MAX_UMASKS are pre-allocated
		 * the rest is in a separate dynamic table
		 */
		for (j=0; j < p->numasks; j++) {
			if (j == PERF_MAX_UMASKS)
				break;
			free((void *)p->umasks[j].uname);
		}
	}
	/*
	 * perf_pe is systematically allocated
	 */
	free(perf_pe);
	perf_pe = NULL;
	perf_pe_free = perf_pe_end = NULL;

	if (perf_um) {
		int n;
		/*
		 * free the dynamic umasks' uname
		 */
		n = perf_um_free - perf_um;
		for(i=0; i < n; i++) {
			free((void *)(perf_um[i].uname));
		}
		free(perf_um);
		perf_um = NULL;
		perf_um_free = perf_um_end = NULL;
	}
}

static int
pfm_perf_validate_table(void *this, FILE *fp)
{
	const char *name = perf_event_support.name;
	perf_umask_t *um;
	int i, j;
	int error = 0;

	for(i=0; i < perf_event_support.pme_count; i++) {

		if (!perf_pe[i].name) {
			fprintf(fp, "pmu: %s event%d: :: no name (prev event was %s)\n", name, i,
			i > 1 ? perf_pe[i-1].name : "??");
			error++;
		}

		if (!perf_pe[i].desc) {
			fprintf(fp, "pmu: %s event%d: %s :: no description\n", name, i, perf_pe[i].name);
			error++;
		}

		if (perf_pe[i].type < PERF_TYPE_HARDWARE || perf_pe[i].type >= PERF_TYPE_MAX) {
			fprintf(fp, "pmu: %s event%d: %s :: invalid type\n", name, i, perf_pe[i].name);
			error++;
		}

		if (perf_pe[i].numasks >= PERF_MAX_UMASKS && perf_pe[i].umask_ovfl_idx == -1) {
			fprintf(fp, "pmu: %s event%d: %s :: numasks too big (<%d)\n", name, i, perf_pe[i].name, PERF_MAX_UMASKS);
			error++;
		}

		if (perf_pe[i].numasks < PERF_MAX_UMASKS && perf_pe[i].umask_ovfl_idx != -1) {
			fprintf(fp, "pmu: %s event%d: %s :: overflow umask idx defined but not needed (<%d)\n", name, i, perf_pe[i].name, PERF_MAX_UMASKS);
			error++;
		}

		if (perf_pe[i].numasks && perf_pe[i].ngrp == 0) {
			fprintf(fp, "pmu: %s event%d: %s :: ngrp cannot be zero\n", name, i, perf_pe[i].name);
			error++;
		}

		if (perf_pe[i].numasks == 0 && perf_pe[i].ngrp) {
			fprintf(fp, "pmu: %s event%d: %s :: ngrp must be zero\n", name, i, perf_pe[i].name);
			error++;
		}

		for(j= 0; j < perf_pe[i].numasks; j++) {

			if (j < PERF_MAX_UMASKS){
				um = perf_pe[i].umasks+j;
			} else {
				um = perf_get_ovfl_umask(i);
				um += j - PERF_MAX_UMASKS;
			}
			if (!um->uname) {
				fprintf(fp, "pmu: %s event%d: %s umask%d :: no name\n", name, i, perf_pe[i].name, j);
				error++;
			}

			if (!um->udesc) {
				fprintf(fp, "pmu: %s event%d:%s umask%d: %s :: no description\n", name, i, perf_pe[i].name, j, um->uname);
				error++;
			}

			if (perf_pe[i].ngrp && um->grpid >= perf_pe[i].ngrp) {
				fprintf(fp, "pmu: %s event%d: %s umask%d: %s :: invalid grpid %d (must be < %d)\n", name, i, perf_pe[i].name, j, um->uname, um->grpid, perf_pe[i].ngrp);
				error++;
			}
		}

		/* check for excess unit masks */
		for(; j < PERF_MAX_UMASKS; j++) {
			if (perf_pe[i].umasks[j].uname || perf_pe[i].umasks[j].udesc) {
				fprintf(fp, "pmu: %s event%d: %s :: numasks (%d) invalid more events exists\n", name, i, perf_pe[i].name, perf_pe[i].numasks);
				error++;
			}
		}
	}
	return error ? PFM_ERR_INVAL : PFM_SUCCESS;
}

pfmlib_pmu_t perf_event_support={
	.desc			= "perf_events generic PMU",
	.name			= "perf",
	.pmu			= PFM_PMU_PERF_EVENT,
	.pme_count		= PME_PERF_EVENT_COUNT,
	.max_encoding		= 1,
	.pmu_detect		= pfm_perf_detect,
	.pmu_init		= pfm_perf_init,
	.pmu_terminate		= pfm_perf_terminate,
	.get_event_encoding	= pfm_perf_get_encoding,
	.get_event_first	= pfm_perf_get_event_first,
	.get_event_next		= pfm_perf_get_event_next,
	.event_is_valid		= pfm_perf_event_is_valid,
	.get_event_perf_type	= pfm_perf_get_event_perf_type,
	.get_event_info		= pfm_perf_get_event_info,
	.get_event_attr_info	= pfm_perf_get_event_attr_info,
	.validate_table		= pfm_perf_validate_table,
};
