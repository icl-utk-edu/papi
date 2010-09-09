/*
 * pfmlib_sparc.c : 	support for SPARC processors
 * 
 * Based upon gen_powerpc code which is:
 * Copyright (C) IBM Corporation, 2007.  All rights reserved.
 * Contributed by Corey Ashford (cjashfor@us.ibm.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_sparc_priv.h"

#ifdef CONFIG_PFMLIB_OS_LINUX
/*
 * getl(): our own equivalent to GNU getline() extension.
 * This avoids a dependency on having a C library with
 * support for getline().
 */
static int
getl(char **buffer, size_t *len, FILE *fp)
{
#define	GETL_DFL_LEN	32
	char *b;
	int c, i = 0;
	size_t maxsz, maxi, d;

	if (!len || !fp || !buffer)
		return -1;

	b = *buffer;

	if (!b)
		*len = 0;

	maxsz = *len;
	maxi = maxsz - 2;

	while ((c = fgetc(fp)) != EOF) {
		if (maxsz == 0 || i == maxi) {
			if (maxsz == 0)
				maxsz = GETL_DFL_LEN;
			else
				maxsz <<= 1;

			if (*buffer)
				d = &b[i] - *buffer;
			else
				d = 0;

			*buffer = realloc(*buffer, maxsz);
			if (!*buffer)
				return -1;

			b = *buffer + d;
			maxi = maxsz - d - 2;
			i = 0;
			*len = maxsz;
		}
		b[i++] = c;
		if (c == '\n')
			break;
	}
	b[i] = '\0';
	return c != EOF ? 0 : -1;
}

/*
 * helper function to retrieve one value from /proc/cpuinfo
 * for internal libpfm use only
 * attr: the attribute (line) to look for
 * ret_buf: a buffer to store the value of the attribute (as a string)
 * maxlen : number of bytes of capacity in ret_buf
 *
 * ret_buf is null terminated.
 *
 * Return:
 * 	0 : attribute found, ret_buf populated
 * 	-1: attribute not found
 */

static int
pfmlib_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen)
{
	FILE *fp = NULL;
	int ret = -1;
	size_t attr_len, buf_len = 0;
	char *p, *value = NULL;
	char *buffer = NULL;

	if (attr == NULL || ret_buf == NULL || maxlen < 1)
		return -1;

	attr_len = strlen(attr);

	fp = fopen("/proc/cpuinfo", "r");
	if (fp == NULL)
		return -1;

	while(getl(&buffer, &buf_len, fp) != -1){

		/* skip  blank lines */
		if (*buffer == '\n')
			continue;

		p = strchr(buffer, ':');
		if (p == NULL)
			goto error;

		/*
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0';
		value = p+2;

		value[strlen(value)-1] = '\0';

		if (!strncmp(attr, buffer, attr_len))
			break;
	}
	strncpy(ret_buf, value, maxlen-1);
	ret_buf[maxlen-1] = '\0';
	ret = 0;
error:
	free(buffer);
	fclose(fp);
	return ret;
}
#else
static int
pfmlib_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen)
{
	return -1;
}
#endif

static pfm_pmu_t
pmu_name_to_pmu_type(char *name)
{
	if (!strcmp(name, "ultra12"))
		return PFM_PMU_SPARC_ULTRA12;
	if (!strcmp(name, "ultra3"))
		return PFM_PMU_SPARC_ULTRA3;
	if (!strcmp(name, "ultra3i"))
		return PFM_PMU_SPARC_ULTRA3I;
	if (!strcmp(name, "ultra3+"))
		return PFM_PMU_SPARC_ULTRA3PLUS;
	if (!strcmp(name, "ultra4+"))
		return PFM_PMU_SPARC_ULTRA4PLUS;
	if (!strcmp(name, "niagara2"))
		return PFM_PMU_SPARC_NIAGARA2;
	if (!strcmp(name, "niagara"))
		return PFM_PMU_SPARC_NIAGARA1;
	return PFM_PMU_NONE;
}

int pfm_sparc_detect(void)
{
	int ret;
	char buffer[32];

	ret = pfmlib_getcpuinfo_attr("pmu", buffer, sizeof(buffer));
	if (ret == -1)
		return PFM_ERR_NOTSUPP;
	return pmu_name_to_pmu_type(buffer);
}

void
pfm_sparc_display_reg(int code, char *fstr)
{
	__pfm_vbprintf("[0x%x] %s\n", code, fstr);
}

int
pfm_sparc_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{
	const sparc_entry_t *pe = this_pe(this);

	*count = 1;
	*codes = pe[e->event].code;
	evt_strcat(e->fstr, "%s", pe[e->event].name);

	pfm_sparc_display_reg(pe[e->event].code, e->fstr);

	return PFM_SUCCESS;
}

int
pfm_sparc_get_event_first(void *this)
{
	return 0;
}

int
pfm_sparc_get_event_next(void *this, int idx)
{
	pfmlib_pmu_t *p = this;

	if (idx >= (p->pme_count-1))
		return -1;

	return idx+1;
}

int
pfm_sparc_event_is_valid(void *this, int pidx)
{
	pfmlib_pmu_t *p = this;
	return pidx >= 0 && pidx < p->pme_count;
}

int
pfm_sparc_get_event_perf_type(void *this, int pidx)
{
	return PERF_TYPE_RAW;
}

int
pfm_sparc_validate_table(void *this, FILE *fp)
{

	pfmlib_pmu_t *pmu = this;
	const sparc_entry_t *pe = this_pe(this);
	int i, j, error = 0;

	for(i=0; i < pmu->pme_count; i++) {
		if (!pe[i].name) {
			fprintf(fp, "pmu: %s event%d: :: no name (prev event was %s)\n", pmu->name, i,
			i > 1 ? pe[i-1].name : "??");
			error++;
		}
		if (!pe[i].desc) {
			fprintf(fp, "pmu: %s event%d: %s :: no description\n", pmu->name, i, pe[i].name);
			error++;
		}
		for(j=i+1; j < pmu->pme_count; j++) {
			if (pe[i].code == pe[j].code) {
				fprintf(fp, "pmu: %s event%d: %s code: 0x%x is duplicated in event%d : %s\n", pmu->name, i, pe[i].name, pe[i].code, j, pe[j].name);
				error++;
			}
		}
	}
	return error ? PFM_ERR_INVAL : PFM_SUCCESS;
}

int
pfm_sparc_get_event_attr_info(void *this, int pidx, int attr_idx, pfm_event_attr_info_t *info)
{
	return PFM_ERR_INVAL;
}

int
pfm_sparc_get_event_info(void *this, int idx, pfm_event_info_t *info)
{

	const sparc_entry_t *pe = this_pe(this);
	/*
	 * pmu and idx filled out by caller
	 */
	info->name  = pe[idx].name;
	info->desc  = pe[idx].desc;
	info->code  = pe[idx].code;
	info->equiv = NULL;

	/* no attributes defined for ARM yet */
	info->nattrs  = 0;

	return PFM_SUCCESS;
}
