#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_perf_event_priv.h"
#include "pfmlib_arm_priv.h"

typedef union {
	uint64_t val;
	struct {
		unsigned long unc_event:8;	/* event code */
		unsigned long unc_umask:8;	/* unit mask */
		unsigned long unc_res1:1;	/* reserved */
		unsigned long unc_rst:1;	/* reset */
		unsigned long unc_edge:1;	/* edge detect */
		unsigned long unc_res2:3;	/* reserved */
		unsigned long unc_en:1;		/* enable */
		unsigned long unc_inv:1;	/* invert counter mask */
		unsigned long unc_thres:8;	/* counter mask */
		unsigned long unc_res3:32;	/* reserved */
	} com; /* covers common fields for DMC/L3C */
} tx2_unc_data_t;

static void
display_reg(void *this, pfmlib_event_desc_t *e, tx2_unc_data_t reg);
static void
display_com(void *this, pfmlib_event_desc_t *e, void *val);
static int
find_pmu_type_by_name(const char *name);

int
pfm_tx2_unc_get_event_encoding(void *this, pfmlib_event_desc_t *e)
{
	//from pe field in for the uncore, get the array with all the event defs
	const arm_entry_t *event_list = this_pe(this);
	tx2_unc_data_t reg;
	//get code for the event from the table
	reg.val = event_list[e->event].code;
	//pass the data back to the caller
	e->codes[0] = reg.val;
	e->count = 1;
	evt_strcat(e->fstr, "%s", event_list[e->event].name);
	display_reg(this, e, reg);
	return PFM_SUCCESS;
}

int
pfm_tx2_unc_get_perf_encoding(void *this, pfmlib_event_desc_t *e)
{
	pfmlib_pmu_t *pmu = this;
	struct perf_event_attr *attr = e->os_data;
	tx2_unc_data_t reg;
	int ret;

	if (!pmu->get_event_encoding[PFM_OS_NONE])
		return PFM_ERR_NOTSUPP;

	ret = pmu->get_event_encoding[PFM_OS_NONE](this, e);
	if (ret != PFM_SUCCESS)
		return ret;
	//get pmu type to probe
	ret = find_pmu_type_by_name(pmu->perf_name);
	if (ret < 0)
		return ret;

	attr->type = ret;
	//get code to provide to the uncore pmu probe
	reg.val = e->codes[0];
	attr->config = reg.val;

	// if needed, can use attr->config1 or attr->config2 for extra info from event structure defines e->codes[i]

	// uncore measures at all priv levels
	attr->exclude_hv = 0;
	attr->exclude_kernel = 0;
	attr->exclude_user = 0;

	return PFM_SUCCESS;
}


static void
display_reg(void *this, pfmlib_event_desc_t *e, tx2_unc_data_t reg)
{
	pfmlib_pmu_t *pmu = this;
	if (pmu->display_reg)
		pmu->display_reg(this, e, &reg);
	else
		display_com(this, e, &reg);
}

static void
display_com(void *this, pfmlib_event_desc_t *e, void *val)
{
	const arm_entry_t *pe = this_pe(this);
	tx2_unc_data_t *reg = val;

	__pfm_vbprintf("[UNC=0x%"PRIx64" event=0x%x umask=0x%x en=%d "
		       "inv=%d edge=%d thres=%d] %s\n",
			reg->val,
			reg->com.unc_event,
			reg->com.unc_umask,
			reg->com.unc_en,
			reg->com.unc_inv,
			reg->com.unc_edge,
			reg->com.unc_thres,
			pe[e->event].name);
}

static int
find_pmu_type_by_name(const char *name)
{
	char filename[PATH_MAX];
	FILE *fp;
	int ret, type;

	if (!name)
		return PFM_ERR_NOTSUPP;

	sprintf(filename, "/sys/bus/event_source/devices/%s/type", name);

	fp = fopen(filename, "r");
	if (!fp)
		return PFM_ERR_NOTSUPP;

	ret = fscanf(fp, "%d", &type);
	if (ret != 1)
		type = PFM_ERR_NOTSUPP;

	fclose(fp);

	return type;
}

