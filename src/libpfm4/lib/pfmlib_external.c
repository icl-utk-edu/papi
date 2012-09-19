/* So we're stuck on the question, how do house external events...
   libpfm uses a compile-time list of pmu's

   sometime

   */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>

#include <perfmon/pfmlib.h>

#include "pfmlib_priv.h"
#include "pfmlib_external.h"

int 
pfm_gen_external_detect( void *this )
{
  return PFM_SUCCESS;
}

int 
pfm_gen_external_get_event_first( void *this )
{
  return 0;
}

int 
pfm_gen_external_get_event_next( void* this, int idx )
{
  pfmlib_pmu_t* p = this;
  if (idx >= (p->pme_count-1) )
	return -1;
  return idx + 1;
}

int 
pfm_gen_external_get_event_info( void *this, int pidx, pfm_event_info_t *info)
{
  const pme_external_entry_t* pe= this_pe(this);

  info->name = pe[pidx].pme_name;
  info->desc = pe[pidx].pme_long_desc;

  return PFM_SUCCESS;
}

int 
pfm_gen_external_event_is_valid( void *this, int pidx )
{
  pfmlib_pmu_t *p = this;
  return pidx >= 0 && pidx < p->pme_count;
}

	int 		 pfm_gen_external_pmu_init(void *this)	{
	  
	}/* optional */
	void		 pfm_gen_external_pmu_terminate(void *this) {
	  
	}/* optional */
	int		 pfm_gen_external_get_event_nattrs(void *this, int pidx) {

	}
	int		 pfm_gen_external_event_is_valid(void *this, int pidx) {

	}

	int		 (*get_event_attr_info)(void *this, int pidx, int umask_idx, pfm_event_attr_info_t *info);
	int		 (*get_event_encoding[PFM_OS_MAX])(void *this, pfmlib_event_desc_t *e);

	void		 (*validate_pattrs[PFM_OS_MAX])(void *this, pfmlib_event_desc_t *e);
	int		 pfm_gen_external_validate_table(void *this, FILE *fp) {

	}
