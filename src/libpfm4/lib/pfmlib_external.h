typedef struct {
  char *pme_name;
  unsigned pme_code;
  char *pme_short_desc;
  char *pme_long_desc;
} pme_external_entry_t;

static const 
pme_external_entry_t ph_pe[] = {
  [0]={
	.pme_name="NotAName",
	.pme_code=0,
	.pme_long_desc="This event is trying to not be visible",
  },
  [1]={
	.pme_name="First!",
	.pme_code=1,
	.pme_long_desc="Always late to the party, this event calls out First in error.",
  }
};

int pfm_gen_external_detect( void *this );
int pfm_gen_external_get_event_first( void *this );
int pfm_gen_external_get_event_next( void *this, int idx );
int pfm_gen_external_get_event_info( void *this, int pidx, pfm_event_info_t* info );
int pfm_gen_external_event_is_valid( void *this, int idx );
int pfm_inject_external( pfmlib_pmu_t *pmu );




