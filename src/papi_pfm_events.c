/*
* File:    papi_pfm_events.c
* CVS:     $Id$
* Author:  Dan Terpstra: blantantly extracted from Phil's perfmon.c
*          mucci@cs.utk.edu
*/

/* TODO LIST:
    - Events for all platforms
xxx - Derived events for all platforms
xxx - hwd_ntv_name_to_code
xxx - Make native map carry major events, not umasks
xxx - Enum event uses native_map not pfm()
xxx - bits_to_info uses native_map not pfm()
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "papi_pfm_events.h"

extern papi_vector_t MY_VECTOR;

/* These routines are defined externally for PERFCTR_PFM_EVENTS == TRUE, or
    internally for PERFCTR_PFM_EVENTS == FALSE */
extern int _papi_pfm_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits);
extern int _papi_pfm_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count);

/* Globals declared extern elsewhere */

volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];

/* NOTE: PAPI stores umask info in a variable sized (16 bit?) bitfield.
    Perfmon2 stores umask info in a large (48 element?) array of values.
    Native event encodings for perfmon2 contain array indices
    encoded as bits in this bitfield. These indices must be converted
    into a umask value before programming the counters. For Perfmon,
    this is done by converting back to an array of values; for 
    perfctr, it must be done by looking up the values.
*/

/* This routine is used to step through all possible combinations of umask
    values. It assumes that mask contains a valid combination of array indices
    for this event. */
static inline int encode_native_event_raw(unsigned int event, unsigned int mask)
{
  unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
  SUBDBG("Old native index was 0x%08x with 0x%08x mask\n",tmp,mask);
  tmp = tmp | (mask << PAPI_NATIVE_UMASK_SHIFT);
  SUBDBG("New encoding is 0x%08x\n",tmp|PAPI_NATIVE_MASK);
  return (int)(tmp|PAPI_NATIVE_MASK);
}

/* This routine converts array indices contained in the mask_values array
    into bits in the umask field that is OR'd into the native event code.
    These bits are NOT the mask values themselves, but indices into an array
    of mask values contained in the native event table. */
static inline int encode_native_event(unsigned int event, unsigned int num_mask, unsigned int *mask_values)
{
  unsigned int i;
  unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
  SUBDBG("Native base event is 0x%08x with %d masks\n",tmp,num_mask);
  for (i=0;i<num_mask;i++) {
      SUBDBG("Mask index is 0x%08x\n",mask_values[i]);
      tmp = tmp | ((1 << mask_values[i]) << PAPI_NATIVE_UMASK_SHIFT);
  }
  SUBDBG("Full native encoding is 0x%08x\n",tmp|PAPI_NATIVE_MASK);
  return (int)(tmp|PAPI_NATIVE_MASK);
}

static int setup_preset_term(int *native, pfmlib_event_t *event)
{
    /* It seems this could be greatly simplified. If impl_cnt is non-zero,
	the event lives on a counter. Therefore the entire routine could be:
	if (impl_cnt!= 0) encode_native_event.
	Am I wrong?
    */
  pfmlib_regmask_t impl_cnt, evnt_cnt;
  unsigned int n;
  int ret;
  unsigned int j;

  /* find out which counters it lives on */
  if ((ret = pfm_get_event_counters(event->event,&evnt_cnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_event_counters(%d,%p): %s",event->event,&evnt_cnt,pfm_strerror(ret));
      return(PAPI_EBUG);
    }
  if ((ret = pfm_get_impl_counters(&impl_cnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_impl_counters(%p): %s", &impl_cnt, pfm_strerror(ret));
      return(PAPI_EBUG);
    }

  /* Make sure this event lives on some counter, if so, put in the description. If not, BUG */
  if ((ret = pfm_get_num_counters(&n)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_num_counters(%d): %s", n, pfm_strerror(ret));
      return(PAPI_EBUG);
    }

  for (j=0;n;j++)
    {
      if (pfm_regmask_isset(&impl_cnt, j))
	{
	  n--;
	  if (pfm_regmask_isset(&evnt_cnt,j))
	    {
	      *native = encode_native_event(event->event,event->num_masks,event->unit_masks);
	      return(PAPI_OK);
	    }
	}
    }

  PAPIERROR("PAPI preset 0x%08x PFM event %d did not have any available counters", event->event, j);
  return(PAPI_ENOEVNT);
}

/*  Trims blank space from both ends of a string (in place).
    Returns pointer to new start address */
static inline char *trim_string(char *in)
{
  int len, i = 0;
  char *start = in;

  if (in == NULL)
    return(in);
  len = (int)strlen(in);
  if (len == 0)
    return(in);
  /* Trim left */
  while (i < len)
    {
      if (isblank(in[i]))
	{
	  in[i] = '\0';
	  start++;
	}
      else
	break;
      i++;
    }
  /* Trim right */
  i = (int)strlen(start) - 1;
  while (i >= 0)
    {
      if (isblank(start[i]))
	start[i] = '\0';
      else
	break;
      i--;
    }
  return(start);
}


/*  Calls trim_string to remove blank space;
    Removes paired punctuation delimiters from
    beginning and end of string. If the same punctuation 
    appears first and last (quotes, slashes) they are trimmed;
    Also checks for the following pairs: () <> {} [] */
static inline char *trim_note(char *in)
{
  int len;
  char *note, start, end;

  note = trim_string(in);
  if (note != NULL) {
    len = (int)strlen(note);
    if (len > 0) {
      if (ispunct(*note)) {
	start = *note;
	end = note[len-1];
	if ((start == end)
	    || ((start == '(') && (end == ')'))
	    || ((start == '<') && (end == '>'))
	    || ((start == '{') && (end == '}'))
	    || ((start == '[') && (end == ']'))) {
	  note[len-1] = '\0';
	  *note = '\0';
	  note++;
	}
      }
    }
  }
  return(note);
}

static inline int find_preset_code(char *tmp, int *code)
{
  int i = 0;
  extern hwi_presets_t _papi_hwi_presets;
  while (_papi_hwi_presets.info[i].symbol != NULL)
    {
      if (strcasecmp(tmp,_papi_hwi_presets.info[i].symbol) == 0)
	{
	  *code = (int)(i|PAPI_PRESET_MASK);
	  return(PAPI_OK);
	}
      i++;
    }
  return(PAPI_EINVAL);
}

/* Look for an event file 'name' in a couple common locations.
   Return a valid file handle if found */
static FILE *open_event_table(char *name)
{
  FILE *table;

  SUBDBG("Opening %s\n",name);
  table = fopen(name,"r");
  if (table == NULL)
  {
    SUBDBG("Open %s failed, trying ./%s.\n",name,PAPI_EVENT_FILE);
    sprintf(name,"%s",PAPI_EVENT_FILE);
    table = fopen(name,"r");
  }
  if (table == NULL)
  {
    SUBDBG("Open ./%s failed, trying ../%s.\n",name,PAPI_EVENT_FILE);
    sprintf(name,"../%s",PAPI_EVENT_FILE);
    table = fopen(name,"r");
  }
  if (table) SUBDBG("Open %s succeeded.\n",name);
  return (table);
}

/* parse a single line from either a file or character table
   Strip trailing <cr>; return 0 if empty */
static int get_event_line(char *line, FILE *table, char **tmp_perfmon_events_table)
{
  int ret;
  int i;

  if (table) {
    if(fgets(line, LINE_MAX, table)) {
      ret = 1;
      i = (int)strlen(line);
      if (line[i-1] == '\n')
      line[i-1] = '\0';
    } else ret = 0;
  } else {
    for (i=0; **tmp_perfmon_events_table && **tmp_perfmon_events_table != '\n'; i++) {
      line[i] = **tmp_perfmon_events_table;
      (*tmp_perfmon_events_table)++;
    }
    if (**tmp_perfmon_events_table == '\n') {
      (*tmp_perfmon_events_table)++;
    }
    line[i] = '\0';
    ret = **tmp_perfmon_events_table;
  }
  return(ret);
}

/* Static version of the events file. */
#if defined(STATIC_PAPI_EVENTS_TABLE)
#include "papi_events_table.h"
#else
  static char *papi_events_table = NULL;
#endif

/* #define SHOW_LOADS */

static int load_preset_table(char *pmu_str, int pmu_type, pfm_preset_search_entry_t *here)
{
  pfmlib_event_t event;
  char pmu_name[PAPI_MIN_STR_LEN];
  char line[LINE_MAX];
  char name[PATH_MAX] = "builtin papi_events_table";
  char *tmp_papi_events_table = NULL;
  char *tmpn;
  FILE *table;
  int line_no = 1, derived = 0, insert = 0, preset = 0;
  int get_presets = 0;   /* only get PRESETS after CPU is identified */
  int found_presets = 0; /* only terminate search after PRESETS are found */
						 /* this allows support for synonyms for CPU names */

#ifdef SHOW_LOADS
  SUBDBG("%p\n",here);
#endif

  /* copy the pmu identifier, stripping commas if found */
  tmpn = pmu_name;
  while(*pmu_str) {
    if(*pmu_str != ',') *tmpn++ = *pmu_str;
	pmu_str++;
  }
  *tmpn = '\0';

  /* make sure these events are supported before adding them */
  if (pfm_get_cycle_event(&event) != PFMLIB_ERR_NOTSUPP) {
    here[insert].preset = (int)PAPI_TOT_CYC;
    here[insert++].derived = -1;
  }
  if (pfm_get_inst_retired_event(&event) != PFMLIB_ERR_NOTSUPP) {
    here[insert].preset = (int)PAPI_TOT_INS;
    here[insert++].derived = -1;
  }

  /* try the environment variable first */
  if ((tmpn = getenv("PAPI_PERFMON_EVENT_FILE")) && (strlen(tmpn) != 0)) {
    sprintf(name,"%s",tmpn);
    table = fopen(name,"r");
  }
  /* if no valid environment variable, look for built-in table */
  else if (papi_events_table) {
	  tmp_papi_events_table = papi_events_table;
	  table = NULL;
  }
  /* if no env var and no built-in, search for default file */
  else {
#ifdef PAPI_DATADIR
    sprintf(name,"%s/%s",PAPI_DATADIR,PAPI_EVENT_FILE);
#else
    sprintf(name,"%s",PAPI_EVENT_FILE);
#endif
	table = open_event_table(name);
  }
  /* if no valid file or built-in table, bail */
  if (table == NULL && tmp_papi_events_table == NULL) {
    PAPIERROR("fopen(%s): %s, please set the PAPI_PERFMON_EVENT_FILE env. variable",name,strerror(errno));
    return(PAPI_ESYS);
  }

  /* at this point either a valid file pointer or built-in table pointer */
  while (get_event_line(line, table, &tmp_papi_events_table)) {
      char *t;
	  int i;
      t = trim_string(strtok(line,","));
      if ((t == NULL) || (strlen(t) == 0))
	continue;
      if (t[0] == '#')
	{
/*	  SUBDBG("Comment found on line %d\n",line_no); */
	  goto nextline;
	}
      else if (strcasecmp(t,"CPU") == 0)
	{
#ifdef SHOW_LOADS
	  SUBDBG("CPU token found on line %d\n",line_no);
#endif
	  if (get_presets != 0 && found_presets != 0)
	    {
#ifdef SHOW_LOADS
	      SUBDBG("Ending preset scanning at line %d of %s.\n",line_no,name);
#endif
	      goto done;
	    }
	  t = trim_string(strtok(NULL,","));
	  if ((t == NULL) || (strlen(t) == 0))
	    {
	      PAPIERROR("Expected name after CPU token at line %d of %s -- ignoring",line_no,name);
	      goto nextline;
	    }
#ifdef SHOW_LOADS
	  SUBDBG("Examining CPU (%s) vs. (%s)\n",t,pmu_name);
#endif
	  if (strcasecmp(t, pmu_name) == 0)
	    {
	      int type;

//#ifdef SHOW_LOADS
	      SUBDBG("Found CPU %s at line %d of %s.\n",t,line_no,name);
//#endif
	      t = trim_string(strtok(NULL,","));
	      if ((t == NULL) || (strlen(t) == 0))
		{
#ifdef SHOW_LOADS
		  SUBDBG("No additional qualifier found, matching on string.\n");
#endif
		  get_presets = 1;
		}
	      else if ((sscanf(t,"%d",&type) == 1) && (type == pmu_type))
		{
#ifdef SHOW_LOADS
		  SUBDBG("Found CPU %s type %d at line %d of %s.\n",pmu_name,type,line_no,name);
#endif
		  get_presets = 1;
		}
	      else
                {
#ifdef SHOW_LOADS
		  SUBDBG("Additional qualifier match failed %d vs %d.\n",pmu_type,type);
#endif
		}	      
	    }
	}
      else if (strcasecmp(t,"PRESET") == 0)
	{
#ifdef SHOW_LOADS
	  SUBDBG("PRESET token found on line %d\n",line_no);
#endif
	  if (get_presets == 0)
	    goto nextline;
	  found_presets = 1;
	  t = trim_string(strtok(NULL,","));
	  if ((t == NULL) || (strlen(t) == 0))
	    {
	      PAPIERROR("Expected name after PRESET token at line %d of %s -- ignoring",line_no,name);
	      goto nextline;
	    }
#ifdef SHOW_LOADS
	  SUBDBG("Examining preset %s\n",t);
#endif
	  if (find_preset_code(t,&preset) != PAPI_OK)
	    {
	      PAPIERROR("Invalid preset name %s after PRESET token at line %d of %s -- ignoring",t,line_no,name);
	      goto nextline;
	    }
#ifdef SHOW_LOADS
	  SUBDBG("Found 0x%08x for %s\n",preset,t);
#endif
	  t = trim_string(strtok(NULL,","));
	  if ((t == NULL) || (strlen(t) == 0))
	    {
	      PAPIERROR("Expected derived type after PRESET token at line %d of %s -- ignoring",line_no,name);
	      goto nextline;
	    }
#ifdef SHOW_LOADS
	  SUBDBG("Examining derived %s\n",t);
#endif
	  if (_papi_hwi_derived_type(t,&derived) != PAPI_OK)
	    {
	      PAPIERROR("Invalid derived name %s after PRESET token at line %d of %s -- ignoring",t,line_no,name);
	      goto nextline;
	    }
#ifdef SHOW_LOADS
	  SUBDBG("Found %d for %s\n",derived,t);
	  SUBDBG("Adding 0x%x,%d to preset search table.\n",preset,derived);
#endif
	  here[insert].preset = preset;
	  here[insert].derived = derived;

	  /* Derived support starts here */
	  /* Special handling for postfix */
	  if (derived == DERIVED_POSTFIX) {
	    t = trim_string(strtok(NULL,","));
	    if ((t == NULL) || (strlen(t) == 0)) {
	      PAPIERROR("Expected Operation string after derived type DERIVED_POSTFIX at line %d of %s -- ignoring",line_no,name);
	      goto nextline;
	    }
#ifdef SHOW_LOADS
	    SUBDBG("Saving PostFix operations %s\n",t);
#endif
	    here[insert].operation = strdup(t);
	  }
	  /* All derived terms collected here */
	  i = 0;
	  do {
	    t = trim_string(strtok(NULL,","));
	    if ((t == NULL) || (strlen(t) == 0)) break;
	    if (strcasecmp(t,"NOTE") == 0) break;
	    here[insert].findme[i] = strdup(t);
#ifdef SHOW_LOADS
	    SUBDBG("Adding term (%d) %s to preset event 0x%x.\n",i,t,preset);
#endif
	  } while (++i < MAX_COUNTER_TERMS);
	  /* End of derived support */

	  if (i == 0) {
	    PAPIERROR("Expected PFM event after DERIVED token at line %d of %s -- ignoring",line_no,name);
	    goto nextline;
	  }
	  if (i == MAX_COUNTER_TERMS)
	    t = trim_string(strtok(NULL,","));

	  /* Handle optional NOTEs */
	  if (t && (strcasecmp(t,"NOTE") == 0)) {
#ifdef SHOW_LOADS
	    SUBDBG("%s found on line %d\n",t,line_no);
#endif
	    t = trim_note(strtok(NULL,"")); /* read the rest of the line */
	    if ((t == NULL) || (strlen(t) == 0))
	      PAPIERROR("Expected Note string at line %d of %s\n",line_no,name);
	    else {
	      here[insert].note = strdup(t);
#ifdef SHOW_LOADS
	      SUBDBG("NOTE: --%s-- found on line %d\n",t,line_no);
#endif
	    }
	  }

	  insert++;
	  SUBDBG("# events inserted: --%d-- \n",insert);
	}
      else
	{
	  PAPIERROR("Unrecognized token %s at line %d of %s -- ignoring",t,line_no,name);
	  goto nextline;
	}
    nextline:
      line_no++;
    }
 done:
  if (table)
    fclose(table);
  return(PAPI_OK);
}

/* Frees memory for all the strdup'd char strings in a preset string array.
    Assumes the array is initialized to 0 and has at least one 0 entry at the end.
    free()ing a NULL pointer is a NOP. */
static void free_preset_table(pfm_preset_search_entry_t *here)
{
    int i = 0, j;
    while (here[i].preset) {
      for (j=0;j<MAX_COUNTER_TERMS;j++)
	free(here[i].findme[j]);
      free(here[i].operation);
      free(here[i].note);
      i++;
    }
}

static void free_notes(hwi_dev_notes_t *here)
{
    int i = 0;
    while (here[i].event_code) {
      free(here[i].dev_note);
      i++;
    }
}

static int generate_preset_search_map(hwi_search_t **maploc, hwi_dev_notes_t **noteloc, pfm_preset_search_entry_t *strmap)
{
	int k = 0, term, ret;
        unsigned int i = 0, j = 0;
	hwi_search_t *psmap;
	hwi_dev_notes_t *notemap;
	pfmlib_event_t event;

	/* Count up the proposed presets */
	while (strmap[i].preset)
		i++;
	SUBDBG("generate_preset_search_map(%p,%p,%p) %d proposed presets\n",maploc,noteloc,strmap,i);
	i++;

	/* Add null entry */
	psmap = (hwi_search_t *)malloc(i*sizeof(hwi_search_t));
	notemap = (hwi_dev_notes_t *)malloc(i*sizeof(hwi_dev_notes_t));
	if ((psmap == NULL) || (notemap == NULL))
		return(PAPI_ENOMEM);
	memset(psmap,0x0,i*sizeof(hwi_search_t));
	memset(notemap,0x0,i*sizeof(hwi_dev_notes_t));

	i = 0;
	while (strmap[i].preset) {
	  if ((strmap[i].preset == (int)PAPI_TOT_CYC) && (strmap[i].derived == -1)) {
			SUBDBG("pfm_get_cycle_event(%p)\n",&event);
			if ((ret = pfm_get_cycle_event(&event)) == PFMLIB_SUCCESS) {
				if (setup_preset_term(&psmap[j].data.native[0], &event) == PAPI_OK) {
					psmap[j].event_code = (unsigned int)strmap[i].preset;
					psmap[j].data.derived = NOT_DERIVED;
					psmap[j].data.native[1] = PAPI_NULL;
					j++;
				}
			}
			else
				SUBDBG("pfm_get_cycle_event(%p): %s\n",&event, pfm_strerror(ret));
		}
	  else if ((strmap[i].preset == (int)PAPI_TOT_INS) && (strmap[i].derived == -1)) {
			SUBDBG("pfm_get_inst_retired_event(%p)\n",&event);
			if ((ret = pfm_get_inst_retired_event(&event)) == PFMLIB_SUCCESS) {
				if (setup_preset_term(&psmap[j].data.native[0], &event) == PAPI_OK) {
				  psmap[j].event_code = (unsigned int)strmap[i].preset;
					psmap[j].data.derived = NOT_DERIVED;
					psmap[j].data.native[1] = PAPI_NULL;
					j++;
				}
			}
			else
				SUBDBG("pfm_get_inst_retired_event(%p): %s\n",&event, pfm_strerror(ret));
		}
		else {
			/* Handle derived events */
			term = 0;
			do {
				SUBDBG("pfm_find_full_event(%s,%p)\n",strmap[i].findme[term],&event);
				if ((ret = pfm_find_full_event(strmap[i].findme[term],&event)) == PFMLIB_SUCCESS) {
					if ((ret = setup_preset_term(&psmap[j].data.native[term], &event)) == PAPI_OK) {
						term++;
					}
					else break;
				}
				else {
					PAPIERROR("pfm_find_full_event(%s,%p): %s",strmap[i].findme[term],&event,pfm_strerror(ret));
					term++;
				}
			} while (strmap[i].findme[term] != NULL && term < MAX_COUNTER_TERMS);

			/* terminate the native term array with PAPI_NULL */
			if (term < MAX_COUNTER_TERMS) psmap[j].data.native[term] = PAPI_NULL;

			if (ret == PAPI_OK)
			{
				psmap[j].event_code = (unsigned int)strmap[i].preset;
				psmap[j].data.derived = strmap[i].derived;
				if (strmap[i].derived == DERIVED_POSTFIX) {
					strncpy(psmap[j].data.operation, strmap[i].operation, PAPI_MIN_STR_LEN);
				}
				if (strmap[i].note) {
					notemap[k].event_code = (unsigned int)strmap[i].preset;
					notemap[k].dev_note = strdup(strmap[i].note);
					k++;
				}
				j++;
			}
		}
		i++;
	}
	if (i != j) {
		PAPIERROR("%d of %d events in %s were not valid",i-j,i,PAPI_EVENT_FILE);
	}
	SUBDBG("generate_preset_search_map(%p,%p,%p) %d actual presets\n",maploc,noteloc,strmap,j);
	*maploc = psmap;
	*noteloc = notemap;
	return (PAPI_OK);
}

/* Break a PAPI native event code into its composite event code and pfm mask bits */
inline int _pfm_decode_native_event(unsigned int EventCode, unsigned int *event, unsigned int *umask)
{
  unsigned int tevent, major, minor;

  tevent = EventCode & PAPI_NATIVE_AND_MASK;
  major = (tevent & PAPI_NATIVE_EVENT_AND_MASK) >> PAPI_NATIVE_EVENT_SHIFT;
  if ((int)major >= MY_VECTOR.cmp_info.num_native_events)
    return(PAPI_ENOEVNT);

  minor = (tevent & PAPI_NATIVE_UMASK_AND_MASK) >> PAPI_NATIVE_UMASK_SHIFT;
  *event = major;
  *umask = minor;
  SUBDBG("EventCode 0x%08x is event %d, umask 0x%x\n",EventCode,major,minor);
  return(PAPI_OK);
}

/* convert a collection of pfm mask bits into an array of pfm mask indices */
static inline int prepare_umask(unsigned int foo,unsigned int *values)
{
  unsigned int tmp = foo, i;
  int j = 0;

  SUBDBG("umask 0x%x\n",tmp);
  while ((i = (unsigned int)ffs((int)tmp)))
    {
      tmp = tmp ^ (1 << (i-1));
      values[j] = i - 1;
      SUBDBG("umask %d is %d\n",j,values[j]);
      j++;
    }
  return(j);
}

/* convert the mask values in a pfm event structure into a PAPI unit mask */
static inline unsigned int convert_pfm_masks(pfmlib_event_t *gete)
{
  int ret;
  unsigned int i, code, tmp = 0;

  for (i=0;i<gete->num_masks;i++) {
    if ((ret = pfm_get_event_mask_code(gete->event, gete->unit_masks[i], &code)) == PFMLIB_SUCCESS) {
      SUBDBG("Mask value is 0x%08x\n",code);
      tmp |= code;
    } else {
      PAPIERROR("pfm_get_event_mask_code(0x%x,%d,%p): %s",gete->event,i,&code,pfm_strerror(ret));
    }
  }
  return(tmp);
}
/* convert an event code and pfm unit mask into a PAPI unit mask */
inline unsigned int _pfm_convert_umask(unsigned int event, unsigned int umask)
{
  pfmlib_event_t gete;
  memset(&gete,0,sizeof(gete));  
  gete.event = event;
  gete.num_masks = (unsigned int)prepare_umask(umask,gete.unit_masks);
  return(convert_pfm_masks(&gete));
}

int _papi_pfm_setup_presets(char *pmu_name, int pmu_type)
{
  int retval;
  hwi_search_t *preset_search_map = NULL;
  hwi_dev_notes_t *notemap = NULL;
  pfm_preset_search_entry_t *_perfmon2_pfm_preset_search_map;

  /* allocate and clear array of search string structures */
  _perfmon2_pfm_preset_search_map = malloc(sizeof(pfm_preset_search_entry_t)*PAPI_MAX_PRESET_EVENTS);
  if (_perfmon2_pfm_preset_search_map == NULL)
    return(PAPI_ENOMEM);
  memset(_perfmon2_pfm_preset_search_map,0x0,sizeof(pfm_preset_search_entry_t)*PAPI_MAX_PRESET_EVENTS);

   retval = load_preset_table(pmu_name, pmu_type, _perfmon2_pfm_preset_search_map);
   if (retval)
     return(retval);

   retval = generate_preset_search_map(&preset_search_map,&notemap,_perfmon2_pfm_preset_search_map);
    free_preset_table(_perfmon2_pfm_preset_search_map);
    free(_perfmon2_pfm_preset_search_map);
    if (retval)
      return (retval);

   retval = _papi_hwi_setup_all_presets(preset_search_map, notemap);
   if (retval)
     {
       free(preset_search_map);
       free_notes(notemap);
       free(notemap);
       return (retval);
     }
   return (PAPI_OK);
}

int _papi_pfm_init()
{
  int retval;
  unsigned int ncnt;

   /* Opened once for all threads. */
   SUBDBG("pfm_initialize()\n");
   if ((retval = pfm_initialize()) != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_initialize(): %s", pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }

   /* Fill in MY_VECTOR.cmp_info.num_native_events */

   SUBDBG("pfm_get_num_events(%p)\n",&ncnt);
   if ((retval = pfm_get_num_events(&ncnt)) != PFMLIB_SUCCESS)
   {
      PAPIERROR("pfm_get_num_events(%p): %s", &ncnt, pfm_strerror(retval));
      return(PAPI_ESBSTR);
   }
   SUBDBG("pfm_get_num_events() returns: %d\n",ncnt);
   MY_VECTOR.cmp_info.num_native_events = (int)ncnt;
  return (PAPI_OK);
}

unsigned int _papi_pfm_ntv_name_to_code(char *name, int *event_code)
{
  pfmlib_event_t event;
  unsigned int i;
  int ret;

  SUBDBG("pfm_find_full_event(%s,%p)\n",name,&event);
  ret = pfm_find_full_event(name,&event);
  if (ret == PFMLIB_SUCCESS) {
	/* we can only capture PAPI_NATIVE_UMASK_MAX or fewer masks */
	if (event.num_masks > PAPI_NATIVE_UMASK_MAX) {
	  SUBDBG("num_masks (%d) > max masks (%d)\n",event.num_masks, PAPI_NATIVE_UMASK_MAX);
	  return (unsigned int)PAPI_ENOEVNT;
	}
	else {
	/* no mask index can exceed PAPI_NATIVE_UMASK_MAX */
	  for (i=0; i<event.num_masks; i++) {
		if (event.unit_masks[i] > PAPI_NATIVE_UMASK_MAX) {
		  SUBDBG("mask index (%d) > max masks (%d)\n",event.unit_masks[i], PAPI_NATIVE_UMASK_MAX);
		  return (unsigned int)PAPI_ENOEVNT;
		}
	  }
	  *event_code = encode_native_event(event.event, event.num_masks, event.unit_masks);
	  return(PAPI_OK);
	}
  } else if (ret == PFMLIB_ERR_UMASK) {
	ret = pfm_find_event(name, &event.event);
	if (ret == PFMLIB_SUCCESS) {
	  *event_code = encode_native_event(event.event, 0, 0);
	  return(PAPI_OK);
	}
  }
  return (unsigned int)PAPI_ENOEVNT;
}

int _papi_pfm_ntv_code_to_name(unsigned int EventCode, char *ntv_name, int len)
{
  int ret;
  unsigned int event, umask;
  pfmlib_event_t gete;

  memset(&gete,0,sizeof(gete));
  
  if (_pfm_decode_native_event(EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);
  
  gete.event = event;
  gete.num_masks = (unsigned int)prepare_umask(umask,gete.unit_masks);
  if (gete.num_masks == 0)
    ret = pfm_get_event_name(gete.event, ntv_name, (size_t)len);
  else
    ret = pfm_get_full_event_name(&gete, ntv_name, (size_t)len);
  if (ret != PFMLIB_SUCCESS)
    {
      char tmp[PAPI_2MAX_STR_LEN];
      pfm_get_event_name(gete.event,tmp,sizeof(tmp));
      /* Skip error message if event is not supported by host cpu;
	   * we don't need to give this info away for papi_native_avail util */
	  if ( ret != PFMLIB_ERR_BADHOST )			
	  PAPIERROR("pfm_get_full_event_name(%p(event %d,%s,%d masks),%p,%d): %d -- %s",
		&gete,gete.event,tmp,gete.num_masks,ntv_name,len,ret,pfm_strerror(ret));
      if (ret == PFMLIB_ERR_FULL) return(PAPI_EBUF);
      return(PAPI_ESBSTR);
    }
  return(PAPI_OK);
}

int _papi_pfm_ntv_code_to_descr(unsigned int EventCode, char *ntv_descr, int len)
{
  unsigned int event, umask;
  char *eventd, **maskd, *tmp;
  int i, ret;
  pfmlib_event_t gete;
  size_t total_len = 0;

  memset(&gete,0,sizeof(gete));
  
  if (_pfm_decode_native_event(EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);
  
  ret = pfm_get_event_description(event,&eventd);
  if (ret != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_event_description(%d,%p): %s",
		event,&eventd,pfm_strerror(ret));
      return(PAPI_ENOEVNT);
    }

  if ((gete.num_masks = (unsigned int)prepare_umask(umask,gete.unit_masks)))
    {
      maskd = (char **)malloc(gete.num_masks*sizeof(char *));
      if (maskd == NULL)
	{
	  free(eventd);
	  return(PAPI_ENOMEM);
	}
      for (i=0;i<(int)gete.num_masks;i++)
	{
	  ret = pfm_get_event_mask_description(event,gete.unit_masks[i],&maskd[i]);
	  if (ret != PFMLIB_SUCCESS)
	    {
	      PAPIERROR("pfm_get_event_mask_description(%d,%d,%p): %s",
			event,umask,&maskd,pfm_strerror(ret));
	      free(eventd);
	      for (;i>=0;i--)
		free(maskd[i]);
	      free(maskd);
	      return(PAPI_EINVAL);
	    }
	  total_len += strlen(maskd[i]);
	}
      tmp = (char *)malloc(strlen(eventd)+strlen(", masks:")+total_len+gete.num_masks+1);
      if (tmp == NULL)
	{
	  for (i=(int)gete.num_masks-1;i>=0;i--)
	    free(maskd[i]);
	  free(maskd);
	  free(eventd);
	}
      tmp[0] = '\0';
      strcat(tmp,eventd);
      strcat(tmp,", masks:");
      for (i=0;i<(int)gete.num_masks;i++)
	{
	  if (i!=0)
	    strcat(tmp,",");
	  strcat(tmp,maskd[i]);
	  free(maskd[i]);
	}
      free(maskd);
    }
  else
    {
      tmp = (char *)malloc(strlen(eventd)+1); 
      if (tmp == NULL)
	{
	  free(eventd);
	  return(PAPI_ENOMEM);
	}
      tmp[0] = '\0';
      strcat(tmp,eventd);
      free(eventd);
    }
  strncpy(ntv_descr, tmp, (size_t)len);
  if ((int)strlen(tmp) > len-1) ret = PAPI_EBUF;
  else ret = PAPI_OK;
  free(tmp);
  return(ret);
}

int _papi_pfm_ntv_enum_events(unsigned int *EventCode, int modifier)
{
  unsigned int event, umask, num_masks;
  int ret;

  if (modifier == PAPI_ENUM_FIRST) {
    *EventCode = PAPI_NATIVE_MASK; /* assumes first native event is always 0x4000000 */
    return (PAPI_OK);
  }

  if (_pfm_decode_native_event(*EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);

  ret = pfm_get_num_event_masks(event,&num_masks);
  if (ret != PFMLIB_SUCCESS) {
    PAPIERROR("pfm_get_num_event_masks(%d,%p): %s",event,&num_masks,pfm_strerror(ret));
    return(PAPI_ENOEVNT);
  }
  if (num_masks > PAPI_NATIVE_UMASK_MAX) num_masks = PAPI_NATIVE_UMASK_MAX;
  SUBDBG("This is umask %d of %d\n",umask,num_masks);

  if (modifier == PAPI_ENUM_EVENTS) {
    if (event < (unsigned int)MY_VECTOR.cmp_info.num_native_events - 1) {
      *EventCode = (unsigned int)encode_native_event_raw(event+1,0);
	  return (PAPI_OK);
	}
    return (PAPI_ENOEVNT);
  }
  else if (modifier == PAPI_NTV_ENUM_UMASK_COMBOS){
    if (umask+1 < (unsigned int)(1<<num_masks)) {
      *EventCode = (unsigned int)encode_native_event_raw(event,umask+1);
	  return (PAPI_OK);
	}
    return(PAPI_ENOEVNT);
  }
  else if (modifier == PAPI_NTV_ENUM_UMASKS) {
    int thisbit = ffs((int)umask);

    SUBDBG("First bit is %d in %08x\b\n",thisbit-1,umask);
    thisbit = 1 << thisbit;

    if (thisbit & ((1<<num_masks)-1)) {
      *EventCode = (unsigned int)encode_native_event_raw(event,(unsigned int)thisbit);
	  return (PAPI_OK);
	}
    return(PAPI_ENOEVNT);
  }
  else
    return(PAPI_EINVAL);
}

#ifndef PENTIUM4

/* This call is broken. Selector can be much bigger than 32 bits. It should be a pfmlib_regmask_t - pjm */
/* Also, libpfm assumes events can live on different counters with different codes. This call only returns
    the first occurence found. */
/* Right now its only called by ntv_code_to_bits in p3_pfm_events, so we're ok. But for it to be
    generally useful it should be fixed. - dkt */
int _pfm_get_counter_info(unsigned int event, unsigned int *selector, int *code)
{
    pfmlib_regmask_t cnt, impl;
    unsigned int num;
    unsigned int i, first = 1;
    int ret;

    if ((ret = pfm_get_event_counters(event,&cnt)) != PFMLIB_SUCCESS) {
      PAPIERROR("pfm_get_event_counters(%d,%p): %s", event,&cnt,pfm_strerror(ret));
      return(PAPI_ESBSTR);
    }
    if ((ret = pfm_get_num_counters(&num)) != PFMLIB_SUCCESS) {
      PAPIERROR("pfm_get_num_counters(%p): %s", num,pfm_strerror(ret));
      return(PAPI_ESBSTR);
    }
    if ((ret = pfm_get_impl_counters(&impl)) != PFMLIB_SUCCESS) {
      PAPIERROR("pfm_get_impl_counters(%p): %s", &impl, pfm_strerror(ret));
      return(PAPI_ESBSTR);
    }

	*selector = 0;
	for (i=0; num; i++) {
		if (pfm_regmask_isset(&impl, i))
			num--;
		if (pfm_regmask_isset(&cnt, i)) {
			if (first) {
				if ((ret = pfm_get_event_code_counter(event,i,code)) != PFMLIB_SUCCESS) {
					PAPIERROR("pfm_get_event_code_counter(%p, %d, %p): %s", event, i, code, pfm_strerror(ret));
					return(PAPI_ESBSTR);
				}
				first = 0;
			}
			*selector |= 1 << i;
		}
	}
	return(PAPI_OK);
}
#endif


#ifndef PERFCTR_PFM_EVENTS

int _papi_pfm_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
  unsigned int event, umask;
  pfmlib_event_t gete;

  /* For PFM & Perfmon, native info is just an index into the PFM event table. */
  if (_pfm_decode_native_event(EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);

  memset(&gete,0x0,sizeof(pfmlib_event_t));

  gete.event = event;
  gete.num_masks = prepare_umask(umask,gete.unit_masks);
  
  memcpy(bits,&gete,sizeof(pfmlib_event_t));
  return (PAPI_OK);
}

static char *_pmc_name(int i)
{
  /* Should get this from /sys */
  extern int _perfmon2_pfm_pmu_type;

  switch (_perfmon2_pfm_pmu_type)
    {
#if defined(PFMLIB_MIPS_ICE9A_PMU)
      /* All the counters after the 2 CPU counters, the 4 sample counters are SCB registers. */
    case PFMLIB_MIPS_ICE9A_PMU:
    case PFMLIB_MIPS_ICE9B_PMU:
      switch (i) {
      case 0:
	return "Core counter 0";
      case 1:
	return "Core counter 1";
      default:
	return "SCB counter";
      }
      break;
#endif
    default:
      return "Event Code";
    }
}

int _papi_pfm_ntv_bits_to_info(hwd_register_t *bits, char *names,
								unsigned int *values, int name_len, int count)
{
	int ret;
	pfmlib_regmask_t selector;
	int j, n = MY_VECTOR.cmp_info.num_cntrs;
	int foo, did_something=0;
	unsigned int umask;

	if ((ret = pfm_get_event_counters(((pfm_register_t *)bits)->event,&selector)) != PFMLIB_SUCCESS) {
		PAPIERROR("pfm_get_event_counters(%d,%p): %s",((pfm_register_t *)bits)->event,&selector,pfm_strerror(ret));
		return(PAPI_ESBSTR);
	}

#if defined(PFMLIB_MIPS_ICE9A_PMU)
	extern int _perfmon2_pfm_pmu_type;
	switch (_perfmon2_pfm_pmu_type) {
		/* All the counters after the 2 CPU counters, the 4 sample counters are SCB registers. */
		case PFMLIB_MIPS_ICE9A_PMU:
		case PFMLIB_MIPS_ICE9B_PMU:
			if (n > 7) n = 7;
				break;
		default:
			break;
	}
#endif

	for (j=0;n;j++) {
		if (pfm_regmask_isset(&selector,j)) {
			if ((ret = pfm_get_event_code_counter(((pfm_register_t *)bits)->event,j,&foo)) != PFMLIB_SUCCESS) {
				PAPIERROR("pfm_get_event_code_counter(%p,%d,%d,%p): %s",*((pfm_register_t *)bits),((pfm_register_t *)bits)->event,j,&foo,pfm_strerror(ret));
				return(PAPI_EBUG);
			}
			/* Overflow check */
			if ((did_something*name_len + strlen(_pmc_name(j)) + 1) >= count*name_len) {
				SUBDBG("Would overflow register name array.");
				return(did_something);
			}
			values[did_something] = foo;
			strncpy(&names[did_something*name_len],_pmc_name(j),name_len);
			did_something++;
			if (did_something == count) break;
		}
		n--;
	}
	/* assumes umask is unchanged, even if event code changes */
	umask = convert_pfm_masks(bits);
	if (umask && (did_something < count)) {
		values[did_something] = umask;
		if (strlen(&names[did_something*name_len]))
		  strncpy(&names[did_something*name_len]," Unit Mask",name_len);
		else
		  strncpy(&names[did_something*name_len],"Unit Mask",name_len);
		did_something++;
	}
	return(did_something);
}

#endif /* PERFCTR_PFM_EVENTS */

/*
papi_svector_t _papi_pfm_event_vectors[] = {
  {(void (*)())_papi_pfm_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
  {(void (*)())_papi_pfm_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
  {(void (*)())_papi_pfm_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
  {(void (*)())_papi_pfm_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
  {(void (*)())_papi_pfm_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {NULL, VEC_PAPI_END}
};
*/

