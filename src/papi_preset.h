#ifndef _PAPI_PRESET  /* _PAPI_PRESET */
#define _PAPI_PRESET


#define OPS MAX_COUNTERS*3

typedef struct preset_search {
  /* Preset code */
  unsigned int preset;
  /* Derived code */
  int derived;
  /* native event codes: must be signed since -1 is used as a terminator flag */
  int natEvent[MAX_COUNTER_TERMS];
} preset_search_t;

typedef struct hwi_preset {
  /* Derived code */
  int derived;
  /* number of metrics the preset consists */
  int metric_count;
  /* index array of native events */
  int  natIndex[MAX_COUNTER_TERMS];
  /* operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
  char operation[OPS];
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwi_preset_t;

extern preset_search_t *preset_search_map;

extern int _papi_hwi_preset_query(int preset_index, int *flags, char **note);
extern int setup_all_presets(preset_search_t *preset_search_map);

#endif /* _PAPI_PRESET */
