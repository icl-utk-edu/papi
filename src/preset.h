#ifndef _PAPI_PRESET  /* _PAPI_PRESET */
#define _PAPI_PRESET

#include SUBSTRATE

#define OPS MAX_COUNTERS*3

typedef struct preset_search {
  /* Preset code */
  int preset;
  /* Derived code */
  int derived;
  /* native event codes */
  unsigned int natEvent[POWER_MAX_COUNTERS];
} preset_search_t;

typedef struct hwi_preset {
  /* Derived code */
  int derived;
  /* number of metrics the preset consists */
  int metric_count;
  /* index array of native events */
  int  natIndex[MAX_COUNTERS];
  /* operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
  char operation[OPS];
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwi_preset_t;

extern preset_search_t *preset_search_map;
extern hwi_preset_t _papi_hwi_preset_map[];

extern int _papi_hwi_preset_query(int preset_index, int *flags, char **note);
extern int setup_all_presets(preset_search_t *preset_search_map);

#endif /* _PAPI_PRESET */
