#include SUBSTRATE


hwd_pminfo_t pminfo;
pm_groups_info_t pmgroups;
native_event_entry_t native_table[PAPI_MAX_NATIVE_EVENTS];
hwd_groups_t group_map[MAX_GROUPS] = { 0 };

/* to initialize the native_table */
void initialize_native_table()
{
   int i, j;

   memset(native_table, 0, PAPI_MAX_NATIVE_EVENTS * sizeof(native_event_entry_t));
   for (i = 0; i < PAPI_MAX_NATIVE_EVENTS; i++) {
      for (j = 0; j < MAX_COUNTERS; j++)
         native_table[i].resources.counter_cmd[j] = -1;
   }
}

/* to setup native_table group value */
static void ppc64_setup_gps(int total)
{
   int i, j, gnum;

   for (i = 0; i < total; i++) {
      for (j = 0; j < MAX_COUNTERS; j++) {
         /*      native_table[i].resources.rgg[j]=-1; */
         if (native_table[i].resources.selector & (1 << j)) {
            for (gnum = 0; gnum < pmgroups.maxgroups; gnum++) {
               if (native_table[i].resources.counter_cmd[j] ==
                   pmgroups.event_groups[gnum].events[j]) {
                  /* could use gnum instead of pmgroups.event_groups[gnum].group_id */
                  native_table[i].resources.group[pmgroups.event_groups[gnum].group_id /
                                                  32] |=
                      1 << (pmgroups.event_groups[gnum].group_id % 32);
                  /*native_table[i].resources.rgg[j]=gnum; */
               }
            }
         }
      }
   }

   for (gnum = 0; gnum < pmgroups.maxgroups; gnum++) {
      for (i = 0; i < MAX_COUNTERS; i++)
         group_map[gnum].counter_cmd[i] = pmgroups.event_groups[gnum].events[i];
   }
}

/* to setup native_table values, and return number of entries */
void ppc64_setup_native_table()
{
   hwd_pmevents_t *wevp;
   hwd_pminfo_t * info;
   int pmc, ev, i, j, index;

   info = &pminfo;
   index = 0;
   initialize_native_table();
   for (pmc = 0; pmc < info->maxpmcs; pmc++) {
      wevp = info->list_events[pmc];
      for (ev = 0; ev < info->maxevents[pmc]; ev++, wevp++) {
         for (i = 0; i < index; i++) {
            if (strcmp(wevp->short_name, native_table[i].name) == 0) {
               native_table[i].resources.selector |= 1 << pmc;
               native_table[i].resources.counter_cmd[pmc] = wevp->event_id;
               break;
            }
         }
         if (i == index) {
            /*native_table[i].index=i; */
            native_table[i].resources.selector |= 1 << pmc;
            native_table[i].resources.counter_cmd[pmc] = wevp->event_id;
            native_table[i].name = wevp->short_name;
            native_table[i].description = wevp->description;
            index++;
            for (j = 0; j < MAX_NATNAME_MAP_INDEX; j++) {
               if (strcmp(native_table[i].name, native_name_map[j].name) == 0)
                  native_name_map[j].index = i;
            }
         }
      }
   }
   ppc64_setup_gps(index);
}
