#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <libgen.h>
#include <sys/systemcfg.h>
#include <sys/processor.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <procinfo.h>
#include <sys/atomic_op.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "pmapi.h"
/*#include "papi.h"*/

#define MAX_NATIVE_EVENT 256
#define MAX_COUNTERS 8
#define NATIVE_MASK 0x40000000
#define PAPI_MAX_STR_LEN 129
#define GROUP_INTS 2
#define MAX_GROUPS (GROUP_INTS * 32)

typedef struct PWR_register {
  /* indicate which counters this event can live on */
  unsigned int selector;
  /* Buffers containing counter cmds for each possible metric */
  int counter_cmd[MAX_COUNTERS];
#ifdef _POWER4
  /* which group this event belongs to */
  unsigned int group[GROUP_INTS];
  /* dense array contains group numbers for the correspondent 
  counter the event can stay */
  /*int rgg[MAX_COUNTERS];*/
#endif
} PWR_register_t;

#ifdef _POWER4
typedef struct PWR4_groups {
  /* group number from the pmapi pm_groups_t struct */
  /*int group_id;*/
  /* Buffer containing counter cmds for this group */
  unsigned char counter_cmd[MAX_COUNTERS];
} PWR4_groups_t;
#endif

typedef struct native_event_entry{
  /* description of the resources required by this native event */
  PWR_register_t resources;
  /* If it exists, then this is the name of this event */
  char name[PAPI_MAX_STR_LEN];
  /* If it exists, then this is the description of this event */
  char *description;
} native_event_entry_t;  

/* globals */
native_event_entry_t native_table[MAX_NATIVE_EVENT] = { 0 };
pm_info_t pminfo;
#ifdef _AIXVERSION_510
  pm_groups_info_t pmgroups;
#endif
#ifdef _POWER4
PWR4_groups_t group_map[MAX_GROUPS] = { 0 };
#endif

/* to initialize the native_table */
void initialize_native_table()
{
  int i, j;
  
  for(i=0;i<MAX_NATIVE_EVENT;i++){
	for(j=0;j<MAX_COUNTERS;j++)
		native_table[i].resources.counter_cmd[j]=-1;
  }
}

#ifdef _POWER4
/* to setup native_table group value */
void setup_gps(int total)
{
  int i, j, gnum;
  
  for(i=0;i<total;i++){
  	for(j=0;j<MAX_COUNTERS;j++){
	/*	native_table[i].resources.rgg[j]=-1;*/
		if(native_table[i].resources.selector & (1<<j)){
			for(gnum=0;gnum<pmgroups.maxgroups;gnum++){
  				if(native_table[i].resources.counter_cmd[j]==pmgroups.event_groups[gnum].events[j]){
					/* could use gnum instead of pmgroups.event_groups[gnum].group_id */
					native_table[i].resources.group[pmgroups.event_groups[gnum].group_id/32] |= \
				   	  1<<(pmgroups.event_groups[gnum].group_id%32);
					/*native_table[i].resources.rgg[j]=gnum;*/
				}
			}
		}
	}
  }

  for(gnum=0;gnum<pmgroups.maxgroups;gnum++){
	  for(i=0;i<MAX_COUNTERS;i++)
		  group_map[gnum].counter_cmd[i]=pmgroups.event_groups[gnum].events[i];
  }
}
#endif

/* to setup native_table values, and return number of entries */
int setup_native_table()
{
  pm_events_t *wevp;
  pm_info_t *info;
  int pmc, ev, i, index;
  
  info=&pminfo;
  index=0;
  for (pmc = 0; pmc < info->maxpmcs; pmc++){
  	wevp = info->list_events[pmc];
  	for (ev = 0; ev < info->maxevents[pmc]; ev++, wevp++){
 		for(i=0;i<index;i++){
			if (strcmp(wevp->short_name, native_table[i].name) == 0){
				native_table[i].resources.selector |= 1<<pmc;
				native_table[i].resources.counter_cmd[pmc]=wevp->event_id;
				break;
			}
		}
		if(i==index){
			/*native_table[i].index=i;*/
			native_table[i].resources.selector |= 1<<pmc;
			native_table[i].resources.counter_cmd[pmc]=wevp->event_id;
			strcpy(native_table[i].name, wevp->short_name);
			native_table[i].description=strdup(wevp->description);
			index++;
		}
	}
  }

#ifdef _POWER4
  setup_gps(index);
#endif

  return index;
}  

void main()
{
  FILE *fp[2];
  int retval, total, maxgroups, i, j, k;
  char *str, *tmp;

#ifdef _POWER4
  #define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT|PM_GET_GROUPS
#else
  #define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT
#endif

/* get counter information from system*/
#ifdef _AIXVERSION_510
  /*DBG((stderr,"Calling AIX 5 version of pm_init...\n"));*/
  retval = pm_init(PM_INIT_FLAGS, &pminfo, &pmgroups);
#else
  /*DBG((stderr,"Calling AIX 4 version of pm_init...\n"));*/
  retval = pm_init(PM_INIT_FLAGS,&pminfo);
#endif
  
  fp[0]=fopen("native.h", "w");
  if(fp[0]==NULL){
  	perror("open");
	exit(1);
  }
  fp[1]=fopen("native.c", "w");
  if(fp[1]==NULL){
  	perror("open");
	exit(1);
  }
  
  initialize_native_table();
  total=setup_native_table();
  maxgroups=pmgroups.maxgroups;

  /* write into native.h */
  fprintf(fp[0], "#ifndef _PAPI_NATIVE  /* _PAPI_NATIVE */\n");
  fprintf(fp[0], "#define _PAPI_NATIVE\n\n");
  fprintf(fp[0], "#include SUBSTRATE\n\n");
  fprintf(fp[0], "#define PAPI_MAX_NATIVE_EVENTS %d\n", total);
#ifdef _POWER4
  fprintf(fp[0], "#define GROUP_INTS 2\n");
  fprintf(fp[0], "#define MAX_GROUPS %d\n", pmgroups.maxgroups);
#endif

  if (__power_630())
    fprintf(fp[0], "#define PAPI_POWER_630\n");
  else{
    if (__power_604())
    {
      if (strstr(pminfo.proc_name,"604e"))
	     fprintf(fp[0], "#define PAPI_POWER_604e\n");
      else
	     fprintf(fp[0], "#define PAPI_POWER_604\n");
    }
  }

#ifdef _POWER4
  fprintf(fp[0], "\ntypedef struct PWR4_reg{\n");
#else
  fprintf(fp[0], "\ntypedef struct PWR3_reg{\n");
#endif
  fprintf(fp[0], "  /* indicate which counters this event can live on */\n");
  fprintf(fp[0], "  unsigned int selector;\n");
  fprintf(fp[0], "  /* Buffers containing counter cmds for each possible metric */\n");
  fprintf(fp[0], "  int counter_cmd[MAX_COUNTERS];\n");
#ifdef _POWER4
  fprintf(fp[0], "  /* which group this event belongs */\n");
  fprintf(fp[0], "  unsigned int group[GROUP_INTS];\n");
/*  fprintf(fp[0], "  /* dense array contains group numbers for the correspondent\n"); 
  fprintf(fp[0], "  counter the event can stay \n"); 
  fprintf(fp[0], "  int rgg[MAX_COUNTERS];\n"); */
#endif
  fprintf(fp[0], "} hwd_reg_t;\n\n"); 

#ifdef _POWER4
  fprintf(fp[0], "typedef struct PWR4_groups {\n");
  /* group number from the pmapi pm_groups_t struct */
  /*int group_id;*/
  fprintf(fp[0], "/* Buffer containing counter cmds for this group */\n");
  fprintf(fp[0], "unsigned char counter_cmd[MAX_COUNTERS];\n");
  fprintf(fp[0], "} hwd_groups_t;\n");
#endif

  fprintf(fp[0], "\ntypedef struct native_event_entry{\n");
  fprintf(fp[0], "    /* description of the resources required by this native event */\n");
  fprintf(fp[0], "    hwd_reg_t resources;\n");
  fprintf(fp[0], "  /* If it exists, then this is the name of this event */\n");
  fprintf(fp[0], "  char name[PAPI_MAX_STR_LEN];\n");
  fprintf(fp[0], "  /* If it exists, then this is the description of this event */\n");
  fprintf(fp[0], "  char *description;\n");
  fprintf(fp[0], "} native_event_entry_t;\n\n"); 

  fprintf(fp[0], "extern native_event_entry_t native_table[PAPI_MAX_NATIVE_EVENTS];\n\n"); 
  fprintf(fp[0], "extern hwd_groups_t group_map[MAX_GROUPS];\n\n"); 

  for(i=0;i<total;i++){
  	fprintf(fp[0], "#define PNE_%-40s 0x%x\n", native_table[i].name, NATIVE_MASK+i);
  }
  fprintf(fp[0], "\n#endif /* _PAPI_NATIVE */\n");
    
  if(fclose(fp[0])!=0){
  	perror("fclose");
	exit(1);
  } 

  /* write into native.c */
  fprintf(fp[1], "#include \"native.h\"\n\n");
  fprintf(fp[1], "native_event_entry_t native_table[PAPI_MAX_NATIVE_EVENTS] = {");
  for(i=0;i<total;i++){
  	if(i==0)
		fprintf(fp[1], "\n { {0x%x, {%d", native_table[i].resources.selector, native_table[i].resources.counter_cmd[0]);
	else
	  	fprintf(fp[1], ",\n { {0x%x, {%d", native_table[i].resources.selector, native_table[i].resources.counter_cmd[0]);
    for(j=1;j<MAX_COUNTERS;j++){
		fprintf(fp[1], ",%d", native_table[i].resources.counter_cmd[j]);
	}
	fprintf(fp[1], "}, ");

  #ifdef _POWER4
  	fprintf(fp[1], "{ 0x%x", native_table[i].resources.group[0]);
  	for(k=1;k<GROUP_INTS;k++){
		fprintf(fp[1], ", 0x%x", native_table[i].resources.group[k]);
	}
	fprintf(fp[1], "}, ");
  	
/*	{
		int kk=0;
		fprintf(fp[1], "{");
  	for(k=0;k<MAX_COUNTERS;k++){
		if(native_table[i].resources.rgg[k]>=0){
			if(kk)
				fprintf(fp[1], ",");
			fprintf(fp[1], " %d", native_table[i].resources.rgg[k]);
			kk++;
		}
	}
	fprintf(fp[1], " }, ");
	}*/
  #endif
	fprintf(fp[1], "}, \"%s\", ", native_table[i].name);
  	fprintf(fp[1], "\"");
	str=strdup(native_table[i].description);
	while((tmp=strchr(str, '\n'))!=NULL){
		*tmp='\0';
		fprintf(fp[1], "%s\\n", str);
		str=tmp+1;
	}		
	fprintf(fp[1], "\"}");
  }
  fprintf(fp[1], "};\n");

  #ifdef _POWER4
  fprintf(fp[1], "hwd_groups_t group_map[MAX_GROUPS] = {");
  for(i=0;i<maxgroups;i++){
  	if(i==0)
		fprintf(fp[1], "\n	{ {%d", group_map[i].counter_cmd[0]);
	else
	  	fprintf(fp[1], ",\n	{ {%d", group_map[i].counter_cmd[0]);
    for(j=1;j<MAX_COUNTERS;j++){
		fprintf(fp[1], ",%d", group_map[i].counter_cmd[j]);
	}
	fprintf(fp[1], "} }");
  }
  fprintf(fp[1], "};\n");
  #endif
  
  if(fclose(fp[1])!=0){
  	perror("fclose");
	exit(1);
  } 

}  
  
