/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

#ifndef __PFMLIB_POWER6_PRIV_H__
#define __PFMLIB_POWER6_PRIV_H__

/*
* File:    pfmlib_power6_priv.h
* CVS:
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*
* (C) Copyright IBM Corporation, 2007.  All Rights Reserved.
* Contributed by Corey Ashford <cjashfor.ibm.com>
*
* Note: This code was automatically generated and should not be modified by
* hand.
*
*/

#define POWER6_NUM_EVENT_COUNTERS 6
#define POWER6_NUM_GROUP_VEC 4
#define POWER6_NUM_CONTROL_REGS 3

typedef struct {
   char *pme_name;
   char *pme_short_desc;
   char *pme_long_desc;
   int pme_event_ids[POWER6_NUM_EVENT_COUNTERS];
   unsigned long long pme_group_vector[POWER6_NUM_GROUP_VEC];
} pme_power6_entry_t;

typedef struct {
   char *pmg_name;
   char *pmg_desc;
   int pmg_event_ids[POWER6_NUM_EVENT_COUNTERS];
   unsigned long long pmg_mmcr0;
   unsigned long long pmg_mmcr1;
   unsigned long long pmg_mmcra;
} pmg_power6_group_t;


#endif

