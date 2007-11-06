/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

#ifndef __PFMLIB_PPC970MP_PRIV_H__
#define __PFMLIB_PPC970MP_PRIV_H__

/*
* File:    pfmlib_ppc970mp_priv.h
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

#define PPC970MP_NUM_EVENT_COUNTERS 8
#define PPC970MP_NUM_GROUP_VEC 1
#define PPC970MP_NUM_CONTROL_REGS 3

typedef struct {
   char *pme_name;
   char *pme_short_desc;
   char *pme_long_desc;
   int pme_event_ids[PPC970MP_NUM_EVENT_COUNTERS];
   unsigned long long pme_group_vector[PPC970MP_NUM_GROUP_VEC];
} pme_ppc970mp_entry_t;

typedef struct {
   char *pmg_name;
   char *pmg_desc;
   int pmg_event_ids[PPC970MP_NUM_EVENT_COUNTERS];
   unsigned long long pmg_mmcr0;
   unsigned long long pmg_mmcr1;
   unsigned long long pmg_mmcra;
} pmg_ppc970mp_group_t;


#endif

