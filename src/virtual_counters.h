/*
 * DCPI/DADD virtual counters
 */

/*
 * Author:  Paul J. Drongowski
 * Address: DUDE
 *          Compaq Computer Corporation
 *          110 Spit Brook Road
 *          Nashua, NH
 * Date:    20 March 2002
 * Version: 1.6
 *
 * Copyright (c) 2002 Compaq Computer Corporation
 */

#ifndef _VIRTUAL_COUNTERS_H
#define _VIRTUAL_COUNTERS_H

#include <sys/time.h>

/*
 * The virtual_counters structure is a set of 64-bit counts that can be
 * used to tally PAPI-like performance measures.
 */

typedef struct {
  struct timeval vc_time_stamp ;

  unsigned long vc_total_cycles ;               /* EV67/EV7 aggregate counts */
  unsigned long vc_bcache_misses ;

  unsigned long vc_total_dtbmiss ;              /* EV67/EV7 ProfileMe events */
  unsigned long vc_nyp_events ;
  unsigned long vc_taken_events ;
  unsigned long vc_mispredict_events ;
  unsigned long vc_ld_st_order_traps ;

  unsigned long vc_total_instr_issued ;         /* PAPI */
  unsigned long vc_total_instr_executed ;

  unsigned long vc_int_instr_executed ;         /* PAPI */
  unsigned long vc_load_instr_executed ;
  unsigned long vc_store_instr_executed ;
  unsigned long vc_total_load_store_executed ;  /* Sum of load and store */
  unsigned long vc_synch_instr_executed ;
  unsigned long vc_nop_instr_executed ;
  unsigned long vc_prefetch_instr_executed ;

  unsigned long vc_fa_instr_executed ;         /* PAPI */
  unsigned long vc_fm_instr_executed ;
  unsigned long vc_fd_instr_executed ;
  unsigned long vc_fsq_instr_executed ;
  unsigned long vc_fp_instr_executed ;

  unsigned long vc_uncond_br_executed ;        /* PAPI */
  unsigned long vc_cond_br_executed ;
  unsigned long vc_cond_br_taken ;
  unsigned long vc_cond_br_not_taken ;
  unsigned long vc_cond_br_mispredicted ;
  unsigned long vc_cond_br_predicted ;

  unsigned long vc_replay_traps ;               /* EV67/EV7 ProfileMe traps */
  unsigned long vc_no_traps ;     
  unsigned long vc_dtb2miss3_traps ;
  unsigned long vc_dtb2miss4_traps ;
  unsigned long vc_fpdisabled_traps ;
  unsigned long vc_unalign_traps ;
  unsigned long vc_dtbmiss_traps ;
  unsigned long vc_dfault_traps ;
  unsigned long vc_opcdec_traps ;
  unsigned long vc_mispredict_traps ;
  unsigned long vc_mchk_traps ;
  unsigned long vc_itbmiss_traps ;
  unsigned long vc_arith_traps ;
  unsigned long vc_interrupt_traps ;
  unsigned long vc_mt_fpcr_traps ;
  unsigned long vc_iacv_traps ;

  unsigned long vc_did_not_retire ;            /* Debugging purposes only */
  unsigned long vc_early_kill ;
  unsigned long vc_update_count ;
} virtual_counters ;


/*
 ****************************************************
 * Functions exported from virtual_counters.c
 ****************************************************
 */


/*
 * Function: print_virtual_counters
 * Purpose: Print current state of a set of virtual counters
 * Arguments:
 *   vc: Pointer to virtual counter information to be printed
 *   verbose_flag: If TRUE, display trap info and other extra stuff
 * Returns: Nothing
 */

extern void print_virtual_counters(virtual_counters *pvc, int verbose_flag) ;

#endif /* _VIRTUAL_COUNTERS_H */
