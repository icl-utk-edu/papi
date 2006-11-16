/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/
#include <stdlib.h>
#include <stdio.h>
#include "papi_test.h"

int main(int argc, char **argv)
{
   int retval;

   const PAPI_substrate_info_t *subinfo;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if ((subinfo = PAPI_get_substrate_info()) == NULL) 
     test_fail(__FILE__, __LINE__, "PAPI_get_substrate_info", retval);

   printf("name: %s\n",subinfo->name);
   printf("substrate_version: %s\n",subinfo->version);
   printf("support_version: %s\n",subinfo->support_version);
   printf("kernel_version: %s\n",subinfo->kernel_version);
   printf("num_cntrs: %d\n",subinfo->num_cntrs);
   printf("num_mpx_cntrs: %d\n",subinfo->num_mpx_cntrs);
   printf("num_preset_events: %d\n",subinfo->num_preset_events);           /* Number of counters the substrate supports */
   printf("num_native_events: %d\n",subinfo->num_native_events);           /* Number of counters the substrate supports */
   printf("default_domain: 0x%x (%s)\n",subinfo->default_domain,stringify_all_domains(subinfo->default_domain));
   printf("available_domains: 0x%x (%s)\n",subinfo->available_domains,stringify_all_domains(subinfo->available_domains));       /* Available domains */ 
   printf("default_granularity: 0x%x (%s)\n",subinfo->default_granularity,stringify_granularity(subinfo->default_granularity));
   /* The default granularity when this substrate is used */
   printf("available_granularities: 0x%x (%s)\n",subinfo->available_granularities,stringify_all_granularities(subinfo->available_granularities)); /* Available granularities */
   printf("multiplex_timer_sig: %d\n",subinfo->multiplex_timer_sig);      /* Width of opcode matcher if exists, 0 if not */
   printf("multiplex_timer_num: %d\n",subinfo->multiplex_timer_num);      /* Width of opcode matcher if exists, 0 if not */
   printf("multiplex_timer_us: %d\n",subinfo->multiplex_timer_us);      /* Width of opcode matcher if exists, 0 if not */
   printf("hardware_intr_sig: %d\n",subinfo->hardware_intr_sig);      /* Width of opcode matcher if exists, 0 if not */
   printf("opcode_match_width: %d\n",subinfo->opcode_match_width);      /* Width of opcode matcher if exists, 0 if not */
/*   printf("reserved_ints[4]: %d\n",subinfo->reserved_ints[4]); */
   printf("hardware_intr: %d\n",subinfo->hardware_intr);         /* Needs hw overflow intr to be emulated in software*/
   printf("precise_intr: %d\n",subinfo->precise_intr);          /* Performance interrupts happen precisely */
   printf("posix1b_timers: %d\n",subinfo->posix1b_timers);          /* Performance interrupts happen precisely */
   printf("kernel_profile: %d\n",subinfo->kernel_profile);        /* Needs kernel profile support (buffered interrupts) to be emulated */
   printf("kernel_multiplex: %d\n",subinfo->kernel_multiplex);      /* In kernel multiplexing */
   printf("data_address_range: %d\n",subinfo->data_address_range);    /* Supports data address range limiting */
   printf("instr_address_range: %d\n",subinfo->instr_address_range);   /* Supports instruction address range limiting */
   printf("fast_counter_read: %d\n",subinfo->fast_counter_read);       /* Has a fast counter read */
   printf("fast_real_timer: %d\n",subinfo->fast_real_timer);       /* Has a fast real timer */
   printf("fast_virtual_timer: %d\n",subinfo->fast_virtual_timer);    /* Has a fast virtual timer */
   printf("attach: %d\n",subinfo->attach);    /* Has a fast virtual timer */
   printf("attach_must_ptrace: %d\n",subinfo->attach_must_ptrace);    /* Has a fast virtual timer */
   printf("edge_detect: %d\n",subinfo->edge_detect);    /* Has a fast virtual timer */
   printf("invert: %d\n",subinfo->invert);    /* Has a fast virtual timer */
   printf("profile_ear: %d\n",subinfo->profile_ear);     /* Supports data/instr/tlb miss address sampling */
   printf("grouped_cntrs: %d\n",subinfo->grouped_cntrs);           /* Number of counters the substrate supports */

   test_pass(__FILE__, NULL, 0);
   exit(0);
}
