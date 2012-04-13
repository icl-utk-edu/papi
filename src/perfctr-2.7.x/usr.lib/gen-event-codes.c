/* $Id: gen-event-codes.c,v 1.9 2005/03/23 02:02:54 mikpe Exp $
 *
 * Generate symbolic constants for performance counter events.
 *
 * Copyright (C) 2003-2004  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>	/* for exit() */
#include "libperfctr.h"

/* detect sharing in a DAG of immutable nodes */
static int test_and_set(const struct perfctr_event_set *event_set)
{
    static const struct perfctr_event *marked[32];
    unsigned int i;
    const struct perfctr_event *events = event_set->events;

    for(i = 0; i < sizeof(marked)/sizeof(marked[0]); ++i) {
	if( !marked[i] ) {
	    marked[i] = events;
	    return 0;
	}
	if( marked[i] == events )
	    return 1;
    }
    fprintf(stderr, "gen-event-set: too many distinct event sets\n");
    exit(1);
    /*NOTREACHED*/
}

static void print_event_set(const struct perfctr_event_set *event_set)
{
    unsigned int i;

    if( test_and_set(event_set) )
	return;
    if( event_set->include )
	print_event_set(event_set->include);
    printf("\n");
    for(i = 0; i < event_set->nevents; ++i)
	printf("#define %s%s\t0x%02X\n",
	       event_set->event_prefix,
	       event_set->events[i].name,
	       event_set->events[i].evntsel);
}

static void print_cpu_type(unsigned int cpu_type)
{
    print_event_set(perfctr_cpu_event_set(cpu_type));
}

int main(void)
{
    printf("/* automatically generated, do not edit */\n");
#if defined(__i386__)
    print_cpu_type(PERFCTR_X86_INTEL_P5);
    print_cpu_type(PERFCTR_X86_INTEL_P5MMX);
    print_cpu_type(PERFCTR_X86_INTEL_P6);
    print_cpu_type(PERFCTR_X86_INTEL_PII);
    print_cpu_type(PERFCTR_X86_INTEL_PIII);
    print_cpu_type(PERFCTR_X86_INTEL_PENTM);
    print_cpu_type(PERFCTR_X86_CYRIX_MII);
    print_cpu_type(PERFCTR_X86_VIA_C3);
    print_cpu_type(PERFCTR_X86_WINCHIP_C6);
    print_cpu_type(PERFCTR_X86_WINCHIP_2);
    print_cpu_type(PERFCTR_X86_AMD_K7);
#endif
#if defined(__i386__) || defined(__x86_64__)
    print_cpu_type(PERFCTR_X86_INTEL_P4M3);
    print_cpu_type(PERFCTR_X86_AMD_K8);
    print_cpu_type(PERFCTR_X86_AMD_K8C);
#endif
#if defined(__powerpc__) && !defined(PPC64)
    print_cpu_type(PERFCTR_PPC_604);
    print_cpu_type(PERFCTR_PPC_604e);
    print_cpu_type(PERFCTR_PPC_750);
#endif
    return 0;
}
