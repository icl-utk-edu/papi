/* $Id: ppc.c,v 1.1.2.2 2010/06/08 20:48:56 mikpe Exp $
 * PPC32-specific code.
 *
 * Copyright (C) 2004-2010  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include "libperfctr.h"
#include "arch.h"

void do_print(FILE *resfile,
	      const struct perfctr_info *info,
	      const struct perfctr_cpu_control *cpu_control,
	      const struct perfctr_sum_ctrs *sum)
{
    unsigned int nrctrs, i;

    if( cpu_control->tsc_on )
	fprintf(resfile, "tsc\t\t\t%19lld\n", sum->tsc);
    nrctrs = cpu_control->nractrs;
    for(i = 0; i < nrctrs; ++i) {
	fprintf(resfile, "event 0x%08X",
		cpu_control->evntsel[i]);
	fprintf(resfile, "\t%19lld\n", sum->pmc[i]);
    }
    if( cpu_control->ppc.mmcr0 )
	fprintf(resfile, "mmcr0 0x%08X\n",
		cpu_control->ppc.mmcr0);
    if( cpu_control->ppc.mmcr2 )
	fprintf(resfile, "mmcr2 0x%08X\n",
		cpu_control->ppc.mmcr2);
}

void do_arch_usage(void)
{
    fprintf(stderr, "\t--mmcr0=<value>\t\t\tValue for MMCR0\n");
    fprintf(stderr, "\t--mmcr2=<value>\t\t\tValue for MMCR2\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Syntax of event specifiers:\n");
    fprintf(stderr, "\tevent ::= evntsel[@pmc]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tevntsel and pmc are decimal or hexadecimal numbers.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tevntsel is the primary processor-specific event selection code\n");
    fprintf(stderr, "\tto use for this counter. This field is mandatory.\n");
    fprintf(stderr, "\tEvntsel is written to a field in MMCR0 or MMCR1.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tpmc describes which CPU counter to use for this event.\n");
    fprintf(stderr, "\tBy default the events use counters 0 and up in the order listed.\n");
}

static int parse_event_spec(const char *arg, unsigned int *evntsel,
			    unsigned int *pmc)
{
    char *endp;

    *evntsel = my_strtoul(arg, &endp);
    if( endp[0] != '@' ) {
	*pmc = (unsigned int)-1;
    } else {
	arg = endp + 1;
	*pmc = my_strtoul(arg, &endp);
    }
    return endp[0] != '\0';
}

unsigned int do_event_spec(unsigned int n,
			   const char *arg,
			   struct perfctr_cpu_control *cpu_control)
{
    unsigned int spec_evntsel, spec_pmc;

    if( parse_event_spec(arg, &spec_evntsel, &spec_pmc) ) {
	fprintf(stderr, "perfex: invalid event specifier: '%s'\n", arg);
	exit(1);
    }
    if( n >= ARRAY_SIZE(cpu_control->evntsel) ) {
	fprintf(stderr, "perfex: too many event specifiers\n");
	exit(1);
    }
    if( spec_pmc == (unsigned int)-1 )
	spec_pmc = n;
    cpu_control->evntsel[n] = spec_evntsel;
    cpu_control->pmc_map[n] = spec_pmc;
    cpu_control->nractrs = ++n;
    return n;
}

static int parse_value(const char *arg, unsigned int *value)
{
    char *endp;

    *value = my_strtoul(arg, &endp);
    return endp[0] != '\0';
}

int do_arch_option(int ch,
		   const char *arg,
		   struct perfctr_cpu_control *cpu_control)
{
    unsigned int spec_value;

    switch( ch ) {
      case 1:
	if( parse_value(arg, &spec_value) ) {
	    fprintf(stderr, "perfex: invalid value: '%s'\n", arg);
	    exit(1);
	}
	cpu_control->ppc.mmcr0 = spec_value;
	return 0;
      case 2:
	if( parse_value(arg, &spec_value) ) {
	    fprintf(stderr, "perfex: invalid value: '%s'\n", arg);
	    exit(1);
	}
	cpu_control->ppc.mmcr2 = spec_value;
	return 0;
    }
    return -1;
}
