/* $Id: x86.c,v 1.1.2.6 2010/06/08 20:48:56 mikpe Exp $
 * x86-specific code.
 *
 * Copyright (C) 1999-2010  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include "libperfctr.h"
#include "arch.h"

static int info_is_p4(const struct perfctr_info *info)
{
    switch (info->cpu_type) {
#if !defined(__x86_64__)
    case PERFCTR_X86_INTEL_P4:
    case PERFCTR_X86_INTEL_P4M2:
#endif
    case PERFCTR_X86_INTEL_P4M3:
	return 1;
    default:
	return 0;
    }
}

void do_print(FILE *resfile,
	      const struct perfctr_info *info,
	      const struct perfctr_cpu_control *cpu_control,
	      const struct perfctr_sum_ctrs *sum)
{
    unsigned int nrctrs, i;
    int is_p4;

    is_p4 = info_is_p4(info);
    if (cpu_control->tsc_on)
	fprintf(resfile, "tsc\t\t\t\t%19lld\n", sum->tsc);
    nrctrs = cpu_control->nractrs;
    for(i = 0; i < nrctrs; ++i) {
	fprintf(resfile, "event 0x%08X",
		cpu_control->evntsel[i]);
	/* p4.escr[] overlaps evntsel_high[], but the output syntax
	   is the same regardless of whether is_p4 is true or not */
	if (cpu_control->p4.escr[i])
	    fprintf(resfile, "/0x%08X",
		    cpu_control->p4.escr[i]);
	if (cpu_control->pmc_map[i] >= 18)
	    fprintf(resfile, "@0x%08x\t", cpu_control->pmc_map[i]);
	else
	    fprintf(resfile, "@%u\t\t", cpu_control->pmc_map[i]);
	fprintf(resfile, "%19lld\n", sum->pmc[i]);
    }
    /* p4.pebs_{enable,matrix_vert} overlap nhlm.offcore_rsp[],
       and we want to adjust the output based on is_p4 */
    if (cpu_control->p4.pebs_enable)
	fprintf(resfile, "%s 0x%08X\n",
		is_p4 ? "PEBS_ENABLE" : "NHLM_OFFCORE_RSP_0",
		cpu_control->p4.pebs_enable);
    if (cpu_control->p4.pebs_matrix_vert)
	fprintf(resfile, "%s 0x%08X\n",
		is_p4 ? "PEBS_MATRIX_VERT" : "NHLM_OFFCORE_RSP_1",
		cpu_control->p4.pebs_matrix_vert);
}

void do_arch_usage(void)
{
    fprintf(stderr, "\t--nhlm_offcore_rsp_0=<value>\tValue for OFFCORE_RSP_0 (Nehalem only)\n");
    fprintf(stderr, "\t--nhlm_offcore_rsp_1=<value>\tValue for OFFCORE_RSP_1 (Nehalem only)\n");
    fprintf(stderr, "\t--p4pe=<value>\t\t\tValue for PEBS_ENABLE (P4 only)\n");
    fprintf(stderr, "\t--p4_pebs_enable=<value>\tSame as --p4pe=<value>\n");
    fprintf(stderr, "\t--p4pmv=<value>\t\t\tValue for PEBS_MATRIX_VERT (P4 only)\n");
    fprintf(stderr, "\t--p4_pebs_matrix_vert=<value>\tSame as --p4pmv=<value>\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Syntax of event specifiers:\n");
    fprintf(stderr, "\tevent ::= evntsel[/evntsel2][@pmc]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tevntsel, evntsel2, and pmc are decimal or hexadecimal numbers.\n");
    fprintf(stderr, "\t/ and @ are literal characters. [...] denotes an optional field.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tevntsel is the primary processor-specific event selection code\n");
    fprintf(stderr, "\tto use for this counter. This field is mandatory.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tevntsel2 provides auxiliary event selection code to use for this\n");
    fprintf(stderr, "\tcounter. Currently only used for P4 and AMD Family 10h, on other\n");
    fprintf(stderr, "\tprocessors this field should be omitted.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tpmc describes which CPU counter to use for this event.\n");
    fprintf(stderr, "\tBy default the events use counters 0 and up in the order listed.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tOn Core2, the fixed-function performance counters are numbered\n");
    fprintf(stderr, "\t0x40000000 to 0x40000002. To use them, explicit counter assignment\n");
    fprintf(stderr, "\tvia the @pmc notation is mandatory.\n");
    fprintf(stderr, "\tOn Core2, a fixed-function performance counter has an evntsel\n");
    fprintf(stderr, "\tjust like a programmable performance counter has, but only the\n");
    fprintf(stderr, "\tCPL (bits 16 and 17) and Enable (bit 22) fields are relevant.\n");
    fprintf(stderr, "\t(The INT field (bit 20) is also honoured, but perfex cannot set\n");
    fprintf(stderr, "\tup interrupt-mode counting, so it should not be specified.)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tAtom is similar to Core2, but appears to only support a single\n");
    fprintf(stderr, "\t(the first) fixed-function counter.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tOn AMD Family 10h, evntsel is written to the low 32 bits of the\n");
    fprintf(stderr, "\tcounter's EVNTSEL register, and evntsel2 is written to the high\n");
    fprintf(stderr, "\t32 bits of that register. Only a few events require evntsel2.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tOn a P4, evntsel is written to the counter's CCCR register.\n");
    fprintf(stderr, "\tOn a P4, evntsel2 is written to the counter's ESCR register.\n");
    fprintf(stderr, "\tOn P4, each event is compatible with only a small subset of the\n");
    fprintf(stderr, "\tcounters, and explicit counter assignment via @pmc is mandatory.\n");
    fprintf(stderr, "\tOn P4, bit 31 should be set in pmc to enable 'fast rdpmc'.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tVIA C3 accepts a single event only, but it must use counter 1.\n");
}

static int parse_event_spec(const char *arg, unsigned int *evntsel,
			    unsigned int *escr, unsigned int *pmc)
{
    char *endp;

    *evntsel = my_strtoul(arg, &endp);
    if (endp[0] != '/') {
	*escr = 0;
    } else {
	arg = endp + 1;
	*escr = my_strtoul(arg, &endp);
    }
    if (endp[0] != '@') {
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
    unsigned int spec_evntsel, spec_escr, spec_pmc;

    if (parse_event_spec(arg, &spec_evntsel, &spec_escr, &spec_pmc)) {
	fprintf(stderr, "perfex: invalid event specifier: '%s'\n", arg);
	exit(1);
    }
    if (n >= ARRAY_SIZE(cpu_control->evntsel)) {
	fprintf(stderr, "perfex: too many event specifiers\n");
	exit(1);
    }
    if (spec_pmc == (unsigned int)-1)
	spec_pmc = n;
    cpu_control->evntsel[n] = spec_evntsel;
    cpu_control->p4.escr[n] = spec_escr;
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

    switch (ch) {
      case 1:
	if (parse_value(arg, &spec_value)) {
	    fprintf(stderr, "perfex: invalid value: '%s'\n", arg);
	    exit(1);
	}
	cpu_control->p4.pebs_enable = spec_value;
	return 0;
      case 2:
	if (parse_value(arg, &spec_value)) {
	    fprintf(stderr, "perfex: invalid value: '%s'\n", arg);
	    exit(1);
	}
	cpu_control->p4.pebs_matrix_vert = spec_value;
	return 0;
    }
    return -1;
}
