#include <stdio.h>
#include <stdlib.h>
#include "libperfctr.h"
#include "arch.h"

void do_print(FILE *resfile,
	      const struct perfctr_cpu_control *cpu_control,
	      const struct perfctr_sum_ctrs *sum,
	      const struct perfctr_sum_ctrs *children)	      
{
    unsigned int nrctrs, i;

    if( cpu_control->tsc_on )
	fprintf(resfile, "tsc\t\t\t%19lld\n", sum->tsc);
    nrctrs = cpu_control->nractrs;
    for(i = 0; i < nrctrs; ++i) {
	fprintf(resfile, "\t%19lld\n", sum->pmc[i]);
    }
    if( cpu_control->ppc64.mmcr0 )
	fprintf(resfile, "mmcr0 0x%08X\n",
		cpu_control->ppc64.mmcr0);
    if( cpu_control->ppc64.mmcr1 )
	fprintf(resfile, "mmcr1 0x%016llX\n",
		(unsigned long long)cpu_control->ppc64.mmcr1);
    if( cpu_control->ppc64.mmcra )
	fprintf(resfile, "mmcra 0x%08X\n",
		cpu_control->ppc64.mmcra);
		
}

void do_arch_usage(void)
{
    fprintf(stderr, "\t--mmcr0=<value>\t\t\tValue for MMCR0\n");
    fprintf(stderr, "\t--mmcr1=<value>\t\t\tValue for MMCR1\n");
    fprintf(stderr, "\t--mmcra=<value>\t\t\tValue for MMCRA\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Syntax of event specifiers:\n");
    fprintf(stderr, "\tevent ::= @pmc\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tpmc is a decimal or hexadecimal number.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tpmc describes which CPU counter to use for this event.\n");
    fprintf(stderr, "\tBy default the events use counters 0 and up in the order listed.\n");
}

static int parse_event_spec(const char *arg, unsigned int *pmc)
{
    char *endp;
    if( arg[0] != '@' )
    	return -1;
    *pmc = strtoul(arg+1, &endp, 0);
    return endp[0] != '\0';

}

unsigned int do_event_spec(unsigned int n,
			   const char *arg,
			   struct perfctr_cpu_control *cpu_control)
{
    unsigned int spec_pmc;

    if( parse_event_spec(arg, &spec_pmc) ) {
	fprintf(stderr, "perfex: invalid event specifier: '%s'\n", arg);
	exit(1);
    }
    if (n >= ARRAY_SIZE(cpu_control->pmc_map)) {
	fprintf(stderr, "perfex: too many event specifiers\n");
	exit(1);
    }
    if( spec_pmc == (unsigned int)-1 )
	spec_pmc = n;

    cpu_control->pmc_map[n] = spec_pmc;
    cpu_control->nractrs = ++n;
    return n;
}

static int parse_value(const char *arg, unsigned long *value)
{
    char *endp;

    *value = strtoul(arg, &endp, 16);
    return endp[0] != '\0';
}

static int parse_value_ull(const char *arg, unsigned long long *value)
{
    char *endp;

    *value = strtoull(arg, &endp, 16);
    return endp[0] != '\0';
}

int do_arch_option(int ch,
		   const char *arg,
		   struct perfctr_cpu_control *cpu_control)
{
    unsigned long spec_value;
    unsigned long long spec_value_ull;

    switch( ch ) {
      case 1:
	if( parse_value(arg, &spec_value) ) {
	    fprintf(stderr, "perfex: invalid value: '%s'\n", arg);
	    exit(1);
	}
	cpu_control->ppc64.mmcr0 = spec_value;
	return 0;
      case 2:
	if( parse_value_ull(arg, &spec_value_ull) ) {
	    fprintf(stderr, "perfex: invalid value: '%s'\n", arg);
	    exit(1);
	}
	cpu_control->ppc64.mmcr1 = spec_value_ull;
	return 0;
      case 3:
	if( parse_value(arg, &spec_value) ) {
	    fprintf(stderr, "perfex: invalid value: '%s'\n", arg);
	    exit(1);
	}
	cpu_control->ppc64.mmcra = spec_value;
	return 0;
    }
    return -1;
}
