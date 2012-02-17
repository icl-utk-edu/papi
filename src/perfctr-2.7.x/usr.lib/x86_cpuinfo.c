/* $Id: x86_cpuinfo.c,v 1.1 2004/05/22 20:59:58 mikpe Exp $
 * Copyright (C) 2004  Mikael Pettersson
 */
#include <string.h>
#include "x86_cpuinfo.h"

struct cpuid { /* The field order must not be changed. */
    unsigned int eax, ebx, edx, ecx;
};

#ifdef __x86_64__
static void get_cpuid(unsigned int op, struct cpuid *cpuid)
{
    __asm__("cpuid"
	    : "=a"(cpuid->eax),
	      "=b"(cpuid->ebx),
	      "=c"(cpuid->ecx),
	      "=d"(cpuid->edx)
	    : "0"(op));
}
#else
/* Many versions of gcc fail on x86 if an asm() clobbers %ebx when
   -fPIC is specified. So we do the cpuid in hand-written assembly code. */
extern void get_cpuid(unsigned int, struct cpuid*); /* x86_cpuid.S */
#endif

static const struct {
    const char vendor_string[12];
    unsigned int vendor_code;
} vendors[] = {
    { "GenuineIntel", X86_VENDOR_INTEL },
    { "AuthenticAMD", X86_VENDOR_AMD },
    { "CyrixInstead", X86_VENDOR_CYRIX },
    { "CentaurHauls", X86_VENDOR_CENTAUR },
};

static unsigned int check_vendor(const char cpuid_vendor[12])
{
    int i;

    for(i = 0; i < sizeof vendors / sizeof(vendors[0]); ++i)
        if (memcmp(cpuid_vendor, vendors[i].vendor_string, 12) == 0)
            return vendors[i].vendor_code;
    return X86_VENDOR_UNKNOWN;
}

void identify_cpu(struct cpuinfo *cpuinfo)
{
    struct cpuid cpuid[2];

    /* Skip EFLAGS.ID check. We will only get here if
       the kernel has created a perfctr state for us,
       and that will never happen on pre-CPUID CPUs. */

    get_cpuid(0, &cpuid[0]);

    /* Quirk for Intel A-step Pentium. */
    if ((cpuid[0].eax & 0xFFFFFF00) == 0x0500) {
	cpuinfo->vendor = X86_VENDOR_INTEL;
	cpuinfo->signature = cpuid[0].eax;
	cpuinfo->features = 0x1BF; /* CX8,MCE,MSR,TSC,PSE,DE,VME,FPU */
	return;
    }

    cpuinfo->vendor = check_vendor((const char*)&cpuid[0].ebx);

    if (cpuid[0].eax == 0) {
	cpuinfo->signature = 0;
	cpuinfo->features = 0;
    } else {
	get_cpuid(1, &cpuid[1]);
	cpuinfo->signature = cpuid[1].eax;
	cpuinfo->features = cpuid[1].edx;
    }
}
