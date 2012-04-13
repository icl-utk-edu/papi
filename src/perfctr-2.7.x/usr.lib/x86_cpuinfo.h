/* $Id: x86_cpuinfo.h,v 1.1 2004/05/22 20:59:58 mikpe Exp $
 * Copyright (C) 2004  Mikael Pettersson
 */

struct cpuinfo {
    unsigned int vendor;
    unsigned int signature;	/* cpuid[1].eax */
    unsigned int features;	/* cpuid[1].edx */
};

#define X86_VENDOR_UNKNOWN	0
#define X86_VENDOR_INTEL	1
#define X86_VENDOR_AMD		2
#define X86_VENDOR_CYRIX	3
#define X86_VENDOR_CENTAUR	4

#define X86_FEATURE_TSC		4
#define X86_FEATURE_MSR		5
#define X86_FEATURE_MMX		23

#define cpu_type(cpuinfo)	(((cpuinfo)->signature >> 12) & 3)
#define cpu_family(cpuinfo)	(((cpuinfo)->signature >> 8) & 0xF)
#define cpu_model(cpuinfo)	(((cpuinfo)->signature >> 4) & 0xF)
#define cpu_stepping(cpuinfo)	((cpuinfo)->signature & 0xF)

#define cpu_has(cpuinfo, bit)	((cpuinfo)->features & (1<<(bit)))

extern void identify_cpu(struct cpuinfo*);
