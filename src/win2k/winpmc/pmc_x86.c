/* 
 * x86 performance-monitoring counters driver support routines.
 *
 * Modified for Windows NT/2000 by dkterpstra 05/2001
 */
#include <ntddk.h>
#include <stdio.h>
#include "pmc_kernel.h"
#include "pmc_x86.h"

/****************************************************************
 *																*
 * Driver procedures.											*
 * NOTE: Each write procedure also clears the affected 40-bit	*
 *	event counters by writing a 64-bit 0. No attempt is made	*
 *	to protect high order 'unused' bits in the counter. This	*
 *	is *probably* safe, but has not been rigorously confirmed.	*
 *																*
 ****************************************************************/

static int p5_check_control(const struct pmc_control *control)
{
	/* protect reserved and pin control bits */
	if( control->evntsel[0] & 0xFE00FE00 )
		return STATUS_P5_RESERVED;
	/* CTR1 is on if its CPL field is non-zero */
	if( control->evntsel[0] & 0x00C00000 )
		return 3;
	/* CTR0 is on if its CPL field is non-zero */
	if( control->evntsel[0] & 0x000000C0 )
		return 2;
	/* Only TSC is on. */
	return 1;
}

static int p5_write_control(int nrctrs, const struct pmc_control *control)
{
	unsigned evntsel;

	if( nrctrs <= 1 )	/* no evntsel to write if only TSC is on */
		return STATUS_SUCCESS;
	evntsel = control->evntsel[0];
	_wrmsr(MSR_P5_CESR, evntsel, 0);
	_wrmsr(MSR_P5_CTR0, 0, 0);			// clear counter 0
	_wrmsr(MSR_P5_CTR1, 0, 0);			// clear counter 1
	return STATUS_SUCCESS;
}

static int p6_check_control(const struct pmc_control *control)
{
	/* protect reserved, interrupt control, and pin control bits */
	if( control->evntsel[0] & 0x00380000 )
		return STATUS_P6_RESERVED0;
	if( control->evntsel[1] & 0x00780000 )
		return STATUS_P6_RESERVED1;
	/* check global enable bit */
	if( control->evntsel[0] & 0x00400000 ) {
		/* check CPL field */
		if( control->evntsel[1] & 0x00030000 )
			return 3;
		if( control->evntsel[0] & 0x00030000 )
			return 2;
	}
	return 1;
}

static int p6_write_control(int nrctrs, struct pmc_control *control)
{
	unsigned evntsel;

	if( nrctrs == 3 ) {
		evntsel = control->evntsel[1];
		_wrmsr(MSR_P6_EVNTSEL1, evntsel, 0);
		_wrmsr(MSR_P6_PERFCTR1, 0, 0);		// clear the counter
	}	
	if( nrctrs >= 2 ) {
		evntsel = control->evntsel[0];
		_wrmsr(MSR_P6_EVNTSEL0, evntsel, 0);
		_wrmsr(MSR_P6_PERFCTR0, 0, 0);		// clear the counter
	}
	// DBG: read the registers and swap their contents as proof they have been written
//		rdmsrl(MSR_P6_EVNTSEL1, evntsel);
//		control->evntsel[0] = evntsel;
//		rdmsrl(MSR_P6_EVNTSEL0, evntsel);
//		control->evntsel[1] = evntsel;

	return STATUS_SUCCESS;
}

static int k7_check_control(const struct pmc_control *control)
{
	int i, last_pmc_on;

	last_pmc_on = -1;
	for(i = 0; i < 4; ++i) {
		/* protect reserved, interrupt control, and pin control bits */
		if( control->evntsel[i] & 0x00380000 ) {
			return STATUS_K7_RESERVED;
		}
		/* check enable bit and CPL field */
		if( !(control->evntsel[i] & 0x00400000) ||
		  !(control->evntsel[i] & 0x00030000) ) {
		    /* not enabled; disallow random junk */
		    // this causes an error exit when only the domain bit is set
		    //if( control->evntsel[i] != 0 )
		    //  return STATUS_K7_DISABLED;
			  break;
		}
		last_pmc_on = i;
	}
	return last_pmc_on + 2;
}

static int k7_write_control(int nrctrs, struct pmc_control *control)
{
	int i;
	unsigned evntsel;

	// clear all four counters
	_wrmsr(MSR_K7_PERFCTR0+0, 0, 0);
	_wrmsr(MSR_K7_PERFCTR0+1, 0, 0);
	_wrmsr(MSR_K7_PERFCTR0+2, 0, 0);
	_wrmsr(MSR_K7_PERFCTR0+3, 0, 0);

	// clear the unused counter control registers
	evntsel = 0;
	switch(nrctrs) {
		case 2:
			_wrmsr(MSR_K7_EVNTSEL0+3, evntsel, 0);
		case 3:
			_wrmsr(MSR_K7_EVNTSEL0+2, evntsel, 0);
		case 4:
			_wrmsr(MSR_K7_EVNTSEL0+1, evntsel, 0);
	}

	// program the counters in use
	switch(nrctrs) {
		case 5:
			evntsel = control->evntsel[3];
			_wrmsr(MSR_K7_EVNTSEL0+3, evntsel, 0);
		case 4:
			evntsel = control->evntsel[2];
			_wrmsr(MSR_K7_EVNTSEL0+2, evntsel, 0);
		case 3:
			evntsel = control->evntsel[1];
			_wrmsr(MSR_K7_EVNTSEL0+1, evntsel, 0);
		case 2:
			evntsel = control->evntsel[0];
			_wrmsr(MSR_K7_EVNTSEL0+0, evntsel, 0);
	}
	return STATUS_SUCCESS;
}

static int (*check_control)(const struct pmc_control *control);
static int pmc_cpu_check_control(const struct pmc_control *control)
{
	return check_control(control);
}

static int (*write_control)(int nrctrs, struct pmc_control *control);
static int pmc_cpu_write_control(int nrctrs, struct pmc_control *control)
{
	return(write_control(nrctrs, control));
}

/****************************************************************
 *																*
 * Processor detection and initialization procedures.			*
 *																*
 ****************************************************************/

static int intel_init(int family, int stepping, int model)
{
	switch(family) {
	case 5:
		write_control = p5_write_control;
		check_control = p5_check_control;
		return STATUS_SUCCESS;
	case 6:
	       /* P-Pro with SMM support will enter halt state if
                * the PCE bit is set and a RDPMC is issued    -KSL
		*/
	       if ( model == 1 && stepping == 9 ) 
		    return STATUS_NO_INTEL_INIT;
		write_control = p6_write_control;
		check_control = p6_check_control;
		return STATUS_SUCCESS;
	}
	return STATUS_NO_INTEL_INIT;
}

static int amd_init(int family)
{
	switch(family) {
	case 6:	   /* K7 Athlon. Model 1 does not have a local APIC. */
	case 15:	/* K8 Opteron. Uses same control routines as Athlon. */
		write_control = k7_write_control;
		check_control = k7_check_control;
		return STATUS_SUCCESS;
	}
	return STATUS_NO_AMD_INIT;
}

/****************************************************************
 *																*
 * cpuid vendor string and features procedure.					*
 *																*
 ****************************************************************/

// Returns zero if no cpuid instruction; else returns 1
// Returns completed pmc_info struct on success 
static int cpu_id(struct pmc_info *info) {
   unsigned int version, features;
	char v[12];

   // NOTE: Earlier versions of this routine checked for the existence of
   // the cpuid instruction before proceeding.
   // It's 2006. I think we can assume that any processor this is running on
   // is post-486 and will thus have the cpuid instruction.

	// Get the Vendor String, features, and family/model/stepping info
   struct cpuidVals vals;
   GetCPUID(0, &vals);
   *(unsigned long *)(&v[0]) = vals.b;
   *(unsigned long *)(&v[4]) = vals.d;
   *(unsigned long *)(&v[8]) = vals.c;
   GetCPUID(1, &vals);
   features = vals.d;
   version = vals.a;

	strncpy(info->vendor, v, 12);
	info->features = features;
	info->family = (version >> 8) & 0x0000000F;		// bits 11 - 8
	info->model  = (version >> 4) & 0x0000000F;             // Bits  7 - 4
	info->stepping=(version)      & 0x0000000F;             // bits  3 - 0
	return 1;
}


#ifndef _WIN64
void GetCPUID(unsigned long id, struct cpuidVals *vals)
{
   unsigned long a, b, c, d;

   __asm {
		mov eax, id
		cpuid
      mov a, eax
		mov b, ebx
		mov c, ecx
		mov d, edx
	}
   vals->a = a;
   vals->b = b;
   vals->c = c;
   vals->d = d;
}
#endif

/****************************************************************
 *																*
 * system visible routines.										*
 *																*
 ****************************************************************/
#define MSR 0x00000020	// bit 5
#define TSC 0x00000010	// bit 4
#define MMX 0x00800000	// bit 23

// returns 0 for success or negative for failure
//int kern_pmc_init(struct pmc_info *info)
int kern_pmc_init(void)
{
	int status = STATUS_UNKNOWN_CPU_INIT;
	struct pmc_info info;
   ULONG64 cr4;

	if (!cpu_id(&info)) return STATUS_NO_CPUID;

	// we need to at least support MSR registers, RDTSC, & RDPMC
	if(!(info.features & MSR)) return STATUS_NO_MSR;
	if(!(info.features & TSC)) return STATUS_NO_TSC;
	if(!(info.features & MMX)) return STATUS_NO_MMX;	// assume MMX tracks RDPMC

	if (!strncmp(info.vendor, "GenuineIntel", 12)) status = intel_init(info.family,info.stepping,info.model);
	else if (!strncmp(info.vendor, "AuthenticAMD", 12)) status = amd_init(info.family);
// we really don't need to claim support for Cyrix, do we?
// else if (!strncmp(info.vendor, "CyrixInstead", 12)) status = cyrix_init(info.family);

	if(status == STATUS_SUCCESS) set_cr4_pce();

	return status;
}

// returns 0 for success or negative for failure
int kern_pmc_info(struct pmc_info *info)
{
	int status = STATUS_SUCCESS;

	if (!cpu_id(info)) return STATUS_NO_CPUID;

	return status;
}

// returns negative for failure; 
// 0 or positive for # of enabled counters
int kern_pmc_control(struct pmc_control *control)
{
	int nrctrs, status;

	nrctrs = pmc_cpu_check_control(control);
	if(nrctrs > 0) status = pmc_cpu_write_control(nrctrs, control);
	if (status < 0) return status;
	else return nrctrs;
}

void kern_pmc_exit(void)
{
	clear_cr4_pce();
}

