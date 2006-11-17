/*
 * File:         cpuinfo.h
 * Author:       Kevin London    
 *               london@cs.utk.edu
 * Mods:         <your name here>
 *               <your email address>
 */

#ifndef CPUINFO_H
#define CPUINFO_H

// Prcoessor definitions
#define PROC_UNKNOWN		0
#define INTEL_486			1
#define	INTEL_PENTIUM		2
#define INTEL_PPRO			3
#define INTEL_OVERDRIVE		4
#define INTEL_P2			5
#define INTEL_P2_XEON		6
#define INTEL_P3			7
#define INTEL_P3_XEON		8
#define INTEL_P4			9
#define INTEL_XEON			10
#define INTEL_CELERON		11
#define INTEL_MOBILE		12
#define AMD_486				100
#define AMD_K5				101
#define AMD_K6				102
#define AMD_K62				103
#define AMD_K63				104
#define AMD_DURON			105
#define AMD_ATHLON			106
#define AMD_OPTERON			107

struct wininfo {
	int vendor;
	unsigned int processor_id;
	unsigned int family;
	unsigned int ext_family;
	unsigned int brand_id;
	unsigned int model;
	unsigned int ext_model;
	unsigned int stepping;
	unsigned int processor_type;
	unsigned int feature_flag;
	unsigned int ncpus;
	unsigned int pagesize;
	unsigned int mhz;
	unsigned int revision;
	unsigned int nrctr;
	unsigned short int arch, proc_level, nnodes, total_cpus;
	unsigned int L1datacache_size;
	unsigned int L1datacache_assoc;
	unsigned int L1datacache_lines;
	unsigned int L1datacache_linesize;
    unsigned int L1instcache_size;
	unsigned int L1instcache_assoc;
	unsigned int L1instcache_lines;
	unsigned int L1instcache_linesize;
	unsigned int L1tracecache_size;
	unsigned int L1tracecache_assoc;
	unsigned int L1cache_sectored;
	unsigned int L2cache_size;
	unsigned int L2cache_assoc;
	unsigned int L2cache_lines;
	unsigned int L2cache_linesize;
	unsigned int L2cache_codedata;
	unsigned int L2cache_sectored;
	unsigned int L3cache_size;
    unsigned int L3cache_assoc;
	unsigned int L3cache_linesize;
    unsigned int total_phys; 
	unsigned int avail_phys;
	unsigned int total_virt;
	unsigned int avail_virt;


	char vendor_string[13];
	char model_string[40];
	// Below are features
	unsigned short int FPU;        // A floating-point unit is available
	unsigned short int TSC;        // A timp stamp counter is availabe in the processor (RDTSC)
	unsigned short int APIC;       // A local APIC unit is available
	unsigned short int MMX;        // MMX unstruction set are supported
	unsigned short int FXSAVE;     // Fast floating-point save and restore are supported (FXSAVE/FXRSTOR)
	unsigned short int DNOW_EXT;  // Extensions to the 3DNow! instruction set are supported
	unsigned short int DNOW;      // 3DNOW! instructions are supported
	unsigned short int SERIAL;    // 96 bit processor serial number is supported and enabled
	unsigned short int ACPI;      // Processor temperature can be monitored and processor performance
								  // can be modulated in predefined duty cycles under software control
	unsigned short int SSE;		  // Streaming SIMD Extension support
	unsigned short int SSE2;	  // Streaming SIMD Extension - 2 Instructions support
	unsigned short int TM;		  // Thermal Monitor automatic thermal control circuit
};

#define UNKNOWN 0
#define INTEL   1
#define AMD     2

// Macros for processor types
#define IS_UNKNOWN(hwinfo)		((hwinfo)->vendor==UNKNOWN)
#define IS_AMD(hwinfo)			((hwinfo)->vendor==AMD)
#define IS_INTEL(hwinfo)		((hwinfo)->vendor==INTEL)

// AMD processors
#define IS_AMD486(hwinfo)       ((hwinfo)->processor_id==AMD_486)
#define IS_AMDK5(hwinfo)	    ((hwinfo)->processor_id==AMD_K5)
#define IS_AMDK6(hwinfo)		((hwinfo)->processor_id==AMD_K6)
#define IS_AMDK62(hwinfo)		((hwinfo)->processor_id==AMD_K62)
#define IS_AMDK63(hwinfo)		((hwinfo)->processor_id==AMD_K63)
#define IS_AMDDURON(hwinfo)		((hwinfo)->processor_id==AMD_DURON)
#define IS_AMDATHLON(hwinfo)	((hwinfo)->processor_id==AMD_ATHLON)
#define IS_AMDOPTERON(hwinfo)	((hwinfo)->processor_id==AMD_OPTERON)
// Intel Processors
#define IS_486(hwinfo)			((hwinfo)->processor_id==INTEL_486)
#define IS_PENTIUM(hwinfo)		((hwinfo)->processor_id==INTEL_PENTIUM)
#define IS_PPRO(hwinfo)			((hwinfo)->processor_id==INTEL_PPRO)
#define IS_OVERDRIVE(hwinfo)	((hwinfo)->processor_id==INTEL_OVERDRIVE)
#define IS_P2(hwinfo)			((hwinfo)->processor_id==INTEL_P2)
#define IS_P2_XEON(hwinfo)		((hwinfo)->processor_id==INTEL_P2_XEON)
#define IS_P3(hwinfo)			((hwinfo)->processor_id==INTEL_P3)
#define IS_P3_XEON(hwinfo)		((hwinfo)->processor_id==INTEL_P3_XEON)
#define IS_P4(hwinfo)			((hwinfo)->processor_id==INTEL_P4)
#define IS_CELERON(hwinfo)		((hwinfo)->processor_id==INTEL_CELERON)
#define IS_MOBILE(hwinfo)		((hwinfo)->processor_id==INTEL_MOBILE)


int init_hwinfo( struct wininfo *);

#endif /* CPUINFO_H */
