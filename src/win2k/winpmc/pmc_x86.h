/* $Id$
 * Performance-monitoring counters driver.
 * x86-specific compatibility definitions
 *
 */


// Machine Specific Register definitions

/* Intel P5, Cyrix 6x86MX/MII/III, Centaur WinChip C6/2/3 */
#define MSR_P5_CESR		0x11
#define MSR_P5_CTR0		0x12
#define MSR_P5_CTR1		0x13

/* Intel P6 */
#define MSR_P6_PERFCTR0		0xC1
#define MSR_P6_PERFCTR1		0xC2
#define MSR_P6_EVNTSEL0		0x186
#define MSR_P6_EVNTSEL1		0x187

/* AMD K7 Athlon */
#define MSR_K7_EVNTSEL0		0xC0010000	/* .. 0xC0010003 */
#define MSR_K7_PERFCTR0		0xC0010004	/* .. 0xC0010007 */

// Control Register 4 definitions

#define X86_CR4_TSD	0x0004	// Time Stamp Disable bit
#define X86_CR4_PCE	0x0100	// Performance Counter (RDPMC) enable bit

// define some needed assembly instructions...
// For some reason, the Microsoft Visual C++ in-line assembler doesn't appear to support
// access to the CR4 control register, even though CR0, CR2 and CR3 are supported. Go Figure...

// Fast Forward to 2006.
// The AMD64 extended compiler in DDK 3790 doesn't even support in-line assembly.
// BUT... it *does* have a series of intrinsics to emulate the assembly needed to  
// read and write these registers! Changes need to be made to always use 64 bit values.

#pragma message("Testing pragma messages...\n")

#if defined (_WIN64)
#pragma message("_WIN64 is defined!\n")

#define _wrmsr(msr,low, hi) \
   __writemsr(msr, low);

/*
static __inline void set_cr4_pce(void)
{
   __writecr4(__readcr4() | X86_CR4_PCE);
}

static __inline void clear_cr4_pce(void)
{
   __writecr4(__readcr4() & ~X86_CR4_PCE);
}
*/
//extern void set_cr4_pce(void);
//extern void clear_cr4_pce(void);

#else
//	__asm mov eax, cr4
#define MOV_EAX_CR4 \
	__asm _emit 0x0F \
	__asm _emit 0x20 \
	__asm _emit 0xE0

//	__asm mov cr4, eax
#define MOV_CR4_EAX \
	__asm _emit 0x0F \
	__asm _emit 0x22 \
	__asm _emit 0xE0

#define rdpmcl(ctr,low) \
	__asm mov ecx, ctr	\
	__asm rdpmc			\
	__asm mov low, eax

#define rdtscl(low)		\
	__asm rdtsc			\
	__asm mov low, eax

#define _rdmsr(msr,low, hi) \
	__asm mov ecx, msr	\
	__asm rdmsr			\
	__asm mov low, eax	\
	__asm mov hi,  edx	\

#define rdmsrl(msr,low) \
	__asm mov ecx, msr	\
	__asm rdmsr			\
	__asm mov low, eax

#define _wrmsr(msr,low, hi) \
	__asm mov ecx, msr	\
	__asm mov eax, low	\
	__asm mov edx, hi	\
	__asm wrmsr

//static __inline void _write_cr4(unsigned int x)
static __inline void __writecr4(unsigned int x)
{
	__asm mov eax, x
	MOV_CR4_EAX
}

//static __inline unsigned int _read_cr4(void)
static __inline unsigned int __readcr4(void)
{
	MOV_EAX_CR4	// eax is the return value
}
/*
static __inline void set_cr4_pce(void)
{
   _write_cr4(_read_cr4() | X86_CR4_PCE);
}

static __inline void clear_cr4_pce(void)
{
   _write_cr4(_read_cr4() & ~X86_CR4_PCE);
}
*/

#endif

static __inline void set_cr4_pce(void)
{
   __writecr4(__readcr4() | X86_CR4_PCE);
}

static __inline void clear_cr4_pce(void)
{
   __writecr4(__readcr4() & ~X86_CR4_PCE);
}


struct cpuidVals
{
      unsigned long a, b, c, d;
};

extern void GetCPUID(unsigned long id, struct cpuidVals *vals);
