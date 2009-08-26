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

__inline void _wrmsr(uint32_t msr, uint32_t lo, uint32_t hi)
{
	__asm
  {
    mov ecx, msr
	  mov eax, lo
    mov edx, hi
	  wrmsr
  }
}

/* #if NTDDI_VERSION < NTIDDI_WINXP */
/* Its 2009, we're going to use the intrinsics newer vc++ provides */
#ifndef _WIN64
__inline void __writecr4(unsigned x)
{
	__asm
  {
    mov eax, x
    _emit 0x0F
    _emit 0x22
    _emit 0xE0
  }
}

static __inline unsigned int __readcr4()
{
  __asm
  {
    _emit 0x0F
    _emit 0x20
    _emit 0xE0
  }
  // eax is the return value
}


// why can't we use __cpuid in intrin.h for device drivers?
void __cpuid(uint32_t *regs, uint32_t command)
{
  __asm
  {
    mov eax, command
    cpuid
    mov esi, regs
    mov [esi], eax
    mov [esi + 4], ebx
    mov [esi + 8], ecx
    mov [esi + 12], edx
  }
}
#endif

static __inline void set_cr4_pce()
{
   __writecr4(__readcr4() | X86_CR4_PCE);
}

static __inline void clear_cr4_pce()
{
   __writecr4(__readcr4() & ~X86_CR4_PCE);
}

