#include <sys/asm_linkage.h>
	! #include "solaris-ultra.h"

	! These functions blatantly stolen from perfmon
	! The author of the package "perfmon" is Richard J. Enbody
	! and the home page for "perfmon" is
	! http://www.cps.msu.edu/~enbody/perfmon/index.html

	!
	! extern void cpu_sync(void);
	!
	! Make sure all instructinos and memory references before us
	! have been completed.
	.global cpu_sync
	ENTRY(cpu_sync)
	membar	#Sync		! Wait for all outstanding things to finish
	retl			! Return to the caller
	  nop			! Delay slot
	SET_SIZE(cpu_sync)

	!
	! extern unsigned long long get_tick(void)
	!
	! Read the tick register and return it
	.global get_tick
	ENTRY(get_tick)
	rd	%tick, %o0	! Get the current value of TICK
	clruw   %o0, %o1	! put the lower 32 bits into %o1
	retl			! Return to the caller
	  srlx  %o0, 32, %o0    ! put the upper 32 bits into %o0
	SET_SIZE(get_tick)

