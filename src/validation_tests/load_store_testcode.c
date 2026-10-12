#include "testcode.h"

/* Execute n stores */
int execute_stores(long long n) {

#if defined(__aarch64__)

	__asm(  ".data\n"
		".balign 8\n"
		"stvar: .xword 1 /* stvar in memory */\n"
		".text\n"
		"	ldr x2, =stvar /* address of stvar */\n"
		"	mov x4, %x0\n"
		"	mov x1, #0\n"
		"str_loop:\n"
		"	str x1, [x2] /* store into stvar */\n"
		"	add x1, x1, #1\n"
		"	cmp x1, x4\n"
		"	bne str_loop\n"
		:
		: "r" (n)
		: "cc", "x1", "x2", "x4" /* clobbered */
	);

	return 0;

#endif
	(void) n;
	return CODE_UNIMPLEMENTED;

}

/* Execute n loads */
int execute_loads(long long n) {

#if defined(__aarch64__)

	__asm(  ".data\n"
		".balign 8\n"
		"ldvar: .xword 1 /* ldvar in memory */\n"
		".text\n"
		"	ldr x2, =ldvar /* address of ldvar */\n"
		"	mov x4, %x0\n"
		"	mov x1, #0\n"
		"ldr_loop:\n"
		"	ldr x3, [x2] /* load from ldvar */\n"
		"	add x1, x1, x3\n"
		"	cmp x1, x4\n"
		"	bne ldr_loop\n"
		:
		: "r" (n)
		: "cc", "x1", "x2", "x4" /* clobbered */

	);

	return 0;

#endif
	(void) n;
	return CODE_UNIMPLEMENTED;

}
